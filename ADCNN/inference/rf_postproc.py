"""Stage-2 RandomForest reranker — training, scoring, and persistence.

Stage 1 is the v7 NN (``predict`` + ``candidates``); stage 2 reranks those candidates
with a RandomForest over ``RF_FEATURES_V2`` (computed in ``features.py``):
  - ``apply_rf_v2`` scores candidates and applies ``DEFAULT_THR``.
  - ``build_rf_postproc_v2`` / ``train_rf_v2`` fit the model; ``train_rf_from_val`` (in
    ``rf_train.py``) is the leakage-safe production entry point.
  - ``label_candidates_by_injection_overlap`` provides training labels from injections.
  - ``load_rf`` / ``save_rf`` persist the fitted forest (shipped at ``models/rf_postproc.pkl``).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ADCNN.inference.candidates import (
    extract_candidates,
    CandidateExtractorConfig,
)
from ADCNN.inference.matched_filter import (
    panel_mad_sigma,
    matched_filter_for_nn_candidates,
)

from ADCNN.inference.features import compute_v2_features, RF_FEATURES_V2, _to_panel_dict



DEFAULT_THR = 0.50  # hard-neg fine-tuned v7 + real-residual-hardened RF: the
# model is threshold-flat (synthetic test_5sigma combined-TP 859->858 across
# 0.10..0.50) so operate high — 0.50 gives cTP 858 @ cFP 6848, beating the
# pre-finetune 0.10 point (838 @ 8876) on BOTH axes and ~halving genuine FP on
# real diffims. (Was 0.10 for the pre-finetune v2-relabel RF.)


def apply_rf_v2(cand_df, rf, *, thr: float | None = None, score_col: str = "score_rf"):
    """Score `cand_df` with a trained RF; optionally filter at `thr`.

    Returns a new DataFrame with `score_col` set (and rows below `thr` removed
    when `thr` is not None).
    """
    X = cand_df[list(RF_FEATURES_V2)].fillna(0.0).to_numpy(dtype=np.float32)
    scores = rf.predict_proba(X)[:, 1]
    out = cand_df.copy()
    out[score_col] = scores
    if thr is not None:
        out = out[out[score_col] >= thr].reset_index(drop=True)
    return out


def label_candidates_by_injection_overlap(
    cand_df,
    catalog,
    panel_probs,
    *,
    psf_width: int = 40,
) -> np.ndarray:
    """Return (N,) int8 label aligned with cand_df: 1 iff the candidate's
    connected component overlaps any catalog injection trail.

    This mirrors `objectwise_confusion`'s matching exactly, so labels and
    eval metric agree — important when training a candidate-level ranker.

    Args:
        cand_df: candidate DataFrame from `compute_v2_features`.
        catalog: pandas DataFrame with columns image_id, x, y, beta, trail_length.
        panel_probs: (N, H, W) probability array OR {pid: (H, W) array}.
        psf_width: PSF width passed into `draw_one_line` for the trail mask.
    """
    from ADCNN.utils.helpers import draw_one_line  # local import (cv2 optional)
    from ADCNN.utils.angle_utils import deg2rad as _deg2rad

    probs_dict = _to_panel_dict(panel_probs)
    pid_sample = next(iter(probs_dict.values()))
    H, W = pid_sample.shape
    half_psf = psf_width // 2
    structure = ndi.generate_binary_structure(2, 2)

    # Build per-panel injection coverage mask
    inj_mask_by_pid = {}
    for pid, sub in catalog.groupby("image_id"):
        pid = int(pid)
        if pid not in probs_dict:
            continue
        canvas = np.zeros((H, W), dtype=np.uint8)
        for _, row in sub.iterrows():
            x = float(row["x"]); y = float(row["y"])
            beta = float(row["beta"]); L = float(row["trail_length"])
            pad = half_psf + 4
            beta_rad = _deg2rad(beta)
            dx = abs(math.cos(beta_rad)) * L
            dy = abs(math.sin(beta_rad)) * L
            x0 = int(max(0, math.floor(x - dx - pad)))
            x1 = int(min(W, math.ceil(x + dx + pad)))
            y0 = int(max(0, math.floor(y - dy - pad)))
            y1 = int(min(H, math.ceil(y + dy + pad)))
            if x1 <= x0 or y1 <= y0:
                continue
            roi_h = y1 - y0; roi_w = x1 - x0
            try:
                m = draw_one_line(
                    np.zeros((roi_h, roi_w), dtype=np.uint8),
                    (x - x0, y - y0), beta, L,
                    true_value=1, line_thickness=half_psf,
                )
                canvas[y0:y1, x0:x1] |= m.astype(np.uint8)
            except Exception:
                cy = int(y); cx = int(x)
                if 0 <= cy < H and 0 <= cx < W:
                    canvas[cy, cx] = 1
        inj_mask_by_pid[pid] = canvas.astype(bool)

    labels = np.zeros(len(cand_df), dtype=np.int8)
    for pid, idxs in cand_df.groupby("panel_id").indices.items():
        pid = int(pid)
        if pid not in probs_dict:
            continue
        inj_mask = inj_mask_by_pid.get(pid)
        if inj_mask is None or not inj_mask.any():
            continue
        sub = cand_df.iloc[idxs]
        panel = probs_dict[pid].astype(np.float32, copy=False)
        eff = float(sub["effective_t_low"].iloc[0])
        cc_labels, _ = ndi.label(panel > eff, structure=structure)
        intersect = set(int(v) for v in np.unique(cc_labels[inj_mask]) if v > 0)
        cids = sub["candidate_id"].astype(int).to_numpy()
        for j, cid in zip(idxs, cids):
            if int(cid) in intersect:
                labels[j] = 1
    return labels


def train_rf_v2(
    cand_df,
    *,
    labels: np.ndarray | None = None,
    n_estimators: int = 500, max_depth: int = 14, min_samples_leaf: int = 5,
    n_jobs: int = 32, random_state: int = 0,
    informational_fp_max_overlap: float = 0.5,
):
    """Train a class-balanced RandomForest on candidates with labels.

    `labels`: optional (N,) array of {0, 1} labels for each cand_df row. When
        provided, used as-is and the training pool excludes negatives that
        overlap LSST stack footprints (`frac_real_label_overlap >=
        informational_fp_max_overlap`) — those are ignored at eval time and
        shouldn't pollute the classifier.

        Recommended source: `label_candidates_by_injection_overlap(...)`,
        which matches `objectwise_confusion`'s definition. The shipped
        `rf_postproc_v2.pkl` was trained this way.

    When `labels` is None, falls back to the legacy iter02-style
    `matched_injection_id >= 0` labels — DO NOT use for new training; it
    suffers from label noise where co-detecting cands are mislabeled
    negative.

    Returns the fitted classifier.
    """
    from sklearn.ensemble import RandomForestClassifier
    if labels is None:
        labels = (cand_df["matched_injection_id"] >= 0).astype(int).to_numpy()
    labels = np.asarray(labels, dtype=np.int8)
    fp_mask = ((labels == 0) &
               (cand_df["frac_real_label_overlap"].to_numpy()
                < informational_fp_max_overlap))
    pool_mask = (labels == 1) | fp_mask
    X = cand_df.loc[pool_mask, list(RF_FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
    y = labels[pool_mask]
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced",
        n_jobs=n_jobs, random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def materialize_label_mask_v2(cand_df, panel_probs_dict, shape):
    """Build an (N, H, W) int32 label mask from a (possibly filtered) cand_df.
    Mirrors the notebook helper but lives in ADCNN so it can be imported.
    """
    out = np.zeros(shape, dtype=np.int32)
    if len(cand_df) == 0:
        return out
    structure = ndi.generate_binary_structure(2, 2)
    for pid, sub in cand_df.groupby("panel_id"):
        pid = int(pid)
        if pid not in panel_probs_dict:
            continue
        panel = panel_probs_dict[pid]
        effective_t = float(sub["effective_t_low"].iloc[0])
        labels, n_lab = ndi.label(panel > effective_t, structure=structure)
        keep = np.zeros(n_lab + 1, dtype=bool)
        cids = sub["candidate_id"].astype(int).to_numpy()
        cids = cids[(cids > 0) & (cids <= n_lab)]
        keep[cids] = True
        out[pid] = labels * keep[labels]
    return out


def load_rf(rf_path: str | Path):
    """Load a pickled RandomForest produced by `train_rf_v2`."""
    import joblib
    return joblib.load(str(rf_path))


def save_rf(rf, rf_path: str | Path) -> None:
    """Pickle a trained RF to disk."""
    import joblib
    Path(rf_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, str(rf_path))


def build_rf_postproc_v2(
    panel_probs,
    diffim_panels,
    orient_sin,
    orient_cos,
    orient_agg,
    catalog,
    *,
    real_labels=None,
    n_estimators: int = 500, max_depth: int = 14, min_samples_leaf: int = 5,
    n_jobs: int = 32, random_state: int = 0,
    save_to: str | Path | None = None,
    verbose: bool = True,
):
    """End-to-end V2 RF trainer.

    Given the four NN output maps + diffims + truth catalog, computes the
    72-feature candidate table, relabels via injection-overlap matching
    (same definition as `objectwise_confusion`), trains a class-balanced
    RandomForest, and optionally pickles it. Returns the fitted classifier.

    Use this when you want to retrain on a new dataset / threshold variation.
    The shipped `rf_postproc_v2.pkl` was built with this function on
    test_5sigma — see `experiments/diffim_runs/pilot_v7/postproc_iter/train_rf_v2.py`.
    """
    if verbose:
        print("[build] computing V2 features ...", flush=True)
    cand_df, _ = compute_v2_features(
        panel_probs, diffim_panels, orient_sin, orient_cos, orient_agg,
        real_labels=real_labels, verbose=verbose,
    )
    if verbose:
        print(f"[build] {len(cand_df)} candidates", flush=True)
        print("[build] relabel via injection-overlap ...", flush=True)
    labels = label_candidates_by_injection_overlap(cand_df, catalog, panel_probs)
    if verbose:
        print(f"[build] pos={int(labels.sum())} neg={int((labels==0).sum())}",
              flush=True)
        print("[build] fitting RandomForest ...", flush=True)
    rf = train_rf_v2(
        cand_df, labels=labels,
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs, random_state=random_state,
    )
    if save_to is not None:
        save_rf(rf, save_to)
        if verbose:
            print(f"[build] saved {save_to}", flush=True)
    return rf


def rf_score_sweep(cand_df, *, score_col: str = "score_rf",
                    thresholds=None) -> pd.DataFrame:
    """For each threshold, report candidates kept and (when truth columns are
    present) the unique (panel_id, matched_injection_id) TP / informational-FP
    counts. During inference the truth columns are absent and the returned
    table contains only `thr` and `n_candidates`.
    """
    if thresholds is None:
        thresholds = np.concatenate([np.arange(0.02, 0.10, 0.005),
                                     np.arange(0.10, 0.50, 0.02)])
    have_truth = ("matched_injection_id" in cand_df.columns
                  and "frac_real_label_overlap" in cand_df.columns)
    rows = []
    for thr in thresholds:
        sub = cand_df[cand_df[score_col] >= thr]
        row = {"thr": float(thr), "n_candidates": int(len(sub))}
        if have_truth:
            if len(sub) == 0:
                row["TP"] = 0; row["FP"] = 0
            else:
                matched = sub[sub["matched_injection_id"] >= 0]
                keys = set(zip(matched["panel_id"].astype(int).tolist(),
                               matched["matched_injection_id"].astype(int).tolist()))
                unmatched = sub[sub["matched_injection_id"] < 0]
                row["TP"] = len(keys)
                row["FP"] = int((unmatched["frac_real_label_overlap"] < 0.5).sum())
        rows.append(row)
    return pd.DataFrame(rows)
