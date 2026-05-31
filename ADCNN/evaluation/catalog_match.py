"""Catalog-vs-catalog trail-overlap matching for evaluation.

The detector emits a *measured* catalog — one row per detection carrying the measured trail
geometry ``(image_id, x, y, beta, length)`` (see :mod:`ADCNN.inference.catalog`). Evaluation
compares it against a *truth* catalog — one row per injected source ``(image_id, x, y, beta,
trail_length)``.

Each trail is modelled as a line **segment** centred on ``(x, y)``, oriented at ``beta``
degrees (image convention, 0 = +x), of the given length. A truth trail is a **true positive**
if any measured trail's segment passes within ``tol_px`` of it (the "any-pixel-overlap"
criterion); a measured trail that matches no truth is a **false positive**; a truth trail
matched by none is a **false negative**. Matching is per ``image_id`` — a measured trail can
only match truth on its own panel.

``tol_px`` is a fixed matching tolerance chosen in advance (≈ PSF scale); it is never tuned
on the evaluation set. Recall uses the **all-trails** denominator (every truth row counts,
including faint ones below the detection limit). Rows with non-finite geometry never match
(NaN-safe), so they fall through to FN/FP as appropriate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import h5py
import numpy as np
import pandas as pd

__all__ = ["match_trail_catalogs", "evaluate_catalog", "match_pairs",
           "stack_sigma_catalog", "dedup_within_panel", "dedup_cross_catalog"]

# Columns every trail catalog must provide (besides its length column).
_REQUIRED_COLUMNS = ("image_id", "x", "y", "beta")
_EPS = 1e-12

CatalogLike = Union[pd.DataFrame, str, Path]


def _require_columns(df: pd.DataFrame, name: str, length_col: str) -> None:
    missing = [c for c in (*_REQUIRED_COLUMNS, length_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{name} catalog is missing required column(s): {missing}")


def _segment_endpoints(df: pd.DataFrame, length_col: str) -> np.ndarray:
    """Endpoints of each trail segment as an ``(N, 2, 2)`` array ``[[x0, y0], [x1, y1]]``.

    The segment is centred on ``(x, y)``, oriented ``beta`` degrees from +x, of the given
    length (so it spans ``±length/2`` about the centre along that direction).
    """
    theta = np.radians(df["beta"].to_numpy(np.float64))
    half = 0.5 * df[length_col].to_numpy(np.float64)
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    dx, dy = np.cos(theta) * half, np.sin(theta) * half
    p0 = np.stack([x - dx, y - dy], axis=-1)
    p1 = np.stack([x + dx, y + dy], axis=-1)
    return np.stack([p0, p1], axis=1)


def _points_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Distance from every point to every segment.

    ``points`` is ``(K, 2)``; ``segments`` is ``(L, 2, 2)``; returns ``(K, L)``.
    """
    a = segments[:, 0, :]                                   # (L, 2)
    ab = segments[:, 1, :] - a                              # (L, 2)
    ap = points[:, None, :] - a[None, :, :]                 # (K, L, 2)
    denom = np.einsum("ld,ld->l", ab, ab)                   # (L,)
    t = np.einsum("kld,ld->kl", ap, ab) / np.where(denom > _EPS, denom, 1.0)
    t = np.clip(t, 0.0, 1.0)                                # (K, L)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]   # (K, L, 2)
    return np.linalg.norm(points[:, None, :] - proj, axis=-1)


def _orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Signed area (×2) of triangle (p, q, r); sign gives the turn direction of p→q→r."""
    return (r[..., 1] - p[..., 1]) * (q[..., 0] - p[..., 0]) \
         - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def _pairwise_segment_distance(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Minimum distance between every segment in ``seg_a`` and every segment in ``seg_b``.

    ``seg_a`` is ``(N, 2, 2)``, ``seg_b`` is ``(M, 2, 2)``; returns ``(N, M)``. The distance
    is 0 where two segments cross, otherwise the smallest of the four endpoint-to-opposite-
    segment distances (the standard segment-to-segment distance).
    """
    a0, a1 = seg_a[:, 0, :], seg_a[:, 1, :]
    b0, b1 = seg_b[:, 0, :], seg_b[:, 1, :]
    endpoint_dist = np.minimum.reduce([
        _points_to_segments(a0, seg_b),        # (N, M)
        _points_to_segments(a1, seg_b),        # (N, M)
        _points_to_segments(b0, seg_a).T,      # (M, N) -> (N, M)
        _points_to_segments(b1, seg_a).T,
    ])
    A0, A1 = a0[:, None, :], a1[:, None, :]    # (N, 1, 2)
    B0, B1 = b0[None, :, :], b1[None, :, :]    # (1, M, 2)
    crosses = ((_orient(A0, B0, B1) * _orient(A1, B0, B1) <= 0)
               & (_orient(B0, A0, A1) * _orient(B1, A0, A1) <= 0))
    return np.where(crosses, 0.0, endpoint_dist)


def match_trail_catalogs(
    measured: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    tol_px: float = 10.0,
    truth_length_col: str = "trail_length",
    meas_length_col: str = "length",
    flag_col: str = "nn_detected",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Trail-overlap match between a measured detection catalog and a truth catalog.

    Both frames need columns ``image_id, x, y, beta`` plus their length column
    (``trail_length`` for truth, ``length`` for measured). Matching runs per ``image_id``
    and is vectorised over the candidate × truth pairs on each panel.

    Args:
        measured: detection catalog (one row per detection).
        truth: injected-source catalog (one row per trail).
        tol_px: a truth trail is matched if a measured trail's segment lies within this
            distance (pixels) of it. Fixed and pre-chosen — not tuned on the eval set.
        truth_length_col / meas_length_col: trail-length column names in each frame.
        flag_col: name of the boolean "matched" column added to the returned truth frame.

    Returns:
        ``(truth_out, measured_out, counts)`` where ``truth_out`` is ``truth`` plus a bool
        ``flag_col`` (matched by ≥1 detection), ``measured_out`` is ``measured`` plus a bool
        ``matched`` column, and ``counts`` is ``{"TP", "FP", "FN"}`` object-level confusion
        (TP = truth matched, FN = truth unmatched, FP = measured matching no truth).

    Raises:
        ValueError: if either catalog is missing a required column.
    """
    truth = truth.copy().reset_index(drop=True)
    measured = measured.copy().reset_index(drop=True)
    _require_columns(truth, "truth", truth_length_col)
    _require_columns(measured, "measured", meas_length_col)

    truth_matched = np.zeros(len(truth), dtype=bool)
    meas_matched = np.zeros(len(measured), dtype=bool)

    if len(truth) and len(measured):
        meas_by_panel = measured.groupby("image_id").indices  # {id: positional indices}
        for img_id, t_pos in truth.groupby("image_id").indices.items():
            m_pos = meas_by_panel.get(img_id)
            if m_pos is None or len(m_pos) == 0:
                continue
            t_seg = _segment_endpoints(truth.iloc[t_pos], truth_length_col)
            m_seg = _segment_endpoints(measured.iloc[m_pos], meas_length_col)
            within = _pairwise_segment_distance(t_seg, m_seg) <= tol_px  # (nt, nm); NaN -> False
            truth_matched[t_pos] = within.any(axis=1)
            meas_matched[m_pos] = within.any(axis=0)

    truth[flag_col] = truth_matched
    measured["matched"] = meas_matched
    counts = {
        "TP": int(truth_matched.sum()),
        "FN": int((~truth_matched).sum()),
        "FP": int((~meas_matched).sum()),
    }
    return truth, measured, counts


def _as_frame(catalog: CatalogLike) -> pd.DataFrame:
    return pd.read_csv(catalog) if isinstance(catalog, (str, Path)) else catalog


def evaluate_catalog(
    measured: CatalogLike,
    truth: CatalogLike,
    *,
    tol_px: float = 20.0,
    flag_col: str = "nn_detected",
    n_panels: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Catalog-based evaluation entry point.

    Matches a measured detection catalog against a truth catalog and returns object-level
    metrics plus the flagged truth catalog (ready for the completeness / histogram plots in
    :mod:`ADCNN.evaluation.plots`, which read the ``flag_col`` and ``stack_detection`` columns).

    Args:
        measured / truth: DataFrames or paths to CSVs.
        tol_px: fixed trail-overlap tolerance (pixels), chosen in advance.
        flag_col: name of the boolean detection flag added to the truth catalog.
        n_panels: total panels processed, for the ``fp_per_panel`` denominator. If None it is
            inferred as the panels carrying a detection OR a truth trail (so empty-background
            panels that produced false positives are still counted); pass the exact panel count
            (e.g. the h5 image count) for full correctness on sets with all-empty panels.

    Returns:
        ``(counts, truth_out)`` where ``counts`` has ``TP, FP, FN`` plus derived
        ``recall`` (over the all-trails denominator), ``n_panels`` and ``fp_per_panel``.
    """
    measured_df = _as_frame(measured)
    truth_df = _as_frame(truth)
    truth_out, _, counts = match_trail_catalogs(
        measured_df, truth_df, tol_px=tol_px, flag_col=flag_col,
    )
    if n_panels is None:
        # Old code used truth_df.image_id.nunique(), which omitted empty-injection panels (no
        # truth rows) whose FPs were still counted -> inflated fp_per_panel. Use the union of
        # panels seen in `measured` or `truth` instead.
        ids: set = set()
        if "image_id" in measured_df.columns:
            ids |= set(measured_df["image_id"].unique())
        if "image_id" in truth_df.columns:
            ids |= set(truth_df["image_id"].unique())
        n_panels = len(ids)
    n_panels = int(n_panels)
    metrics: dict[str, float] = dict(counts)
    metrics["recall"] = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    metrics["n_panels"] = n_panels
    metrics["fp_per_panel"] = counts["FP"] / max(n_panels, 1)
    return metrics, truth_out


def stack_sigma_catalog(h5_path, csv_path, panel_ids: Iterable[int], *, sigma: int = 5) -> pd.DataFrame:
    """Build the LSST stack DETECTION catalog for the given panels at one sigma.

    Returns one row per stack detection -- both sides of confusion:
      * detected injected trails (TP side): rows of `csv_path` whose
        ``stack_detection_<sigma>sigma == True``; carries the truth trail's beta + trail_length
        so segment-overlap matching in :func:`evaluate_catalog` works as for the deployed catalog.
      * real-residual centroids (FP side): per-label centroids from the h5
        ``real_labels_<sigma>sigma`` plane; point-like (length=0, beta=0).

    The set's h5 is indexed by ``image_id`` (cnn_val.h5, test.h5 are both 0..N contiguous), so
    `panel_ids` index both the h5 dataset and the csv ``image_id`` column directly.

    Used by both the FP-budget calibration (:func:`ADCNN.training.cnn_postproc.combined_fpp_threshold`)
    and the eval notebook to compute the dedup'd 5sigma+NN union FP correctly.
    """
    flag = f"stack_detection_{sigma}sigma"
    plane = f"real_labels_{sigma}sigma"
    panel_ids = list(panel_ids)
    truth = pd.read_csv(csv_path)
    if flag not in truth.columns:
        raise ValueError(f"{csv_path}: missing column '{flag}' (rebuild with --multi-sigma-sets "
                         f"covering this set + --test-sigmas {sigma})")
    truth = truth[truth["image_id"].isin(set(panel_ids))]
    det = truth[truth[flag] == True][["image_id", "x", "y", "beta", "trail_length"]].rename(
        columns={"trail_length": "length"}).copy()
    resid_rows = []
    with h5py.File(h5_path, "r") as f:
        if plane not in f:
            raise ValueError(f"{h5_path}: missing dataset '{plane}' (rebuild with "
                             f"--multi-sigma-sets covering this set + --test-sigmas {sigma})")
        for pid in panel_ids:
            rl = f[plane][int(pid)][:]
            mx = int(rl.max()) if rl.size else 0
            if mx <= 0:
                continue
            ys, xs = np.nonzero(rl)
            lab = rl[ys, xs]
            for L in range(1, mx + 1):
                m = lab == L
                if m.any():
                    resid_rows.append((int(pid), float(xs[m].mean()), float(ys[m].mean()), 0.0, 0.0))
    resid = pd.DataFrame(resid_rows, columns=["image_id", "x", "y", "beta", "length"])
    return pd.concat([det, resid], ignore_index=True)


def dedup_cross_catalog(primary: pd.DataFrame, secondary: pd.DataFrame, *,
                        tol_px: float = 20.0) -> pd.DataFrame:
    """Concat `primary` then `secondary`, drop rows in `secondary` that fall within `tol_px` of
    ANY `primary` row on the same panel; rows within `primary`-vs-`primary` or
    `secondary`-vs-`secondary` are PRESERVED.

    This is the right dedup for combining two independent detectors' catalogs (e.g. the LSST
    stack and ADCNN): a source both detectors fire on counts once, but each detector's own
    cluster of detections on the same source is left alone (each catalog reports what its
    pipeline reports). `dedup_within_panel` is the stronger collapse that also drops
    within-catalog coincidences -- not what you want when comparing two independent detectors.
    """
    from scipy.spatial import cKDTree
    if not len(secondary):
        return primary.copy()
    if not len(primary):
        return secondary.copy()
    parts = []
    sec_by_panel = secondary.groupby("image_id", sort=False).indices  # {pid: positional indices}
    pri_by_panel = primary.groupby("image_id", sort=False).indices
    for pid, p_idx in pri_by_panel.items():
        p_rows = primary.iloc[p_idx]
        s_idx = sec_by_panel.get(pid)
        if s_idx is None or not len(s_idx):
            parts.append(p_rows); continue
        s_rows = secondary.iloc[s_idx]
        tree = cKDTree(p_rows[["x", "y"]].values)
        # any secondary row within tol_px of any primary row -> drop that secondary row
        nbr = tree.query_ball_point(s_rows[["x", "y"]].values, r=tol_px)
        keep_s = np.array([len(n) == 0 for n in nbr])
        parts.append(p_rows)
        parts.append(s_rows[keep_s])
    # panels seen only in secondary: keep all secondary rows
    sec_only_panels = set(secondary["image_id"].unique()) - set(pri_by_panel)
    if sec_only_panels:
        parts.append(secondary[secondary["image_id"].isin(sec_only_panels)])
    return pd.concat(parts, ignore_index=True)


def dedup_within_panel(df: pd.DataFrame, *, tol_px: float = 20.0) -> pd.DataFrame:
    """Drop detections coincident within `tol_px` of an EARLIER row on the same panel.

    Per ``image_id``, scipy's ``cKDTree.query_pairs(tol_px)`` returns coincident index pairs;
    the later index is dropped. The frame's row order matters: concatenate the catalog you want
    to prefer FIRST. For the 5sigma+NN union the stack rows come first, so a source both
    detectors fire on is counted once and attributed to the stack.
    """
    from scipy.spatial import cKDTree
    if not len(df):
        return df
    parts = []
    for _, g in df.groupby("image_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) <= 1:
            parts.append(g); continue
        keep = np.ones(len(g), bool)
        for a, b in cKDTree(g[["x", "y"]].values).query_pairs(tol_px):
            if keep[a] and keep[b]:
                keep[b] = False
        parts.append(g[keep])
    return pd.concat(parts, ignore_index=True)


def match_pairs(measured, truth, *, tol_px: float = 20.0,
                truth_length_col: str = "trail_length", meas_length_col: str = "length"):
    """For each truth trail with a match, return the NEAREST measured detection, as a frame
    with the truth row plus ``meas_x/meas_y/meas_beta/meas_length/meas_score`` of that nearest
    detection. Used for parameter-recovery residuals (measured geometry vs truth).

    `measured`/`truth` may be DataFrames or CSV paths. Only matched truths (nearest detection
    within `tol_px`) are returned, one row each.
    """
    measured = _as_frame(measured).reset_index(drop=True)
    truth = _as_frame(truth).reset_index(drop=True)
    if not len(measured) or not len(truth):
        return truth.iloc[0:0].copy()
    meas_by_panel = measured.groupby("image_id").indices
    rows = []
    for img_id, t_pos in truth.groupby("image_id").indices.items():
        m_pos = meas_by_panel.get(img_id)
        if m_pos is None or len(m_pos) == 0:
            continue
        D = _pairwise_segment_distance(_segment_endpoints(truth.iloc[t_pos], truth_length_col),
                                       _segment_endpoints(measured.iloc[m_pos], meas_length_col))
        for ti, t_lbl in enumerate(t_pos):
            j = int(D[ti].argmin())
            if D[ti, j] > tol_px:
                continue
            md = measured.iloc[m_pos[j]]
            row = truth.iloc[t_lbl].to_dict()
            # stage-2 score column is `score` (older catalogs called it `score_rf`)
            meas_score = md.get("score", md.get("score_rf", float("nan")))
            row.update(meas_x=md["x"], meas_y=md["y"], meas_beta=md["beta"],
                       meas_length=md[meas_length_col], meas_score=meas_score)
            rows.append(row)
    return pd.DataFrame(rows)
