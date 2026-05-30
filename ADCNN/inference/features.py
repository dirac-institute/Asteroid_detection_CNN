"""Stage-2 candidate extraction for the diffim detector.

``extract_panel_candidates`` runs the segmentation-model candidate extractor and measures the
matched-filter trail geometry (``mf_length`` / ``mf_beta`` / ``mf_snr`` / ``mf_flux``) for each
candidate. That's all the public detection catalog needs (see
:mod:`ADCNN.inference.catalog._COLMAP`); the cutout CNN supplies the score from the raw
``[diffim/sigma, seg_prob, seg_agg]`` cutout, no further hand-crafted features.

``label_candidates_by_injection_overlap`` labels candidates against an injection truth catalog
(1 = the candidate's connected component overlaps an injected trail) — the supervision signal
for training the stage-2 cutout CNN (:mod:`ADCNN.training.cnn_postproc`).
"""
from __future__ import annotations

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
from ADCNN.utils.helpers import to_panel_dict, trail_bbox


def extract_panel_candidates(
    panel_probs,
    diffim_panels,
    *,
    real_labels=None,
    adaptive: bool = True,
    t_low: float = 0.05,
    min_area: int = 4,
    line_width: int = 2,
    pad_length: int = 4,
    gate_pmax: float = 0.0,
    verbose: bool = False,
):
    """Extract candidates from the segmentation prob map + measure trail geometry.

    Pipeline per panel:
      1. Connected components above ``t_low`` (``extract_candidates``) — footprint + ``max_p``,
         ``area``, ``elongation``, plus optional ``frac_real_label_overlap`` (used by the
         stage-2 trainer's injection-overlap labelling).
      2. Per-candidate matched filter along the footprint principal axis
         (``matched_filter_for_nn_candidates``) — ``mf_beta``, ``mf_length``, ``mf_snr``,
         ``mf_flux``.

    Optional ``gate_pmax`` drops candidates whose peak NN probability is below the cut BEFORE
    the matched-filter pass (cheap; the stage-2 CNN scores them ~0 anyway).

    Returns ``(cand_df, panel_probs_dict)``. The panel_probs dict mirrors the legacy caller
    contract (input cast to float32, no mutation).
    """
    panel_probs = to_panel_dict(panel_probs)
    diffims = to_panel_dict(diffim_panels)
    rl_dict = to_panel_dict(real_labels) if real_labels is not None else None

    pids = sorted(panel_probs.keys())
    cfg = CandidateExtractorConfig(t_low=t_low, min_area=min_area, adaptive_t_low=adaptive)
    cand_dfs = []
    for pid in pids:
        panel = panel_probs[pid].astype(np.float32, copy=False)
        rl = rl_dict[pid] if rl_dict is not None else None
        cand = extract_candidates(panel, real_labels=rl, cfg=cfg, panel_id=pid)
        if len(cand):
            cand_dfs.append(cand)
        panel_probs[pid] = panel
    if not cand_dfs:
        return pd.DataFrame(), panel_probs
    cand_df = pd.concat(cand_dfs, ignore_index=True)

    if gate_pmax > 0.0:
        cand_df = cand_df[cand_df["max_p"] >= gate_pmax].reset_index(drop=True)
        if not len(cand_df):
            return pd.DataFrame(), panel_probs

    diffims = {pid: arr.astype(np.float32, copy=False) for pid, arr in diffims.items()}
    panel_sigmas = {pid: panel_mad_sigma(diffims[pid]) for pid in pids}
    cand_df = matched_filter_for_nn_candidates(
        cand_df, panel_probs=panel_probs, diffim_panels=diffims,
        panel_sigmas=panel_sigmas, line_width=line_width, pad_length=pad_length,
    )
    return cand_df, panel_probs


def label_candidates_by_injection_overlap(
    cand_df,
    catalog,
    panel_probs,
    *,
    psf_width: int = 40,
) -> np.ndarray:
    """Return (N,) int8 labels aligned with ``cand_df``: 1 iff the candidate's connected
    component overlaps any catalog injection trail.

    This mirrors ``objectwise_confusion``'s matching exactly, so the candidate-level labels
    and the eval metric agree — important when training a candidate-level filter (the stage-2
    cutout CNN). Args mirror the upstream callsite: cand_df from ``extract_panel_candidates``,
    truth ``catalog`` with ``image_id, x, y, beta, trail_length``, ``panel_probs`` either a
    stacked array or a ``{pid: (H, W)}`` dict.
    """
    from ADCNN.utils.helpers import draw_one_line  # local import (cv2 optional)

    probs_dict = to_panel_dict(panel_probs)
    pid_sample = next(iter(probs_dict.values()))
    H, W = pid_sample.shape
    half_psf = psf_width // 2
    structure = ndi.generate_binary_structure(2, 2)

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
            x0, x1, y0, y1 = trail_bbox(x, y, beta, L, H, W, pad)
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
