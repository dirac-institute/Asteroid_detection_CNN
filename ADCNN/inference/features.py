"""Stage-2 candidate extraction for the diffim detector.

``extract_panel_candidates`` runs the segmentation-model candidate extractor and measures the
matched-filter trail geometry (``mf_length`` / ``mf_beta`` / ``mf_snr``) for each candidate.
That's all the public detection catalog needs (see :mod:`ADCNN.inference.catalog._COLMAP`); the
cutout CNN supplies the score from the raw ``[diffim/sigma, seg_prob, seg_agg]`` cutout, no further
hand-crafted features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ADCNN.inference.candidates import (
    extract_candidates,
    CandidateExtractorConfig,
)
from ADCNN.data.preprocessing import diffim_mad_sigma
from ADCNN.inference.matched_filter import matched_filter_for_nn_candidates
from ADCNN.utils.helpers import to_panel_dict


def extract_panel_candidates(panel_probs, diffim_panels, *, real_labels, gate_pmax: float):
    """Extract candidates from the segmentation prob map + measure trail geometry.

    Pipeline per panel:
      1. Connected components above the (adaptive) ``t_low`` (``extract_candidates``) — footprint
         + ``max_p``, ``area``, ``elongation``, ``frac_real_label_overlap``.
      2. Per-candidate matched filter along the footprint principal axis
         (``matched_filter_for_nn_candidates``) — ``mf_beta``, ``mf_length``, ``mf_snr``.

    ``gate_pmax`` drops candidates whose peak NN probability is below the cut BEFORE the
    matched-filter pass (cheap; the stage-2 CNN scores them ~0 anyway).

    Returns the candidate frame with ``panel_sigma`` (empty when no candidate survives).
    """
    panel_probs = to_panel_dict(panel_probs)
    diffims = to_panel_dict(diffim_panels)
    rl_dict = to_panel_dict(real_labels)

    pids = sorted(panel_probs.keys())
    cfg = CandidateExtractorConfig()
    cand_dfs = []
    for pid in pids:
        panel = panel_probs[pid].astype(np.float32, copy=False)
        cand = extract_candidates(panel, real_labels=rl_dict[pid], cfg=cfg, panel_id=pid)
        if len(cand):
            cand_dfs.append(cand)
        panel_probs[pid] = panel     # the matched filter thresholds the float32 map
    if not cand_dfs:
        return pd.DataFrame()
    cand_df = pd.concat(cand_dfs, ignore_index=True)

    if gate_pmax > 0.0:
        cand_df = cand_df[cand_df["max_p"] >= gate_pmax].reset_index(drop=True)
        if not len(cand_df):
            return pd.DataFrame()

    diffims = {pid: arr.astype(np.float32, copy=False) for pid, arr in diffims.items()}
    panel_sigmas = {pid: diffim_mad_sigma(diffims[pid]) for pid in pids}
    # Carry the panel noise scale on the candidate frame. It is already computed here, and a second
    # full-panel median costs ~158 ms -- five times the entire matched-filter length estimate that
    # needs it (31 ms). One computation, one source of truth.
    cand_df = matched_filter_for_nn_candidates(
        cand_df, panel_probs=panel_probs, diffim_panels=diffims, panel_sigmas=panel_sigmas)
    cand_df["panel_sigma"] = cand_df["panel_id"].map(panel_sigmas).astype("float32")
    return cand_df
