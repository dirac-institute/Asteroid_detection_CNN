"""Stage 2 of the two-stage ADCNN detector for one panel, run in detect_night's feature workers.

    segmentation model segmentation  ->  candidate components + trail measurement  ->  focal cutout-CNN score

Each row carries the *measured* trail geometry (centroid x/y, orientation ``beta``, ``length``),
the raw NN peak and the stage-2 CNN ``score``. Sky coordinates are added by
ADCNN.inference.detect_night from the panel WCS.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ADCNN.config import ACTIVE as _PIPE  # resolved once at import (ADCNN_PIPELINE selects)

# Public detection-catalog schema: internal candidate column -> emitted column.
_COLMAP = {
    "x_centroid": "x",          # measured centroid (px)
    "y_centroid": "y",
    "mf_beta": "beta",          # trail PA (deg, 0=+x): template bank; footprint PCA at panel edges
    "mf_length": "length",      # physical trail length (px): template bank; de-biased footprint at edges
    "mf_length_raw": "length_raw",  # RAW footprint matched-filter length (px)
    "mf_snr": "mf_snr",
    "max_p": "nn_pmax",         # peak NN segmentation probability
    "score": "score",           # stage-2 cutout-CNN score (operating cut applied before emit)
    # DETECTION-TIME morphology (bright-star diffim dipole/RING measure, DMTN-007): computed on the
    # diffim stamp while it is in memory (no extra IO). See panel_to_catalog_rows +
    # ADCNN.qa.alert_morphology.
    "m_elong": "m_elong", "m_dipole": "m_dipole", "m_neg_ratio": "m_neg_ratio",
    "m_neg_blob": "m_neg_blob", "is_dipole": "is_dipole",
}


@dataclass(frozen=True)
class InferenceConfig:
    """Knobs for the two-stage detector.

      cnn_thr  : CNN score operating point (keep detections with score >= this).
      gate_pmax: skip the expensive candidate features for candidates whose peak NN prob < this
                 (the CNN scores them ~0 anyway; 0 = no gate).
    """
    cnn_thr: float
    gate_pmax: float


# segmentation model's segmentation/matched-filter over-extends trail ends ("ends bloom"): the raw
# footprint mf_length is biased, mf_length ≈ MF_LEN_SLOPE*L_true + MF_LEN_OFFSET. For the detections
# the template bank cannot measure, the emitted `length` INVERTS this to the physical trail length.
#
# The de-bias is MODEL-SPECIFIC (a domain-adapted stage-1 has a different ends-bloom), so it lives in
# the active pipeline config (ADCNN/config.py -> models/<pipeline>/pipeline.json) and TRAVELS WITH THE
# MODEL. Mixing one model with another's de-bias silently corrupts `len_db`.
if _PIPE is None:
    raise RuntimeError("no ADCNN pipeline config (set $ADCNN_PIPELINE)")
MF_LEN_OFFSET = _PIPE.mf_len_offset
MF_LEN_SLOPE = _PIPE.mf_len_slope

print(f"[catalog] MF_LEN de-bias offset={MF_LEN_OFFSET} slope={MF_LEN_SLOPE} "
      f"(pipeline={_PIPE.name})", file=sys.stderr, flush=True)


def panel_to_catalog_rows(prob, img, agg, rl, cnn, config: InferenceConfig) -> Optional[pd.DataFrame]:
    """Stage 2 for one panel: candidate extraction -> cutout-CNN score -> keep score>=cnn_thr
    -> public schema. Returns the per-panel catalog slice, or None if no detection survives.
    """
    from ADCNN.inference.features import extract_panel_candidates
    from ADCNN.inference.cnn_postproc import apply_cnn
    from ADCNN.inference.mf_trail_length import refine_trail_length
    cand = extract_panel_candidates(prob[None], img[None], real_labels=rl[None],
                                    gate_pmax=config.gate_pmax)
    if not len(cand):
        return None
    dev = next(cnn.parameters()).device                       # CNN runs on the GPU (see _load_filter)
    cand = apply_cnn(cand, cnn, img, prob, agg, device=dev)   # cutout score -> `score`
    cand = cand[cand["score"] >= config.cnn_thr].copy()
    if not len(cand):
        return None
    cand["mf_length_raw"] = cand["mf_length"]   # preserve RAW (footprint, pre-debias)
    # TEMPLATE-BANK matched filter (ADCNN.inference.mf_trail_length). The footprint length is
    # the extent of the THRESHOLDED footprint, so it is contrast-dependent: on a faint trail
    # the threshold keeps only the bright middle and the ends are lost (measured median
    # 13.17px against a truth median of ~24px on a faint population).
    #
    # The ends-bloom de-bias is DELIBERATELY NOT APPLIED to template lengths: MF_LEN_OFFSET/SLOPE
    # were calibrated for the FOOTPRINT estimator's bias, so applying them to template lengths
    # would double-correct. The template estimator carries its own single calibration (MF_K).
    #
    # Reuse the panel sigma computed during candidate extraction. Recomputing it dominated
    # the estimator -- 277 ms of a 318 ms call, five times the template bank that needs it --
    # and a second full-panel median would simply pay that cost again.
    _L, _B = refine_trail_length(cand["x_centroid"].to_numpy(), cand["y_centroid"].to_numpy(),
                                 img, cand["mf_length"].to_numpy(), cand["mf_beta"].to_numpy(),
                                 float(cand["panel_sigma"].iloc[0]))
    # refine_trail_length returns the footprint length unchanged for a detection it cannot measure
    # (within MF_STAMP/2 of a panel edge). Left raw, that length is inflated by the ends bloom --
    # measured median 24.09 px against 17.42 px de-biased (ratio 1.370) on 0.87% of detections --
    # and detect_night builds the on-sky trail endpoints from it. Apply the ends-bloom de-bias to
    # exactly the rows that kept the footprint estimate.
    _fellback = np.isclose(_L, cand["mf_length"].to_numpy(), rtol=0, atol=0)
    _L = np.asarray(_L, float).copy()
    if _fellback.any():
        _L[_fellback] = np.clip(
            (_L[_fellback] - MF_LEN_OFFSET) / MF_LEN_SLOPE, 0.0, None)
        print(f"[catalog] {int(_fellback.sum())} edge detections kept the footprint length "
              f"-> ends-bloom de-bias applied to those rows only", flush=True)
    cand["mf_length"] = np.clip(_L, 0.0, None)
    cand["mf_beta"] = _B
    _attach_dipole_morphology(cand, img)             # DETECTION-TIME ring flag (no extra pixel IO)
    return cand[list(_COLMAP)].rename(columns=_COLMAP)


# DMTN-007 dipole/RING morphology at DETECTION time -----------------------------------------------
# The stage-1 image (diffim) is already in memory here, so the per-detection dipole measure costs no
# extra pixel IO.
_MORPH_K = 96          # stamp size for the feature (_features uses win=28)

def _attach_dipole_morphology(cand: pd.DataFrame, img) -> None:
    from ADCNN.qa.alert_morphology import _features, ripple_flag
    from ADCNN.inference.cnn_postproc import _cutout
    xs = cand["x_centroid"].to_numpy(); ys = cand["y_centroid"].to_numpy()
    E = np.empty(len(xs)); D = np.empty(len(xs)); SN = np.empty(len(xs))
    NR = np.empty(len(xs)); NB = np.empty(len(xs))
    for i in range(len(xs)):
        e, d, pk, nr, nb = _features(_cutout(img, xs[i], ys[i], _MORPH_K))
        E[i], D[i], SN[i], NR[i], NB[i] = e, d, pk, nr, nb
    cand["m_elong"] = E; cand["m_dipole"] = D; cand["m_neg_ratio"] = NR; cand["m_neg_blob"] = NB
    cand["is_dipole"] = ripple_flag(E, D, SN, NR, NB)


# --- feature-worker process state (one CNN + config per worker, set once at spawn) ---
_CNN = None
_CONFIG: Optional[InferenceConfig] = None


def _load_filter(cnn_pt: str):
    """Load the stage-2 focal cutout CNN (on the GPU when available)."""
    import torch
    from ADCNN.inference.cnn_postproc import load_cnn
    return load_cnn(str(cnn_pt), device=("cuda" if torch.cuda.is_available() else "cpu"))


def _worker_init(cnn_pt: str, config: InferenceConfig) -> None:
    """Isolate each feature worker: pin BLAS to one thread (we parallelise across panels, so
    per-worker thread pools would only oversubscribe) and load one CNN. The cutout CNN scores
    on the GPU, so the worker keeps the shard's CUDA_VISIBLE_DEVICES (shared with segmentation model)."""
    # Setting these here is TOO LATE: this module imports numpy at module scope, which under spawn
    # runs BEFORE the initializer, so the BLAS pool is already sized from the visible core count.
    # MEASURED: threadpool_info() inside a worker reported 64 OpenBLAS threads, with n_workers =
    # cores-1 = 127 -- i.e. 127x64 threads on 128 cores. Kept for child processes that import later;
    # threadpoolctl below is what actually pins the already-initialised pool.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    global _CNN, _CONFIG
    _CNN = _load_filter(cnn_pt)
    _CONFIG = config


def _worker(args):
    return panel_to_catalog_rows(*args, _CNN, _CONFIG)
