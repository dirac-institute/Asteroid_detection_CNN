"""ENTRY POINT — end-to-end ADCNN inference: diffim panels -> detection catalog.

Runs the full two-stage detector over every panel of an h5 and emits ONE ROW PER KEPT
DETECTION (RF score >= rf_thr) as a CSV catalog:

    v7 segmentation  ->  candidate components + 72 features  ->  RandomForest score

Each row carries the *measured* trail geometry (centroid x/y, orientation ``beta``,
``length``), brightness (``flux``), the raw NN peak, and the RF score — everything an
evaluator needs to overlap-match this catalog against a truth catalog, and everything
HelioLinC needs once sky coordinates are attached.

Sky coordinates (RA/Dec/MJD) are deliberately NOT added here: they require the per-panel
Butler WCS, which lives in the ``lsst_distrib`` env (no torch). This engine runs in the
torch env and emits the pixel-space catalog plus the routing keys (``image_id``, and
``visit``/``detector``/``band`` when a ``panels.csv`` is supplied).
``experiments/heliolinc/adcnn_wcs.py`` is the Butler step that turns those into the
HelioLinC-format catalog (``detid,mjd,ra,dec,mag,band,obscode``).

    python -m ADCNN.inference.catalog \
        --h5 DATA_DIFFIM/test_5sigma/test.h5 \
        --panels DATA_DIFFIM/test_5sigma/panels.csv \
        --out detections_5sigma.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.rf_postproc import (
    RF_FEATURES_V2, compute_v2_features, apply_rf_v2, load_rf, DEFAULT_THR,
)

REPO = Path(__file__).resolve().parents[2]

# Public detection-catalog schema: internal candidate column -> emitted column.
# Keep stable — the eval matcher (ADCNN.evaluation.catalog_match) and the HelioLinC
# bridge read these names.
_COLMAP = {
    "image_id": "image_id",
    "x_centroid": "x",          # measured centroid (px)
    "y_centroid": "y",
    "or_beta": "beta",          # measured orientation (deg, image convention 0=+x)
    "mf_length": "length",      # measured trail length (px)
    "mf_flux": "flux",          # integrated matched-filter flux (brightness proxy)
    "mf_snr": "mf_snr",
    "area": "area",
    "elongation": "elongation",
    "max_p": "nn_pmax",         # peak NN segmentation probability
    "score_rf": "score_rf",     # stage-2 RF score (operating cut applied before emit)
}
CATALOG_COLUMNS = list(_COLMAP.values())


def build_detection_catalog(h5_path, v7_ckpt, rf_pkl, *, panels_csv=None,
                            rf_thr: float = DEFAULT_THR, device: str = "cuda",
                            panel_ids=None) -> pd.DataFrame:
    """Run the two-stage detector over `h5_path`; return one row per kept detection.

    `rf_thr` is the pre-chosen RF operating point (defaults to the model's shipped
    DEFAULT_THR). `panels_csv`, if given, attaches visit/detector/band by `image_id`
    so the downstream Butler WCS step can add sky coordinates for HelioLinC.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(v7_ckpt), map_location=dev).eval()
    rf = load_rf(str(rf_pkl))
    parts = []
    with h5py.File(h5_path, "r") as f:
        ids = range(int(f["images"].shape[0])) if panel_ids is None else panel_ids
        for pid in ids:
            img = f["images"][pid][:].astype(np.float32)
            rl = f["real_labels"][pid][:].astype(np.uint16)
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            cand, _ = compute_v2_features(prob[None], img[None], sin[None], cos[None], agg[None],
                                          real_labels=rl[None], verbose=False)
            if not len(cand):
                continue
            cand[list(RF_FEATURES_V2)] = cand[list(RF_FEATURES_V2)].replace([np.inf, -np.inf], np.nan)
            cand = apply_rf_v2(cand, rf)
            cand = cand[cand["score_rf"] >= rf_thr].copy()
            if not len(cand):
                continue
            cand["image_id"] = int(pid)
            parts.append(cand)

    if not parts:
        cat = pd.DataFrame(columns=CATALOG_COLUMNS)
    else:
        full = pd.concat(parts, ignore_index=True)
        cat = full[[c for c in _COLMAP if c in full.columns]].rename(columns=_COLMAP)

    if panels_csv:
        pan = pd.read_csv(panels_csv)
        keep = [c for c in ("image_id", "visit", "detector", "band") if c in pan.columns]
        if len(keep) > 1:
            cat = cat.merge(pan[keep], on="image_id", how="left")
    return cat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="diffim panel h5 (images + real_labels)")
    ap.add_argument("--panels", help="optional panels.csv -> attach visit/detector/band")
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--rf", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--rf-thr", type=float, default=DEFAULT_THR,
                    help="pre-chosen RF operating point (default = shipped DEFAULT_THR)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all panels")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pids = range(a.limit) if a.limit else None
    cat = build_detection_catalog(a.h5, a.v7, a.rf, panels_csv=a.panels,
                                  rf_thr=a.rf_thr, device=a.device, panel_ids=pids)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(a.out, index=False)
    npan = cat["image_id"].nunique() if len(cat) else 0
    print(f"[catalog] {len(cat)} detections (score>={a.rf_thr}) over {npan} panels -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
