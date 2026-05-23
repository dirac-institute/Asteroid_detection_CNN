"""ENTRY POINT — run the trained v7 + RandomForest on diffim panels.

Two-stage inference: v7 segmentation -> candidate components + 72 features ->
RandomForest score. Emits a detection catalog (one row per kept candidate, with its
panel, centroid (x,y), trail geometry and RF score). This is the per-panel detector
output; converting (x,y) -> (RA,Dec) via the panel WCS (for downstream linking with
HelioLinC) is a separate Butler step.

Defaults point at the deployed models in models/ (reg2 v7 + neg5 RF).

    python -m ADCNN.pipelines.run_inference --h5 DATA_DIFFIM/test_real/test.h5 --out detections.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    RF_FEATURES_V2, compute_v2_features, apply_rf_v2, load_rf,
)

REPO = Path(__file__).resolve().parents[2]


def run(v7_ckpt, rf_pkl, h5_path, panel_ids, rf_thr, device):
    """Yield kept (score >= rf_thr) candidate detections across `panel_ids`."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(v7_ckpt), map_location=dev).eval()
    rf = load_rf(str(rf_pkl))
    rows = []
    with h5py.File(h5_path, "r") as f:
        if panel_ids is None:
            panel_ids = range(int(f["images"].shape[0]))
        for pid in panel_ids:
            img = f["images"][pid][:].astype(np.float32)
            rl = f["real_labels"][pid][:].astype(np.uint16)
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            cand, _ = compute_v2_features(prob[None], img[None], sin[None], cos[None], agg[None],
                                          real_labels=rl[None], verbose=False)
            if not len(cand):
                continue
            cand[list(RF_FEATURES_V2)] = cand[list(RF_FEATURES_V2)].replace([np.inf, -np.inf], np.nan)
            cand = apply_rf_v2(cand, rf)
            for _, c in cand[cand.score_rf >= rf_thr].iterrows():
                rows.append(dict(panel=int(pid), x=float(c.x_centroid), y=float(c.y_centroid),
                                 score_rf=float(c.score_rf)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--rf", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--h5", required=True, help="diffim panel h5 (images + real_labels)")
    ap.add_argument("--rf-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all panels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    pids = range(a.limit) if a.limit else None
    df = run(a.v7, a.rf, a.h5, pids, a.rf_thr, a.device)
    df.to_csv(a.out, index=False)
    print(f"[inference] {len(df)} detections (score>={a.rf_thr}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
