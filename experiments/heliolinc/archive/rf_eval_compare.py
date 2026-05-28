"""Eval-only A/B: does the corrected-orientation RF beat the old one on the synthetic test sets?

Loads the two DEPLOYED RandomForests:
  - new  = models/rf_postproc.pkl              (retrained on footprint-PCA orientation features)
  - old  = models/rf_postproc_nnhead_backup.pkl (the previous RF, NN-head orientation)
and scores each on test_5sigma/4/3 with ITS OWN feature convention (the new RF on pca features,
the old RF on nnhead features — recomputed from the same single segmentation model pass). Reports recall vs
FP/panel via the trail-overlap matcher at the deployed operating points. gate_pmax=0.10 (deployed
eval gate) -> fast on a single GPU.

    python experiments/heliolinc/rf_eval_compare.py            # all test panels, 1 GPU
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
DATA = REPO / "DATA_DIFFIM"
TESTSETS = ["test_5sigma", "test_4sigma", "test_3sigma"]


def main():
    import h5py
    import torch
    from ADCNN.data.preprocessing import diffim_mad_sigma
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import compute_v2_features, _add_orient, RF_FEATURES_V2
    from ADCNN.inference.rf_postproc import load_rf
    from ADCNN.evaluation.catalog_match import evaluate_catalog

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seg-model", default=str(REPO / "models/segmentation_model.pt"))
    ap.add_argument("--rf-new", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--rf-old", default=str(REPO / "models/rf_postproc_nnhead_backup.pkl"))
    ap.add_argument("--eval-panels", type=int, default=0, help="0 = all")
    ap.add_argument("--thrs", default="0.3,0.5")
    a = ap.parse_args()
    feats = list(RF_FEATURES_V2)
    thrs = [float(t) for t in a.thrs.split(",")]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(a.seg_model, map_location=dev).eval()
    rf_new, rf_old = load_rf(a.rf_new), load_rf(a.rf_old)

    print(f"{'set':12s} {'thr':>4s} | {'NEW recall':>11s} {'NEW fp/pan':>10s} | "
          f"{'OLD recall':>10s} {'OLD fp/pan':>10s}", flush=True)
    for s in TESTSETS:
        truth = pd.read_csv(DATA / s / "test.csv")
        ids = sorted(truth.image_id.unique())
        if a.eval_panels:
            ids = ids[:a.eval_panels]
        rows_new, rows_old = [], []
        with h5py.File(DATA / s / "test.h5", "r") as f:
            for pid in ids:
                img = f["images"][pid][:].astype(np.float32)
                rl = f["real_labels"][pid][:].astype(np.uint16)
                p, sn, cs, ag = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
                prob = p.astype(np.float32)[None]
                cand, _ = compute_v2_features(prob, img[None], sn[None], cs[None], ag[None],
                                              real_labels=rl[None], orient_mode="pca",
                                              gate_pmax=0.10, verbose=False)
                if not len(cand):
                    continue
                cand_nn = cand.copy()
                _add_orient(cand_nn, {0: prob[0]}, {0: img},
                            {0: float(diffim_mad_sigma(img))}, {0: sn}, {0: cs}, {0: ag},
                            orient_mode="nnhead")
                for cc, rf, sink in ((cand, rf_new, rows_new), (cand_nn, rf_old, rows_old)):
                    d = cc.copy()
                    d["image_id"] = pid
                    d["x"] = cc["x_centroid"]; d["y"] = cc["y_centroid"]
                    d["beta"] = cc["or_beta"]; d["length"] = cc["mf_length"]
                    d["score_rf"] = rf.predict_proba(cc[feats].fillna(0.0).to_numpy(np.float32))[:, 1]
                    sink.append(d[["image_id", "x", "y", "beta", "length", "score_rf"]])
        mn = pd.concat(rows_new, ignore_index=True) if rows_new else pd.DataFrame()
        mo = pd.concat(rows_old, ignore_index=True) if rows_old else pd.DataFrame()
        truth = truth[truth.image_id.isin(ids)]
        for thr in thrs:
            rn = evaluate_catalog(mn[mn.score_rf >= thr], truth, tol_px=10.0)[0]
            ro = evaluate_catalog(mo[mo.score_rf >= thr], truth, tol_px=10.0)[0]
            print(f"{s:12s} {thr:4.1f} | {rn['recall']*100:10.1f}% {rn['fp_per_panel']:10.2f} | "
                  f"{ro['recall']*100:9.1f}% {ro['fp_per_panel']:10.2f}", flush=True)
    print("RF EVAL COMPARE DONE", flush=True)


if __name__ == "__main__":
    main()
