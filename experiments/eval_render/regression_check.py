"""Regression: the refactored catalog engine reproduces the existing catalog's detections
on real test_5sigma panels (gate_pmax=0.10, as the shipped catalog was built)."""
import sys
from pathlib import Path
import pandas as pd
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0, str(REPO))
from ADCNN.inference.catalog import build_detection_catalog, InferenceConfig

def main():
    ids = [0, 1, 2, 3]
    new = build_detection_catalog(str(REPO/"DATA_DIFFIM/test_5sigma/test.h5"),
                                  str(REPO/"models/v7_diffim_scripted.pt"),
                                  str(REPO/"models/rf_postproc.pkl"),
                                  config=InferenceConfig(gate_pmax=0.10),
                                  panel_ids=ids, device="cuda", n_workers=4)
    old = pd.read_csv(REPO/"Evaluation/catalogs/test_5sigma_detections.csv")
    def key(d):
        return (d[d.image_id.isin(ids)].sort_values(["image_id", "x", "y"])
                [["image_id", "x", "y", "score_rf"]].round(3).reset_index(drop=True))
    kn, ko = key(new), key(old)
    print(f"new={len(kn)} det  existing(panels {ids})={len(ko)}  bit-equal: {kn.equals(ko)}", flush=True)
    # align by nearest integer pixel + report magnitude of any difference (fp16/cudnn noise vs real)
    import numpy as np
    a = kn.copy(); b = ko.copy()
    a["k"] = a.image_id*1e8 + a.x.round().astype(int)*1e4 + a.y.round().astype(int)
    b["k"] = b.image_id*1e8 + b.x.round().astype(int)*1e4 + b.y.round().astype(int)
    same_set = set(a.k) == set(b.k)
    m = a.merge(b, on="k", suffixes=("_n","_o"))
    print(f"same detection set (integer-pixel): {same_set}  | aligned {len(m)}/{len(a)}", flush=True)
    if len(m):
        print(f"max |Δx|={ (m.x_n-m.x_o).abs().max():.2e}  max |Δy|={ (m.y_n-m.y_o).abs().max():.2e}  "
              f"max |Δscore|={ (m.score_rf_n-m.score_rf_o).abs().max():.2e}", flush=True)

if __name__ == "__main__":
    main()
