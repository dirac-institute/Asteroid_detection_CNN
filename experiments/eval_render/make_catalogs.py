"""Produce measured detection catalogs for every test set (the inference half), then
print catalog-vs-catalog trail-overlap metrics as a sanity check. Run on a GPU node.

Writes Evaluation/catalogs/<set>_detections.csv (the measured catalog the analysis
notebook consumes). Inference only — no training, fixed RF operating point.
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.inference.catalog import build_detection_catalog
from ADCNN.evaluation.catalog_match import match_trail_catalogs
from ADCNN.inference.rf_postproc import DEFAULT_THR

V7 = REPO / "models/v7_diffim_scripted.pt"
RF = REPO / "models/rf_postproc.pkl"
OUT = REPO / "Evaluation/catalogs"; OUT.mkdir(parents=True, exist_ok=True)
TOL_PX = 20.0  # fixed trail-match tolerance (~PSF/candidate scale); chosen in advance


def main():
    SETS = ["test_5sigma", "test_4sigma", "test_3sigma", "test_real"]

    for name in SETS:
        d = REPO / "DATA_DIFFIM" / name
        h5 = d / "test.h5"
        if not h5.exists():
            print(f"[skip] {name}: no test.h5", flush=True); continue
        panels = d / "panels.csv"
        cat = build_detection_catalog(str(h5), str(V7), str(RF),
                                      panels_csv=str(panels) if panels.exists() else None,
                                      rf_thr=DEFAULT_THR, device="cuda")
        out_csv = OUT / f"{name}_detections.csv"
        cat.to_csv(out_csv, index=False)
        truth = pd.read_csv(d / "test.csv")
        _, _, c = match_trail_catalogs(cat, truth, tol_px=TOL_PX)
        recall = c["TP"] / max(c["TP"] + c["FN"], 1)
        fp_per_panel = c["FP"] / max(truth["image_id"].nunique(), 1)
        print(f"[{name}] {len(cat)} detections -> {out_csv.name} | "
              f"TP={c['TP']} FP={c['FP']} FN={c['FN']} | recall={recall:.3f} "
              f"FP/panel={fp_per_panel:.1f}  (tol_px={TOL_PX}, thr={DEFAULT_THR})", flush=True)

    print("MAKE-CATALOGS DONE", flush=True)


if __name__ == "__main__":
    main()
