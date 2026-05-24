"""ENTRY POINT (thin CLI) — run the trained v7 + RandomForest on diffim panels.

The actual end-to-end inference engine lives in ``ADCNN.inference.catalog`` — this module
is a convenience CLI that calls it with the deployed models in ``models/`` and writes the
detection catalog CSV. New code should prefer ``python -m ADCNN.inference.catalog``
(richer flags, panels.csv join for HelioLinC routing keys).

    python -m ADCNN.pipelines.run_inference --h5 DATA_DIFFIM/test_real/test.h5 --out detections.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

from ADCNN.inference.catalog import build_detection_catalog, InferenceConfig
from ADCNN.inference.rf_postproc import DEFAULT_THR

REPO = Path(__file__).resolve().parents[2]


def run(v7_ckpt, rf_pkl, h5_path, panel_ids, rf_thr, device, panels_csv=None):
    """Return the detection catalog (one row per kept detection). Thin wrapper around
    ``ADCNN.inference.catalog.build_detection_catalog`` — kept for back-compat."""
    return build_detection_catalog(h5_path, v7_ckpt, rf_pkl,
                                   config=InferenceConfig(rf_thr=rf_thr),
                                   panels_csv=panels_csv, device=device, panel_ids=panel_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--rf", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--h5", required=True, help="diffim panel h5 (images + real_labels)")
    ap.add_argument("--panels", help="optional panels.csv -> attach visit/detector/band")
    ap.add_argument("--rf-thr", type=float, default=DEFAULT_THR)
    ap.add_argument("--limit", type=int, default=0, help="0 = all panels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    pids = range(a.limit) if a.limit else None
    df = run(a.v7, a.rf, a.h5, pids, a.rf_thr, a.device, panels_csv=a.panels)
    df.to_csv(a.out, index=False)
    print(f"[inference] {len(df)} detections (score>={a.rf_thr}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
