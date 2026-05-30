"""Build the test_real per-sighting evaluation catalog from a streaming detection CSV.

Joins the flat detection catalog produced by ``stream_real_inference`` (`test_real_detections.csv`
with one row per detection) against the fast-mover ground-truth (`sv_fast_movers_*csv`,
one row per known asteroid sighting) using the same trail-overlap matcher the simulated
eval uses (``match_trail_catalogs``, tol 20 px). For each sighting we mark ``nn_detected``.

Optionally merges in ``per_sighting_forced_lsst.csv`` to carry ``lsst_psf_snr`` and the
unchanged LSST-stack baseline ``stack_detected`` -- those come from forced photometry on the
truth positions and don't depend on which detector model produced ``nn_detected``.

Outputs (next to the streaming detections CSV):
  ``test_real_per_sighting.csv`` -- truth + nn_detected (+ lsst_psf_snr/stack_detected if joined)
  ``test_real_per_panel_fp.csv`` -- per-panel detection counts (TP/FP split)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.evaluation.catalog_match import match_trail_catalogs


REPO = Path(__file__).resolve().parents[2]
DEF_DET = REPO / "Evaluation/catalogs_seg_v2/test_real_detections.csv"
DEF_TRUTH = REPO / "DATA/sv_fast_movers_for_karlo_fast_with_pixels_rerun.csv"
DEF_FORCED = REPO / "experiments/explore_simreal_gap/test_real_realistic/per_sighting_forced_lsst.csv"


def _canonical_image_id(df: pd.DataFrame, visit_col: str, detector_col: str) -> pd.Series:
    """Globally-unique integer panel id from (visit, detector). Same key in both frames so
    ``match_trail_catalogs`` groups correctly across shards (the streaming CSV's per-shard
    ``image_id`` is NOT globally unique).
    """
    key = df[visit_col].astype(str) + "_" + df[detector_col].astype(str)
    return pd.Categorical(key).codes.astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", default=str(DEF_DET),
                    help="streaming detection catalog (one row per detection)")
    ap.add_argument("--truth", default=str(DEF_TRUTH),
                    help="fast-mover truth CSV (one row per sighting)")
    ap.add_argument("--forced-lsst", default=str(DEF_FORCED),
                    help="per_sighting_forced_lsst.csv for SNR + stack baseline (optional)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: detections file's parent)")
    ap.add_argument("--tol-px", type=float, default=20.0,
                    help="trail-overlap tolerance (pre-chosen, same as simulated eval)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else Path(a.detections).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    det = pd.read_csv(a.detections)
    truth_raw = pd.read_csv(a.truth).rename(columns={"FieldID": "visit"})
    forced_path = Path(a.forced_lsst)
    if not forced_path.exists():
        raise SystemExit(f"forced-phot catalog required for accurate (x,y): {forced_path} missing")
    fl = pd.read_csv(forced_path)
    # IMPORTANT: the CSV's stale x,y are recomputed downstream by LSST forced phot using
    # WCS+ephemeris -- those are the canonical pixel positions to match against. Replace
    # the truth's x,y with the forced-phot x,y; rows without forced phot are dropped from
    # the per-sighting eval (~25% of sv_fast_movers had no overlap with the LSST DRP run).
    fl_x = fl[["ObjID", "visit", "detector", "x", "y", "beta",
               "lsst_psf_snr", "lsst_psf_flux", "stack_detected"]]
    truth = (truth_raw.drop(columns=["x", "y", "beta"], errors="ignore")
                       .merge(fl_x, on=["ObjID", "visit", "detector"], how="inner"))
    print(f"[load] detections={len(det):,}  truth_csv={len(truth_raw):,}  "
          f"truth-with-forced-phot={len(truth):,}", flush=True)

    keys = pd.concat([det[["visit", "detector"]],
                      truth[["visit", "detector"]]]).drop_duplicates().reset_index(drop=True)
    keys["image_id"] = np.arange(len(keys), dtype=int)
    det = det.drop(columns=["image_id"], errors="ignore").merge(keys, on=["visit", "detector"], how="left")
    truth = truth.merge(keys, on=["visit", "detector"], how="left")
    print(f"[panels] unique (visit,detector)={len(keys):,}  "
          f"in detections={det.image_id.nunique():,}  in truth={truth.image_id.nunique():,}",
          flush=True)

    # Match -- adds `nn_detected` to truth, `matched` to det. beta + x/y now from forced phot.
    truth_out, det_out, counts = match_trail_catalogs(
        det, truth, tol_px=a.tol_px,
        truth_length_col="trail_length", meas_length_col="length",
        flag_col="nn_detected",
    )
    print(f"[match] TP={counts['TP']}  FN={counts['FN']}  FP={counts['FP']}  "
          f"recall={counts['TP']/max(counts['TP']+counts['FN'],1):.3f}", flush=True)

    # Per-panel FP/TP split. `det_out['matched']` is True iff the detection matched any truth
    # trail; FP = !matched. Panel id is the canonical key above.
    panel_fp = (det_out.groupby("image_id")
                       .agg(n_det=("matched", "size"),
                            n_tp=("matched", "sum"),
                            visit=("visit", "first"),
                            detector=("detector", "first"))
                       .reset_index())
    panel_fp["nn_fp"] = panel_fp["n_det"] - panel_fp["n_tp"]
    # Carry every truth-bearing panel even if it produced no detections.
    truth_panels = truth_out[["image_id", "visit", "detector"]].drop_duplicates()
    panel_fp = truth_panels.merge(panel_fp.drop(columns=["visit", "detector"]),
                                  on="image_id", how="left").fillna(
        {"n_det": 0, "n_tp": 0, "nn_fp": 0})
    panel_fp[["n_det", "n_tp", "nn_fp"]] = panel_fp[["n_det", "n_tp", "nn_fp"]].astype(int)

    out_sightings = out_dir / "test_real_per_sighting.csv"
    out_panels = out_dir / "test_real_per_panel_fp.csv"
    truth_out.to_csv(out_sightings, index=False)
    panel_fp.to_csv(out_panels, index=False)

    n_panels = int(panel_fp["image_id"].nunique())
    nn_fp_total = int(panel_fp["nn_fp"].sum())
    print(f"[write] {out_sightings}  ({len(truth_out):,} rows)", flush=True)
    print(f"[write] {out_panels}  ({len(panel_fp):,} panels, "
          f"{nn_fp_total} FP total = {nn_fp_total/max(n_panels,1):.1f}/panel)", flush=True)


if __name__ == "__main__":
    main()
