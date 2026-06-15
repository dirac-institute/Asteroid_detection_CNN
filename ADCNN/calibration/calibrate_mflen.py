#!/usr/bin/env python3
"""Fit the matched-filter trail-length de-bias ``(offset, slope)`` for the active stage-1 and
CONFIRM it reproduces the frozen ``mf_len_debias`` in the pipeline config.

Why this exists: a domain-adapted stage-1 has its own matched-filter "ends-bloom", so the raw
measured trail length ``length_raw`` is biased relative to the true injected length. The linker
reads ``len_db = clip((length_raw - offset) / slope, 0)`` and gates on ``len_db >= 6 px``; if
``(offset, slope)`` is stale (fit to a different stage-1) the gate deletes real detections (the
v2_D "0-pairs" failure). The de-bias therefore travels WITH the model (see ``ADCNN/config.py``)
and must be re-derived whenever stage-1 changes.

The fit (re-derivable methodology, ADCNN_V2_MFLEN_DECISION.md): match each injected faint sighting
to the nearest detection in (x,y) within ``--match-px`` per (visit, detector), keep detections at
score >= ``--score-min`` (the alert floor -- the population the de-bias serves), and OLS-fit
``length_raw ~ slope * trail_length + offset``. Apply is the inverse: ``(length_raw-offset)/slope``.

Two input modes:
  * Level-1 (clean-checkout): ``--fit-csv`` -> a small committed table of matched
    (trail_length, length_raw) pairs. Reproducible with no Butler, no 14 GB h5, no on-disk dev dirs.
  * Level-2 (training-time): ``--src``/``--inj`` -> the on-disk dev detection + injection dirs;
    ``--out-csv`` extracts the matched pairs (commit that CSV to enable Level-1).

Usage:
    # Level-1 confirm from the committed pairs:
    PYTHONPATH=. python -m ADCNN.calibration.calibrate_mflen --fit-csv <pairs.csv>
    # Level-2 extract from dev dirs (writes the committable pairs CSV):
    PYTHONPATH=. python -m ADCNN.calibration.calibrate_mflen \
        --src ADCNN/pipelines/heliolinc/run_dev/v2_D_s2 --inj ADCNN/pipelines/heliolinc/run_dev \
        --out-csv ADCNN/calibration/mflen_fit_pairs.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.config import REPO, load_pipeline

# pre-declared fit configuration (the methodology, fixed before measuring)
SCORE_MIN = 0.80          # fit on the alert-floor population the de-bias serves
MATCH_PX = 10.0           # positional match tolerance (the repo-standard 10 px)

# confirm tolerances: the frozen values are a rounded prior fit; require the re-fit to land within
# the fit's own uncertainty scale (not bit-exact -- a different field subset shifts the last digit).
TOL_OFFSET = 0.5          # px
TOL_SLOPE = 0.03

DEFAULT_FIT_CSV = REPO / "ADCNN" / "calibration" / "mflen_fit_pairs.csv"


class MFLenCalibrationError(RuntimeError):
    """Raised when the re-fit de-bias does not reproduce the frozen pipeline values."""


def extract_pairs(src_dir, inj_dir, score_min=SCORE_MIN, match_px=MATCH_PX):
    """Match injected sightings to detections and return a DataFrame of (trail_length,
    length_raw, field, visit, detector, score) pairs."""
    from scipy.spatial import cKDTree
    out = []
    files = sorted(glob.glob(f"{src_dir}/adcnn_dets_masked_*.csv"))
    if not files:
        raise SystemExit(f"no adcnn_dets_masked_*.csv under {src_dir}")
    for f in files:
        k = int(f.split("masked_")[1].split(".")[0])
        ij = f"{inj_dir}/inject_{k}.csv"
        if not os.path.exists(ij):
            continue
        d = pd.read_csv(f)
        inj = pd.read_csv(ij)
        if "length_raw" not in d.columns:
            continue
        if score_min is not None and "score" in d.columns:
            d = d[d.score >= score_min]
        for (v, det), g in inj.groupby(["visit", "detector"]):
            dd = d[(d.visit == v) & (d.detector == det)]
            if len(dd) == 0 or len(g) == 0:
                continue
            tree = cKDTree(dd[["x", "y"]].to_numpy())
            dist, idx = tree.query(g[["x", "y"]].to_numpy(), k=1)
            m = dist <= match_px
            gg = g[m]
            ddm = dd.iloc[idx[m]]
            for tl, lr, sc in zip(gg["trail_length"].to_numpy(),
                                  ddm["length_raw"].to_numpy(),
                                  (ddm["score"].to_numpy() if "score" in ddm else
                                   np.full(len(ddm), np.nan))):
                out.append({"trail_length": float(tl), "length_raw": float(lr),
                            "field": k, "visit": int(v), "detector": int(det), "score": float(sc)})
    return pd.DataFrame(out)


def fit(df):
    """OLS fit length_raw ~ slope*trail_length + offset. Returns dict {offset, slope, fit_n,
    residual_px}. residual_px is on the APPLIED (de-biased) scale = std((raw-fit)/slope)."""
    x = df["trail_length"].to_numpy(float)
    y = df["length_raw"].to_numpy(float)
    if len(x) < 10:
        raise MFLenCalibrationError(f"too few matched pairs to fit ({len(x)})")
    slope, offset = np.polyfit(x, y, 1)
    resid = y - (slope * x + offset)
    return {"offset": float(offset), "slope": float(slope), "fit_n": int(len(x)),
            "residual_px": float((resid / slope).std())}


def confirm_against_frozen(fitted, pipeline=None, tol_offset=TOL_OFFSET, tol_slope=TOL_SLOPE):
    """Assert the re-fit reproduces the active pipeline's frozen de-bias (within fit tolerance)."""
    pipe = pipeline or load_pipeline()
    do = abs(fitted["offset"] - pipe.mf_len_offset)
    ds = abs(fitted["slope"] - pipe.mf_len_slope)
    if do > tol_offset or ds > tol_slope:
        raise MFLenCalibrationError(
            "re-fit MF_LEN de-bias does NOT reproduce the frozen pipeline values "
            f"({pipe.name}).\nThis is a FINDING, not a knob.\n"
            f"  offset: re-fit {fitted['offset']:.4f} vs frozen {pipe.mf_len_offset} (|d|={do:.4f} > {tol_offset})\n"
            f"  slope : re-fit {fitted['slope']:.4f} vs frozen {pipe.mf_len_slope} (|d|={ds:.4f} > {tol_slope})")
    return pipe


def write_mflen(out_dir, fitted, provenance):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"offset": round(fitted["offset"], 4), "slope": round(fitted["slope"], 4),
           "fit_n": fitted["fit_n"], "residual_px": round(fitted["residual_px"], 3),
           "score_min": SCORE_MIN, "match_px": MATCH_PX, "provenance": provenance,
           "apply": "len_db = clip((length_raw - offset) / slope, 0)"}
    path = out / "mflen.json"
    json.dump(rec, open(path, "w"), indent=2)
    return path, rec


def run(fit_csv=None, src=None, inj=None, out_csv=None, out=None, confirm=True):
    """Load (or extract) matched pairs, fit, optionally confirm against frozen, optionally write."""
    if src and inj:
        df = extract_pairs(src, inj)
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
        provenance = f"OLS fit on {src} matched to {inj} (score>={SCORE_MIN}, {MATCH_PX}px)"
    else:
        path = fit_csv or DEFAULT_FIT_CSV
        if not os.path.exists(path):
            raise SystemExit(f"fit-csv {path} not found -- extract it first with --src/--inj/--out-csv")
        df = pd.read_csv(path)
        provenance = f"OLS fit on committed pairs {os.path.basename(str(path))}"
    fitted = fit(df)
    if confirm:
        confirm_against_frozen(fitted)
    rec = None
    if out is not None:
        _, rec = write_mflen(out, fitted, provenance)
    return fitted, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-csv", default=None, help="committed matched-pairs CSV (Level-1)")
    ap.add_argument("--src", default=None, help="on-disk dev detection dir (Level-2)")
    ap.add_argument("--inj", default=None, help="on-disk injection-truth dir (Level-2)")
    ap.add_argument("--out-csv", default=None, help="write the extracted matched pairs here (commit it)")
    ap.add_argument("--out", default=None, help="release dir to write mflen.json")
    ap.add_argument("--no-confirm", action="store_true", help="skip assert-against-frozen (dev only)")
    a = ap.parse_args()
    fitted, _ = run(fit_csv=a.fit_csv, src=a.src, inj=a.inj, out_csv=a.out_csv, out=a.out,
                    confirm=not a.no_confirm)
    print(f"MF_LEN re-fit: offset={fitted['offset']:.4f}  slope={fitted['slope']:.4f}  "
          f"n={fitted['fit_n']}  residual={fitted['residual_px']:.2f}px (applied scale)")
    if a.out_csv:
        print(f"wrote matched pairs -> {a.out_csv}")
    if not a.no_confirm:
        pipe = load_pipeline()
        print(f"CONFIRMED: reproduces frozen de-bias offset={pipe.mf_len_offset} slope={pipe.mf_len_slope} "
              f"(pipeline '{pipe.name}', within tol offset±{TOL_OFFSET}/slope±{TOL_SLOPE}).")
    if a.out:
        print(f"wrote mflen.json -> {a.out}/")


if __name__ == "__main__":
    main()
