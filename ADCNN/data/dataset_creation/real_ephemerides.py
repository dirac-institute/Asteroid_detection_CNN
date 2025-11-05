# real_ephemerides.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from lsst.daf.butler import Butler

from common import (
    ensure_dir, vsky_and_pa, detectors_covering_point, sky_to_pixel, draw_one_line
)
from io_utils import write_csv_rows, h5_init
import h5py

def build_one(butler: Butler, row: pd.Series):
    visit = int(row["FieldID"])
    ra = float(row["RA_deg"]); dec = float(row["Dec_deg"])
    xrate = float(row["RARateCosDec_deg_day"]); yrate = float(row["DecRate_deg_day"])

    refs = detectors_covering_point(butler, visit, ra, dec)
    if not refs:
        raise RuntimeError(f"No detector overlaps visit={visit} at RA,Dec={ra:.6f},{dec:.6f}")

    ref = refs[0]
    calexp = butler.get("calexp", dataId=ref.dataId)

    pixscale = calexp.wcs.getPixelScale().asArcseconds()
    exptime  = calexp.getInfo().getVisitInfo().getExposureTime()

    vsky_deg_day, pa_deg = vsky_and_pa(xrate, yrate)
    vsky_arcsec_s = (vsky_deg_day * 3600.0) / 86400.0
    trail_arcsec  = vsky_arcsec_s * exptime
    trail_px      = trail_arcsec / pixscale

    x0, y0 = sky_to_pixel(calexp, ra, dec)
    H = calexp.getDimensions().getY()
    W = calexp.getDimensions().getX()

    try:
        from lsst.geom import Point2D
        psf_w = int(calexp.psf.getLocalKernel(Point2D(x0, y0)).getWidth())
        thickness = max(psf_w, 3)
    except Exception:
        thickness = 5

    mask = np.zeros((H, W), dtype=np.uint8)
    draw_one_line(mask, (x0, y0), pa_deg, trail_px, true_value=1, line_thickness=thickness)
    mask = mask.astype(bool)

    img = calexp.image.array.astype("float32")

    meta = dict(
        image_id=None,
        visit=visit,
        detector=int(ref.dataId["detector"]),
        ObjID=str(row["ObjID"]),
        RA_deg=ra, Dec_deg=dec,
        vsky_deg_day=vsky_deg_day,
        PA_deg=pa_deg,
        trail_len_pix=float(trail_px),
        exposure_s=float(exptime),
        pixscale_arcsec=float(pixscale),
    )
    return img, mask, meta

def main():
    ap = argparse.ArgumentParser("Build REAL dataset from ephemerides CSV")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--collections", required=True)
    ap.add_argument("--ephemerides-csv", required=True)  # lsstcam_fast_trails_objects.csv
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--speed-thr", type=float, default=0.5)
    ap.add_argument("--random-subset", type=int, default=0)
    ap.add_argument("--train-test-split", type=float, default=0.1)
    ap.add_argument("--parallel", type=int, default=8)  # future: implement parallel mapping
    args = ap.parse_args()

    ensure_dir(args.save_path)
    df = pd.read_csv(args.ephemerides_csv)
    if "vsky_deg_day" not in df.columns:
        df["vsky_deg_day"] = np.sqrt(df["RARateCosDec_deg_day"]**2 + df["DecRate_deg_day"]**2)
    df = df[df["vsky_deg_day"] > args.speed_thr].copy()
    if args.random_subset and args.random_subset > 0:
        df = df.sample(args.random_subset, random_state=42).reset_index(drop=True)

    # split
    if 0.0 < args.train_test_split < 1.0:
        n_test = int(round(len(df) * args.train_test_split))
        df_test = df.iloc[:n_test].reset_index(drop=True)
        df_train = df.iloc[n_test:].reset_index(drop=True)
    else:
        df_train, df_test = df, pd.DataFrame([])

    # probe sizes
    butler = Butler(args.repo, collections=args.collections)
    probe = df_train.iloc[0] if len(df_train) else df_test.iloc[0]
    ref0 = detectors_covering_point(butler, int(probe["FieldID"]), float(probe["RA_deg"]), float(probe["Dec_deg"]))[0]
    dims = butler.get("calexp.dimensions", dataId=ref0.dataId)
    H, W = dims.getY(), dims.getX()

    out_train_h5 = Path(args.save_path) / "train.h5"
    out_test_h5  = Path(args.save_path) / "test.h5"
    csv_train = Path(args.save_path) / "train.csv"
    csv_test  = Path(args.save_path) / "test.csv"

    # init files
    if len(df_train):
        h5_init(out_train_h5, len(df_train), H, W)
    if len(df_test):
        h5_init(out_test_h5, len(df_test), H, W)

    # run serial (safe). If you want parallel, we can add shared memory + locks like your old code.
    def run_block(rows: pd.DataFrame, h5_path: Path, csv_path: Path):
        metas: List[Dict[str, Any]] = []
        with h5py.File(h5_path, "a") as f:
            for idx, row in rows.iterrows():
                try:
                    img, mask, meta = build_one(Butler(args.repo, collections=args.collections), row)
                    f["images"][idx] = img
                    f["masks"][idx]  = mask
                    meta["image_id"] = idx
                    metas.append(meta)
                except Exception as e:
                    print("[WARN]", e)
        write_csv_rows(csv_path, metas)

    if len(df_train):
        run_block(df_train, out_train_h5, csv_train)
    if len(df_test):
        run_block(df_test, out_test_h5, csv_test)

    print("Done.",
          f"train={len(df_train)} rows, test={len(df_test)} rows,",
          f"saved to {args.save_path}")

if __name__ == "__main__":
    main()
