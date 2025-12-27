from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import h5py

from lsst.daf.butler import Butler
import lsst.geom as geom

# ----------------- deterministic & threading -----------------
def set_global_seed(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_deterministic_threads():
    # Minimizes tiny run-to-run float differences from threaded math libs.
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(var, "1")
    try:
        import lsst.afw.math as afwMath
        afwMath.setNumThreads(1)
    except Exception:
        pass

# ----------------- geometry helpers -----------------
def vsky_and_pa(ra_rate_cosdec_deg_day: float, dec_rate_deg_day: float):
    """
    Sorcha rates: east=x=RARateCosDec, north=y=DecRate in deg/day.
    Returns total vsky [deg/day] and PA [deg East of North].
    """
    x = float(ra_rate_cosdec_deg_day)
    y = float(dec_rate_deg_day)
    vsky = np.hypot(x, y)
    pa = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return vsky, pa

def detectors_covering_point(butler: Butler, visit: int, ra_deg: float, dec_deg: float):
    # query all detectors for this visit
    refs = list(butler.registry.queryDatasets("calexp", where=f"visit={int(visit)}", findFirst=True))

    sp = geom.SpherePoint(geom.Angle(ra_deg, geom.degrees), geom.Angle(dec_deg, geom.degrees))

    hits = []
    for ref in refs:
        try:
            wcs  = butler.get("calexp.wcs", dataId=ref.dataId)
            dims = butler.get("calexp.dimensions", dataId=ref.dataId)  # has .x, .y
            x, y = wcs.skyToPixel(sp)
            if (0 <= x < dims.x) and (0 <= y < dims.y):
                hits.append(ref)
        except Exception:
            continue

    return hits

def sky_to_pixel(calexp, ra_deg: float, dec_deg: float):
    sp = geom.SpherePoint(geom.Angle(ra_deg, geom.degrees), geom.Angle(dec_deg, geom.degrees))
    x, y = calexp.wcs.skyToPixel(sp)
    return float(x), float(y)

def draw_one_line(mask: np.ndarray, origin, angle_deg_e_of_n, length_px, true_value=1, line_thickness=5):
    """
    Pure-numpy drawer: stamps disks along the line.
    angle: degrees East of North; image coords +x right, +y down.
    """
    import math
    h, w = mask.shape
    x0, y0 = origin
    theta = math.radians(90.0 - angle_deg_e_of_n)  # convert PA to image dx/dy
    dx, dy = math.cos(theta), math.sin(theta)

    steps = max(2, int(length_px))
    rr = max(1, int(line_thickness // 2))

    for t in np.linspace(0, length_px, steps):
        xc = int(round(x0 + t * dx))
        yc = int(round(y0 - t * dy))
        if 0 <= xc < w and 0 <= yc < h:
            x_min, x_max = max(0, xc - rr), min(w - 1, xc + rr)
            y_min, y_max = max(0, yc - rr), min(h - 1, yc + rr)
            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    if (xx - xc) ** 2 + (yy - yc) ** 2 <= rr ** 2:
                        mask[yy, xx] = true_value
    return mask

# ----------------- dataset writing -----------------
def init_h5(path, n, H, W):
    with h5py.File(path, "w") as f:
        f.create_dataset("images", shape=(n, H, W), dtype="float32")
        f.create_dataset("masks", shape=(n, H, W), dtype="bool")

def append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

# ----------------- build one sample -----------------
def build_one(butler: Butler, row: pd.Series):
    visit = int(row["FieldID"])
    ra = float(row["RA_deg"])
    dec = float(row["Dec_deg"])
    xrate = float(row["RARateCosDec_deg_day"])
    yrate = float(row["DecRate_deg_day"])
    objid = str(row.get("ObjID", ""))

    refs = detectors_covering_point(butler, visit, ra, dec)
    if not refs:
        raise RuntimeError(f"No calexp overlaps (visit={visit}, RA={ra:.6f}, Dec={dec:.6f})")

    ref = refs[0]
    calexp = butler.get("calexp", dataId=ref.dataId)

    pixscale_arcsec = calexp.wcs.getPixelScale().asArcseconds()
    exptime_s = calexp.getInfo().getVisitInfo().getExposureTime()

    vsky_deg_day, pa_deg = vsky_and_pa(xrate, yrate)
    vsky_arcsec_s = (vsky_deg_day * 3600.0) / 86400.0
    trail_len_arcsec = vsky_arcsec_s * exptime_s
    trail_len_px = trail_len_arcsec / pixscale_arcsec

    x0, y0 = sky_to_pixel(calexp, ra, dec)

    H = calexp.getDimensions().getY()
    W = calexp.getDimensions().getX()

    # Thickness: try local PSF width; fallback
    try:
        from lsst.geom import Point2D
        psf_w = int(calexp.psf.getLocalKernel(Point2D(x0, y0)).getWidth())
        thickness = max(psf_w, 3)
    except Exception:
        thickness = 5

    mask = np.zeros((H, W), dtype=np.uint8)
    draw_one_line(mask, (x0, y0), pa_deg, trail_len_px, true_value=1, line_thickness=thickness)
    mask = mask.astype(bool)

    img = calexp.image.array.astype("float32")

    meta = dict(
        visit=visit,
        detector=int(ref.dataId["detector"]),
        ObjID=objid,
        RA_deg=ra,
        Dec_deg=dec,
        RARateCosDec_deg_day=xrate,
        DecRate_deg_day=yrate,
        vsky_deg_day=vsky_deg_day,
        PA_deg=pa_deg,
        trail_len_px=float(trail_len_px),
        exposure_s=float(exptime_s),
        pixscale_arcsec=float(pixscale_arcsec),
    )
    return img, mask, meta

# ----------------- main pipeline -----------------
def main():
    set_deterministic_threads()

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--collections", required=True)
    ap.add_argument("--ephem-csv", required=True)  # Jake’s file or the processed objects CSV
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--speed-thr", type=float, default=0.5)        # deg/day
    ap.add_argument("--random-subset", type=int, default=0)        # 0=all
    ap.add_argument("--train-frac", type=float, default=0.9)       # fraction to train
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    set_global_seed(args.seed)

    df = pd.read_csv(args.ephem_csv)

    # Ensure vsky exists
    if "vsky_deg_day" not in df.columns:
        df["vsky_deg_day"] = np.sqrt(df["RARateCosDec_deg_day"]**2 + df["DecRate_deg_day"]**2)

    df = df[df["vsky_deg_day"] > args.speed_thr].copy()

    # Deterministic ordering before any subsetting/splitting
    df = df.sort_values(["FieldID", "ObjID", "RA_deg", "Dec_deg"]).reset_index(drop=True)

    if args.random_subset > 0:
        df = df.sample(n=args.random_subset, random_state=args.seed).sort_values(
            ["FieldID", "ObjID", "RA_deg", "Dec_deg"]
        ).reset_index(drop=True)

    n = len(df)
    n_train = int(round(n * args.train_frac))
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_test  = df.iloc[n_train:].reset_index(drop=True)

    butler = Butler(args.repo, collections=args.collections)

    # Probe dimensions from first row
    probe = df_train.iloc[0] if len(df_train) else df_test.iloc[0]
    probe_refs = detectors_covering_point(butler, int(probe["FieldID"]), float(probe["RA_deg"]), float(probe["Dec_deg"]))
    dims = butler.get("calexp.dimensions", dataId=probe_refs[0].dataId)
    H, W = int(dims.y), int(dims.x)

    train_h5 = os.path.join(args.save_path, "train.h5")
    test_h5  = os.path.join(args.save_path, "test.h5")
    train_csv = os.path.join(args.save_path, "train.csv")
    test_csv  = os.path.join(args.save_path, "test.csv")

    # Fresh outputs
    for p in [train_h5, test_h5, train_csv, test_csv]:
        if os.path.exists(p):
            os.remove(p)

    if len(df_train):
        init_h5(train_h5, len(df_train), H, W)
    if len(df_test):
        init_h5(test_h5, len(df_test), H, W)

    def run_block(rows: pd.DataFrame, h5_path: str, csv_path: str):
        metas = []
        with h5py.File(h5_path, "a") as f:
            for idx, row in rows.iterrows():
                try:
                    img, mask, meta = build_one(Butler(args.repo, collections=args.collections), row)
                    f["images"][idx] = img
                    f["masks"][idx] = mask
                    meta["image_id"] = idx
                    metas.append(meta)
                except Exception as e:
                    # keep indices aligned: write zeros for failed rows
                    print("[WARN]", e)
                    f["images"][idx] = np.zeros((H, W), dtype="float32")
                    f["masks"][idx] = np.zeros((H, W), dtype="bool")
        append_csv(csv_path, metas)

    if len(df_train):
        run_block(df_train, train_h5, train_csv)
    if len(df_test):
        run_block(df_test, test_h5, test_csv)

    print(f"Done. train={len(df_train)} test={len(df_test)} saved to {args.save_path}")

if __name__ == "__main__":
    main()
