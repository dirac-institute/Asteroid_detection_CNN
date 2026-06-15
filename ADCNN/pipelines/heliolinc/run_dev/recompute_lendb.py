#!/usr/bin/env python3
"""Post-hoc v2_D MF_LEN recalibration: recompute len_db AND trail endpoints from the stored RAW
matched-filter length, using v2_D-specific (offset, slope) — no re-detection.

len_db = clip((length_raw - offset)/slope, 0)        (v1: offset 33.4, slope 0.887; v2_D fit ~7.8/0.934)
endpoints: (x,y) +/- 0.5*len_db*(cos beta, sin beta) -> sky via the panel FITS-approx WCS (manifest wcs_json).
Everything else (ra,dec,mjd,score,mf_snr,...) is carried through unchanged. Frozen len_db>=6 floor and all
alert cuts are applied LATER by the linker/pair stage on these corrected values — unchanged here.

Usage: recompute_lendb.py --src run_dev/v2_D_s2 --manifests run_dev --out run_dev/v2_D_s2cal \
                          --offset 7.8 --slope 0.934 --fields 0 1 2 ...
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

HERE = os.path.dirname(os.path.abspath(__file__))


def wcs_from_json(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        h = fits.Header()
        for k, v in json.loads(s).items():
            if k in ("COMMENT", "HISTORY") or v is None:
                continue
            h[k] = v
        w = WCS(h)
        return w if w.has_celestial else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--slope", type=float, required=True)
    ap.add_argument("--fields", type=int, nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(f"{a.src}/adcnn_dets_masked_*.csv"))
    ks = [int(f.split("masked_")[1].split(".")[0]) for f in files]
    if a.fields is not None:
        ks = [k for k in ks if k in a.fields]
    for k in ks:
        d = pd.read_csv(f"{a.src}/adcnn_dets_masked_{k}.csv")
        if "length_raw" not in d.columns:
            print(f"[recompute] field {k}: NO length_raw -- skip", flush=True); continue
        man = pd.read_csv(f"{a.manifests}/manifest_{k}.csv")
        wmap = {(int(r.visit), int(r.detector)): wcs_from_json(getattr(r, "wcs_json", None))
                for r in man.itertuples()}
        len_db = np.clip((d.length_raw.to_numpy() - a.offset) / a.slope, 0.0, None)
        d["len_db"] = len_db; d["length"] = len_db
        br = np.radians(d.beta.to_numpy(np.float64))
        hdx = 0.5 * len_db * np.cos(br); hdy = 0.5 * len_db * np.sin(br)
        ra0 = np.full(len(d), np.nan); dec0 = ra0.copy(); ra1 = ra0.copy(); dec1 = ra0.copy()
        x = d.x.to_numpy(np.float64); y = d.y.to_numpy(np.float64)
        n_nowcs = 0
        for key, idx in d.groupby(["visit", "detector"]).groups.items():
            w = wmap.get((int(key[0]), int(key[1])))
            if w is None:
                n_nowcs += len(idx)
                continue
            ii = np.array(list(idx))
            p0 = w.all_pix2world(np.stack([x[ii] - hdx[ii], y[ii] - hdy[ii]], 1), 0)
            p1 = w.all_pix2world(np.stack([x[ii] + hdx[ii], y[ii] + hdy[ii]], 1), 0)
            ra0[ii], dec0[ii] = p0[:, 0], p0[:, 1]; ra1[ii], dec1[ii] = p1[:, 0], p1[:, 1]
        d["ra0"], d["dec0"], d["ra1"], d["dec1"] = ra0, dec0, ra1, dec1
        if n_nowcs:
            print(f"[recompute] WARNING field {k}: {n_nowcs} dets had NO panel WCS -> NaN endpoints "
                  f"(linker will drop them; check manifest wcs_json)", flush=True)
        d.to_csv(f"{a.out}/adcnn_dets_masked_{k}.csv", index=False)
        print(f"[recompute] field {k}: {len(d)} dets, len_db median {np.median(len_db):.1f} "
              f">=6px {100*(len_db>=6).mean():.0f}% -> {a.out}", flush=True)
    print("RECOMPUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
