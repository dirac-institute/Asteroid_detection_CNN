"""Flag ADCNN detections that sit on LSST diffim artifact mask planes (SPIKE, SAT, CR, STREAK, ...).

A detection whose trail overlaps Rubin's own pixel-level artifact flags is, by the survey's own
reduction, an instrumental artifact (diffraction/saturation spike, cosmic ray, satellite streak,
saturated/bad pixel) — not an astronomical source. This is the principled, defensible FP filter
(vs. an ad-hoc trail-angle cut). We sample each detection's TRAIL (centroid +/- length/2 along
beta, with a small width) and record, per artifact plane, the fraction of trail pixels flagged.

Reads the mask plane directly from the diffim FITS (HDU 'MASK') + per-file MP_* bit assignments
(they are NOT fixed across files) — no Butler / LSST stack needed. Output: the input catalog plus
one fraction column per artifact plane + `art_frac` (max over the instrumental planes).
"""
from __future__ import annotations
import argparse
import os
import warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

# Instrumental artifact planes whose presence on a trail marks it non-astronomical.
# DETECTED_NEGATIVE included: a real positive moving source is never on a negative-subtraction
# residual; measured 0% of catalogued objects vs 9% of FP — a clean, TP-safe dipole/bad-subtraction
# flag. (NOT DETECTED / INJECTED / *_PSF / NOT_DEBLENDED — those don't mark artifacts.)
ART_PLANES = ["SPIKE", "SAT", "CR", "STREAK", "CROSSTALK", "SAT_TEMPLATE", "SATURATED_TEMPLATE",
              "BAD", "SUSPECT", "EDGE", "SENSOR_EDGE", "NO_DATA", "VIGNETTED", "INTRP",
              "ITL_DIP", "CLIPPED", "UNMASKEDNAN", "HIGH_VARIANCE", "DETECTED_NEGATIVE"]
NSAMP = 25       # points sampled along the trail
HALFWID = 1      # +/- pixels perpendicular sampling (trail core)


def _panel_flags(args):
    fits_path, recs = args
    from ADCNN.inference.diffim_io import open_diffim
    out = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open_diffim(fits_path, memmap=True) as h:
                mhdu = h["MASK"]
                bit = {k[3:]: int(v) for k, v in mhdu.header.items() if k.startswith("MP_")}
                mask = mhdu.data
        H, W = mask.shape
        planes = [(p, bit[p]) for p in ART_PLANES if p in bit]
        for r in recs:
            x, y = float(r["x"]), float(r["y"]); beta = np.radians(float(r["beta"]))
            L = max(float(r.get("len_db", r.get("length", 0.0))), 1.0)
            dx, dy = np.cos(beta), np.sin(beta)
            ts = np.linspace(-0.5 * L, 0.5 * L, NSAMP)
            xs = np.clip(np.round(x + ts * dx).astype(int), 0, W - 1)
            ys = np.clip(np.round(y + ts * dy).astype(int), 0, H - 1)
            # perpendicular widen so a thin trail isn't missed by rounding
            px, py = -dy, dx
            acc = np.zeros(NSAMP, dtype=np.int64)
            for w in range(-HALFWID, HALFWID + 1):
                xw = np.clip(xs + int(round(w * px)), 0, W - 1)
                yw = np.clip(ys + int(round(w * py)), 0, H - 1)
                acc |= mask[yw, xw].astype(np.int64)
            row = {"detid": r["detid"]}
            for p, b in planes:
                row[f"m_{p}"] = float(np.mean((acc >> b) & 1))
            out.append(row)
    except Exception as e:
        for r in recs:
            out.append({"detid": r["detid"], "_err": f"{type(e).__name__}: {e}"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="catalog with detid,x,y,beta,length/len_db,visit,detector")
    ap.add_argument("--manifest", required=True, help="visit,detector,fits_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    d = pd.read_csv(a.dets)
    if "detid" not in d:
        d = d.reset_index(drop=True); d["detid"] = d.index
    man = pd.read_csv(a.manifest)[["visit", "detector", "fits_path"]].drop_duplicates(["visit", "detector"])
    d = d.merge(man, on=["visit", "detector"], how="inner")
    tasks = []
    for (v, det), g in d.groupby(["visit", "detector"]):
        tasks.append((g.fits_path.iloc[0], g.to_dict("records")))
    print(f"[mask] {len(d)} dets over {len(tasks)} panels", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_panel_flags, t) for t in tasks])):
            rows.extend(fut.result())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(tasks)} panels", flush=True)
    fl = pd.DataFrame(rows)
    mcols = [c for c in fl.columns if c.startswith("m_")]
    fl["art_frac"] = fl[mcols].max(axis=1) if mcols else 0.0
    out = d.merge(fl, on="detid", how="left")
    # a FITS that failed to read leaves its dets with no flag row -> NaN; treat unmeasured as NO artifact
    # (0.0) so a mask-read failure can NEVER silently drop a real detection downstream (the linker's
    # `art_frac < cut` is False for NaN, which would drop it). TP-safe: unchecked dets pass the mask cut.
    nmiss = int(out.art_frac.isna().sum())
    for c in mcols + ["art_frac"]:
        out[c] = out[c].fillna(0.0)
    out = out.drop(columns=[c for c in ("_err",) if c in out.columns])   # internal error marker, not a data column
    out.to_csv(a.out, index=False)
    if nmiss:
        print(f"[mask] WARNING: {nmiss} dets had no mask (FITS read fail) -> art_frac=0 (kept)", flush=True)
    print(f"[mask] wrote {len(out)} -> {a.out}", flush=True)
    print(f"[mask] planes found: {[c[2:] for c in mcols]}", flush=True)
    print(f"[mask] dets with ANY artifact on trail (art_frac>0): {int((out.art_frac>0).sum())} "
          f"({100*(out.art_frac>0).mean():.0f}%)", flush=True)


if __name__ == "__main__":
    main()
