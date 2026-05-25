"""B: benchmark the LSST trailed-source measurement (Veres + Naive plugins) against truth
on an injected DIFFIM panel, and compare to the ADCNN footprint-PCA orientation.

Runs in the lsst_distrib env (Butler + meas.extensions.trailedSources). For one or more
(visit, detector) pairs it:
  1. fetches PVI + kernel sources + overlapping template, makes a CLEAN diffim;
  2. injects N trails of KNOWN (x, y, trail_length, beta, mag) into a clone of the PVI;
  3. re-subtracts and runs DetectAndMeasure on the injected diffim with the Veres + Naive
     trailed-source plugins enabled (measueTrails path already in butler_tasks);
  4. matches each measured diaSource to the nearest injected trail by pixel centroid;
  5. reports recovery residuals for Veres and Naive (centroid, trail length, position
     angle) vs the injected truth.

This is the "fit it properly given the location" measurement stage the ADCNN detector
should hand off to. The ADCNN footprint-PCA orientation (mf_beta) benchmarked at ~8-10 deg
MAD on the same sim family; this script puts the LSST model fit on the same footing.

    setup lsst_distrib
    python experiments/heliolinc/veres_benchmark.py --n-detectors 2 --n-inject 25

NOTE: deliberately injects bright-ish trails (default mag 22.0) so detection is not the
bottleneck — we are measuring *measurement* precision, not completeness.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

import lsst.geom as geom
from lsst.daf.butler import Butler
from lsst.geom import Point2D
from lsst.source.injection.inject_exposure import ExposureInjectTask

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.data.dataset_creation.butler_tasks import (  # noqa: E402
    fetch_diffim_inputs, run_subtract, run_detect_diffim,
)
from ADCNN.utils.helpers import draw_one_line  # noqa: E402

REPO_BUTLER = "dp2_prep"
STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP = "lsst_cells_v2"
PIXSCALE = 0.2  # arcsec/px (LSSTCam), for reporting only

TRAIL_COLS = ("injection_id", "ra", "dec", "source_type", "trail_length", "mag", "beta")


def build_injection_catalog(rng, wcs, dims, n_inject, length_range, mag, forbidden=None):
    """Place `n_inject` trails of random length/angle on safe (non-overlapping) spots.
    Returns (astropy Table for ExposureInjectTask, truth DataFrame with pixel x,y)."""
    H, W = int(dims[0]), int(dims[1])
    forbid = np.zeros((H, W), bool) if forbidden is None else forbidden.astype(bool, copy=True)
    rows, truth = [], []
    for k in range(n_inject):
        L = float(rng.uniform(*length_range))
        ang = float(rng.uniform(0.0, 180.0))
        half = int(L / 2) + 25
        placed = False
        for _ in range(500):
            x = float(rng.uniform(half, W - 1 - half))
            y = float(rng.uniform(half, H - 1 - half))
            tmp = np.zeros((H, W), np.uint8)
            draw_one_line(tmp, [x, y], ang, L, true_value=1, line_thickness=4)
            stamp = tmp != 0
            if (stamp & forbid).any():
                continue
            forbid |= stamp
            placed = True
            break
        if not placed:
            continue
        sp = wcs.pixelToSky(x, y)
        rows.append([k, sp.getRa().asDegrees(), sp.getDec().asDegrees(), "Trail", L, float(mag), ang])
        truth.append(dict(injection_id=k, x=x, y=y, trail_length=L, beta=ang, mag=float(mag)))
    cat = Table(rows=rows, names=TRAIL_COLS,
                dtype=("int64", "float64", "float64", "str", "float64", "float64", "float64"))
    return cat, pd.DataFrame(truth)


def inject(pvi_clone, injection_catalog):
    task = ExposureInjectTask()
    res = task.run([injection_catalog], pvi_clone, pvi_clone.psf,
                   pvi_clone.photoCalib, pvi_clone.wcs)
    return res.output_exposure


def _ang_resid_deg(meas_deg, truth_deg):
    """Residual of two position angles, wrapped into (-90, 90] (PA is mod 180)."""
    d = (np.asarray(meas_deg) - np.asarray(truth_deg)) % 180.0
    return np.where(d > 90.0, d - 180.0, d)


def measure_one_detector(butler, dataId, rng, n_inject, length_range, mag):
    pvi, sources, template, phys, _ = fetch_diffim_inputs(
        butler, dataId, skymap=SKYMAP, stage3_collection=STAGE3)
    dims = (pvi.getBBox().getHeight(), pvi.getBBox().getWidth())

    # clean diffim -> forbidden footprints from real residuals
    sub_clean = run_subtract(template=template, science=pvi, sources=sources)
    det_clean = run_detect_diffim(science=pvi, matchedTemplate=sub_clean.matchedTemplate,
                                  difference=sub_clean.difference, threshold=5.0)
    forbid = np.zeros(dims, bool)
    for s in det_clean.diaSources:
        for span in s.getFootprint().spans:
            yy = span.getY()
            if 0 <= yy < dims[0]:
                forbid[yy, max(span.getX0(), 0):min(span.getX1(), dims[1] - 1) + 1] = True

    inj_cat, truth = build_injection_catalog(rng, pvi.wcs, dims, n_inject, length_range, mag, forbid)
    if not len(truth):
        return pd.DataFrame()

    pvi_inj = inject(pvi.clone(), inj_cat)
    sub_inj = run_subtract(template=template, science=pvi_inj, sources=sources)
    det_inj = run_detect_diffim(science=pvi_inj, matchedTemplate=sub_inj.matchedTemplate,
                                difference=sub_inj.difference, threshold=5.0, measueTrails=True)
    src = det_inj.diaSources

    # measured per-diaSource trail params (pixel)
    def col(name):
        return np.array([r[name] for r in src], float)
    cx = np.array([r.getX() for r in src], float)  # slot centroid = SdssCentroid
    cy = np.array([r.getY() for r in src], float)
    meas = pd.DataFrame(dict(
        cx=cx, cy=cy,
        veres_len=col("ext_trailedSources_Veres_length"),
        veres_ang=np.degrees(col("ext_trailedSources_Veres_angle")) % 180.0,
        veres_rchi=col("ext_trailedSources_Veres_rChiSq"),
        veres_flag=col("ext_trailedSources_Veres_flag"),
        naive_len=col("ext_trailedSources_Naive_length"),
        naive_ang=np.degrees(col("ext_trailedSources_Naive_angle")) % 180.0,
        naive_flag=col("ext_trailedSources_Naive_flag"),
    ))

    # match each truth to nearest measured centroid within 8 px
    out = []
    for _, t in truth.iterrows():
        d = np.hypot(meas.cx - t.x, meas.cy - t.y)
        j = int(np.argmin(d)) if len(d) else -1
        if j < 0 or d[j] > 8.0:
            continue
        m = meas.iloc[j]
        out.append(dict(visit=int(dataId["visit"]), detector=int(dataId["detector"]),
                        **t.to_dict(), match_px=float(d[j]),
                        veres_len=float(m.veres_len), veres_ang=float(m.veres_ang),
                        veres_rchi=float(m.veres_rchi), veres_flag=bool(m.veres_flag),
                        naive_len=float(m.naive_len), naive_ang=float(m.naive_ang),
                        naive_flag=bool(m.naive_flag)))
    return pd.DataFrame(out)


def report(df):
    def stats(res, lbl):
        res = np.asarray(res, float); res = res[np.isfinite(res)]
        if not len(res):
            print(f"  {lbl:22s} (no finite values)"); return
        med = np.median(res); mad = np.median(np.abs(res - med))
        print(f"  {lbl:22s} median={med:7.2f}  MAD={mad:6.2f}  std={res.std():7.2f}  n={len(res)}")

    print(f"\n==== Veres/Naive recovery vs truth (n={len(df)} matched injections) ====")
    for tag, lc, ac, fc in [("Veres", "veres_len", "veres_ang", "veres_flag"),
                            ("Naive", "naive_len", "naive_ang", "naive_flag")]:
        ok = df[~df[fc]] if fc in df else df
        print(f"-- {tag} (flag-clean n={len(ok)}/{len(df)}) --")
        stats(ok[lc] - ok["trail_length"], "trail_length resid px")
        stats(_ang_resid_deg(ok[ac], ok["beta"]), "position-angle resid deg")
    stats(df["match_px"], "centroid match px")
    # length residual after a linear de-bias, for direct comparison to the ADCNN +30px bias
    L = df["trail_length"].to_numpy(); M = df["veres_len"].to_numpy()
    g = np.isfinite(M)
    if g.sum() > 5:
        a, b = np.polyfit(L[g], M[g], 1)
        print(f"  Veres length linear fit: meas = {a:.3f}*L + {b:.2f}  (ideal 1,0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--where", default="instrument='LSSTCam' AND day_obs>=20250801 AND "
                    "day_obs<=20250921 AND band in ('r','i')")
    ap.add_argument("--n-detectors", type=int, default=2)
    ap.add_argument("--n-inject", type=int, default=25)
    ap.add_argument("--len-min", type=float, default=8.0)
    ap.add_argument("--len-max", type=float, default=60.0)
    ap.add_argument("--mag", type=float, default=22.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/veres_bench.csv"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    butler = Butler(REPO_BUTLER, collections=[STAGE3, STAGE2])
    refs = list(butler.registry.queryDatasets("preliminary_visit_image", where=a.where,
                                               findFirst=True))
    rng.shuffle(refs)

    frames, done = [], 0
    for ref in refs:
        if done >= a.n_detectors:
            break
        did = dict(instrument="LSSTCam", visit=int(ref.dataId["visit"]),
                   detector=int(ref.dataId["detector"]))
        try:
            df = measure_one_detector(butler, did, rng, a.n_inject, (a.len_min, a.len_max), a.mag)
        except Exception as e:
            print(f"  skip v={did['visit']} d={did['detector']}: {type(e).__name__}: {e}", flush=True)
            continue
        if len(df):
            frames.append(df)
            done += 1
            print(f"[{done}/{a.n_detectors}] v={did['visit']} d={did['detector']}: "
                  f"{len(df)} matched", flush=True)

    if not frames:
        print("no matched injections — nothing to report"); return
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(a.out, index=False)
    print(f"\nwrote {len(allf)} rows -> {a.out}")
    report(allf)


if __name__ == "__main__":
    main()
