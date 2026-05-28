"""Precise trail MEASUREMENT for tracklet construction: run the LSST Veres trailed-PSF fit at
every (trailed) ADCNN detection, on the real difference image with its real per-panel PSF, and
emit accurate sky endpoints (ra0,dec0,ra1,dec1) — the inputs trail_tracklets.py turns into
HelioLinC tracklets.

Why Veres (not the PCA + de-bias approximation): with the real PSF, VeresModel recovers the true
trail length nearly unbiased (+2px, validated) and gives endpoints from a forward-model fit that
accounts for the PSF at the trail ends — the most precise measurement available. VeresModel is
driven directly (seeded with ADCNN x,y,length,beta), bypassing the Naive/SdssShape plugin chain
that crashes on long trails.

Per (visit,detector): butler.get the diffim PSF (component fetch), read the diffim FITS
(image+variance+WCS), build an Exposure, fit each detection, convert endpoints -> sky. Parallel
across panels (lsst_distrib env, CPU).

    setup lsst_distrib
    python veres_measure_catalog.py --dets run_wide/adcnn_dets.csv --manifest run_wide/manifest.csv \
        --length-min 6 --out run_wide_v2/adcnn_dets_veres.csv --workers 32
"""
from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import as_completed
# Pin BLAS/OMP to 1 thread BEFORE numpy imports: we parallelise across panels, so per-worker
# thread pools only oversubscribe the node (a 60-worker run hit load avg ~1700 and crawled).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_BUTLER = None  # one Butler per worker process (init once, reused across panels)


def _worker_init():
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    global _BUTLER
    from lsst.daf.butler import Butler
    _BUTLER = Butler("dp2_prep", collections=[STAGE4])

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"


def _fit_panel(args):
    """Worker: fit Veres for all detections on one (visit,detector). Returns list of row dicts."""
    visit, detector, fits_path, dets = args
    import warnings
    warnings.simplefilter("ignore")
    import numpy as np
    import scipy.optimize as sciOpt
    from astropy.io import fits
    from astropy.wcs import WCS
    import lsst.afw.image as afwImage
    from lsst.meas.extensions.trailedSources import VeresModel
    from lsst.daf.butler import Butler
    import lsst.geom as geom

    try:
        b = _BUTLER  # reused per-worker Butler (set in _worker_init)
        psf = b.get("difference_image.psf", dataId={"instrument": "LSSTCam",
                                                    "visit": int(visit), "detector": int(detector)})
        with fits.open(fits_path, memmap=False) as h:
            img = np.nan_to_num(h[1].data.astype(np.float32))
            var = np.nan_to_num(h[3].data.astype(np.float32))
            wcs = WCS(h[1].header)
    except Exception as e:
        return [("ERR", f"v{visit} d{detector}: {type(e).__name__}: {e}")]

    H, W = img.shape
    exp = afwImage.ExposureF(W, H)
    exp.image.array[:] = img
    exp.variance.array[:] = np.where(var > 0, var, np.median(var[var > 0]) if (var > 0).any() else 1.0)
    exp.setPsf(psf)
    try:
        psf_sig = float(psf.computeShape(psf.getAveragePosition()).getDeterminantRadius())
    except Exception:
        psf_sig = 2.0

    out = []
    for d in dets:
        x, y = float(d["x"]), float(d["y"])
        L0 = max(float(d.get("length", 10.0)), 2.0)   # seed: `length` is already de-biased (catalog.py MF_LEN_*)
        half = int(L0 / 2 + 6 * psf_sig + 6)
        bb = geom.Box2I(geom.Point2I(int(x) - half, int(y) - half), geom.Extent2I(2 * half + 1, 2 * half + 1))
        bb.clip(exp.getBBox())
        if bb.getWidth() < 8 or bb.getHeight() < 8:
            continue
        cut = exp.Factory(exp, bb)
        model = VeresModel(cut)
        seed = np.array([x, y, max(float(d.get("mf_snr", 0)) * 100, 1000.0), L0, np.radians(float(d.get("beta", 0.0)))])
        # BOUNDED gradient optimizer: prevents the Nelder-Mead runaway (length -> 1e7 in noise)
        # and is faster (uses VeresModel.gradient). Length bounded to a physical [1,300]px.
        bounds = [(x - 15, x + 15), (y - 15, y + 15), (0.0, 1e7), (1.0, 300.0), (-np.pi, np.pi)]
        try:
            r = sciOpt.minimize(model, seed, method="L-BFGS-B", jac=model.gradient, bounds=bounds,
                                options=dict(maxiter=500))
            xc, yc, flux, Lf, th = r.x
            rchi = float(r.fun / max(cut.image.array.size - 6, 1))
        except Exception:
            continue
        if not np.isfinite(Lf) or Lf < 2.0 or Lf > 295.0:   # reject collapsed/runaway fits
            continue
        a = Lf / 2.0
        ex0, ey0 = xc - a * np.cos(th), yc - a * np.sin(th)
        ex1, ey1 = xc + a * np.cos(th), yc + a * np.sin(th)
        (ra, dec), (ra0, dec0), (ra1, dec1) = wcs.all_pix2world([[xc, yc], [ex0, ey0], [ex1, ey1]], 0)
        out.append(dict(detid=int(d.get("detid", -1)), mjd=float(d["mjd"]), ra=float(ra), dec=float(dec),
                        ra0=float(ra0), dec0=float(dec0), ra1=float(ra1), dec1=float(dec1),
                        len_db=float(abs(Lf)), theta=float(np.degrees(th) % 180), veres_rchi=rchi,
                        mag=float(d.get("mag", 21.0)), band=str(d.get("band", "r"))[:1] or "r",
                        obscode="I11", score_rf=float(d.get("score_rf", np.nan)),
                        visit=int(visit), detector=int(detector)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", default=str(REPO / "experiments/heliolinc/run_wide/adcnn_dets.csv"))
    ap.add_argument("--manifest", default=str(REPO / "experiments/heliolinc/run_wide/manifest.csv"))
    # length-min is a SPEED pre-gate ONLY: it picks which detections are worth the (expensive) Veres
    # fit, using the ADCNN trail length because the accurate Veres length isn't computed yet. It is
    # NOT a quality cut and is NOT a score filter -- the stage-2 FP/score cut already happened ONCE at
    # detect (CNN score). The precise >1 deg/day length cut is applied later, in clean_fp on the
    # Veres-measured length. Keep this loose enough to not pre-drop borderline fast movers.
    ap.add_argument("--length-min", type=float, default=6.0,
                    help="ADCNN trail-length pre-gate (px) for which dets to Veres-fit (speed only, NOT a cut)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/run_wide_v2/adcnn_dets_veres.csv"))
    a = ap.parse_args()

    d = pd.read_csv(a.dets)
    d = d[d.length >= a.length_min].copy()   # ADCNN-length speed pre-gate only; no score re-filter (done at detect)
    man = pd.read_csv(a.manifest)[["visit", "detector", "fits_path"]].drop_duplicates(["visit", "detector"])
    d = d.merge(man, on=["visit", "detector"], how="inner")
    print(f"[veres-measure] {len(d)} dets to fit over {d.groupby(['visit','detector']).ngroups} panels "
          f"(ADCNN length>={a.length_min}px pre-gate; score cut already applied at detect)", flush=True)

    tasks = []
    for (v, det), g in d.groupby(["visit", "detector"]):
        tasks.append((int(v), int(det), g.fits_path.iloc[0], g.to_dict("records")))

    rows, errs, done = [], 0, 0
    t0 = time.time()
    ntot = len(tasks)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_worker_init) as ex:
        futs = [ex.submit(_fit_panel, t) for t in tasks]
        for f in as_completed(futs):     # out of order -> progress reflects real completions
            res = f.result()
            done += 1
            if res and isinstance(res[0], tuple) and res[0][0] == "ERR":
                errs += 1
            else:
                rows.extend(res)
            if done % 50 == 0 or done == ntot:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (ntot - done) / max(rate, 1e-9)
                print(f"  [{done}/{ntot} panels | {len(rows)} fits | {errs} errs] "
                      f"{rate:.1f} panel/s, elapsed {el/60:.1f}m, ETA {eta/60:.1f}m", flush=True)

    out = pd.DataFrame(rows)
    if "detid" not in out or (out.detid < 0).all():
        out.insert(0, "detid", range(len(out)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[veres-measure] wrote {len(out)} Veres-measured detections -> {a.out}", flush=True)
    if len(out):
        print(f"  Veres length px: med {out.len_db.median():.1f} | rChiSq med {out.veres_rchi.median():.2f}", flush=True)
    print("VERES MEASURE DONE", flush=True)


if __name__ == "__main__":
    main()
