#!/usr/bin/env python3
"""Annotate diffim manifests with per-panel sky WCS as FITS cards (LSST-stack env).

Newer DRP outputs (e.g. DM-53195 d_2025_11_10) do NOT carry an astropy-readable sky WCS in the FITS
headers (the 'A' key is a pixel bookkeeping transform; the exact SkyWcs lives in archive HDUs), and
most of their SkyWcs objects have NO attached FITS approximation (getFitsMetadata raises). This script
Butler-component-reads `difference_image.wcs` per (visit, detector) -- cheap, no pixel I/O -- and
produces FITS-WCS cards per panel, stored as a JSON string column `wcs_json` in the manifest:

  1. the pipeline-attached FITS approximation (getFitsMetadata) when present, else
  2. a TAN-SIP fit to the exact SkyWcs sampled on a pixel grid (astropy fit_wcs_from_points).

Either way the cards are VALIDATED end-to-end: an astropy WCS is built from the JSON (exactly what the
consumers sim_orbits/discover_stream/ephem_to_inject do via _wcs_from_json) and compared to the exact
SkyWcs on a held-out grid; max residual must be < --tol arcsec (recorded per manifest). Injection,
detection and linking all use the same approximation -> self-consistent.

Parallelism is per-ROW (panel) with BLAS capped to 1 thread/worker: the SIP least-squares otherwise
grabs every core per fit and 8 concurrent fits thrash (~50 CPU-s/panel observed).

Usage (stack env):  python annotate_manifest_wcs.py --run run_blind --collection <DM-53195 chain>
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json, glob, sys
from multiprocessing import Pool
from pathlib import Path

# run as a PATH (`python ADCNN/pipelines/heliolinc/annotate_manifest_wcs.py`) sys.path[0] is this
# directory, so neither this process nor its Pool workers can `import ADCNN`. Repo idiom.
_REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

GRID_FIT = 24    # fit grid (per axis)
GRID_VAL = 17    # held-out validation grid (per axis, offset from fit grid)

_B = None        # per-worker Butler
_COLL = None
_TOL = None


def _cards_from_skywcs(w, nx, ny, tol_arcsec):
    """FITS-WCS cards (dict) for SkyWcs `w`, validated against it on a held-out grid.
    Returns (cards, max_residual_arcsec). Raises if no representation meets tol."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import fit_wcs_from_points

    def _validate(cards):
        h = fits.Header()
        for k, v in cards.items():
            if k in ("COMMENT", "HISTORY") or v is None:
                continue
            h[k] = v
        aw = WCS(h)
        if not aw.has_celestial:
            raise RuntimeError("reconstructed WCS not celestial")
        xv, yv = np.meshgrid(np.linspace(7.0, nx - 8.0, GRID_VAL), np.linspace(11.0, ny - 12.0, GRID_VAL))
        xv = xv.ravel(); yv = yv.ravel()
        ra_e, dec_e = w.pixelToSkyArray(xv, yv, degrees=True)
        sky = aw.all_pix2world(np.column_stack([xv, yv]), 0)
        sep = SkyCoord(ra_e * u.deg, dec_e * u.deg).separation(
            SkyCoord(sky[:, 0] * u.deg, sky[:, 1] * u.deg)).arcsec
        return float(np.max(sep))

    # 1) pipeline-attached FITS approximation, if any
    try:
        cards = dict(w.getFitsMetadata().toDict())
        res = _validate(cards)
        if res < tol_arcsec:
            return cards, res
    except Exception:
        pass
    # 2) TAN-SIP fit to the exact transform
    xs, ys = np.meshgrid(np.linspace(0.0, nx - 1.0, GRID_FIT), np.linspace(0.0, ny - 1.0, GRID_FIT))
    xs = xs.ravel(); ys = ys.ravel()
    ra, dec = w.pixelToSkyArray(xs, ys, degrees=True)
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    last = None
    for deg in (3, 4):
        fitted = fit_wcs_from_points((xs, ys), sc, projection="TAN", sip_degree=deg)
        hdr = fitted.to_header(relax=True)
        cards = {k: hdr[k] for k in hdr if k not in ("COMMENT", "HISTORY")}
        last = _validate(cards)
        if last < tol_arcsec:
            return cards, last
    raise RuntimeError(f"TAN-SIP fit residual {last:.3f} arcsec > tol {tol_arcsec}")


def _init(collection, tol, butler_repo):
    global _B, _COLL, _TOL
    from lsst.daf.butler import Butler
    _B = Butler(butler_repo); _COLL = collection; _TOL = tol


def _annotate_row(task):
    """Worker: one panel -> (idx, wcs_json|None, residual, err|None)."""
    i, visit, detector, fits_path = task
    try:
        w = _B.get("difference_image.wcs", instrument="LSSTCam", visit=visit,
                   detector=detector, collections=_COLL)
        # open_diffim (not astropy.io.fits) so S3 datastores work: an embargo path is
        # s3://embargo@rubin-summit-users/..., whose profile@bucket form astropy rejects
        # outright ("Invalid bucket name"), which failed EVERY panel of a prompt-processing night.
        from ADCNN.inference.diffim_io import open_diffim
        with open_diffim(fits_path, memmap=True) as h:
            hdr1 = h[1].header
            cards, res = _cards_from_skywcs(w, int(hdr1["NAXIS1"]), int(hdr1["NAXIS2"]), _TOL)
        return i, json.dumps(cards), res, None
    except Exception as e:
        return i, None, 0.0, f"{type(e).__name__}: {str(e)[:140]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with manifest_*.csv")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--butler-repo", default=os.environ.get("BUTLER_REPO", "main"),
                    help="Butler repo holding the diffim SkyWcs (default $BUTLER_REPO or main)")
    ap.add_argument("--tol", type=float, default=0.1, help="max validation residual (arcsec)")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    with Pool(a.workers, initializer=_init, initargs=(a.collection, a.tol, a.butler_repo)) as pool:
        for mf in sorted(glob.glob(f"{a.run}/manifest_*.csv")):
            m = pd.read_csv(mf)
            if "wcs_json" in m.columns and m.wcs_json.notna().all():
                print(f"[wcs] {os.path.basename(mf)} already annotated", flush=True)
                continue
            tasks = [(i, int(r.visit), int(r.detector), r.fits_path)
                     for i, r in enumerate(m.itertuples(index=False))]
            out = [None] * len(m)
            nbad, worst, first_err = 0, 0.0, None
            for i, js, res, err in pool.imap_unordered(_annotate_row, tasks, chunksize=4):
                if js is None:
                    nbad += 1
                    if first_err is None:
                        first_err = err
                else:
                    out[i] = js; worst = max(worst, res)
            m["wcs_json"] = out
            m.to_csv(mf, index=False)
            msg = f"[wcs] {os.path.basename(mf)}: {len(m)-nbad}/{len(m)} annotated ({nbad} failed), worst residual {worst*1000:.1f} mas"
            if first_err:
                msg += f" | first failure: {first_err}"
            print(msg, flush=True)
    print("WCS_ANNOTATE_DONE", flush=True)


if __name__ == "__main__":
    main()
