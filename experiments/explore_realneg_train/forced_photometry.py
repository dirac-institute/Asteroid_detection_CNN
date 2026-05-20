#!/usr/bin/env python
"""LSST PSF forced photometry on test_real diffims at ephemeris-projected (x,y).

For every sighting in DATA_DIFFIM/test_real/per_sighting.csv:
  - Re-fetch the PVI + template_coadd via Butler, re-run AL subtraction (gives
    a diffim Exposure with PSF + variance plane that the test.h5 cached image
    does not retain).
  - Look up the ephemeris-projected trail midpoint (x, y) from test.csv.
  - Run optimal PSF photometry at (x, y) — same math as `base_PsfFlux` —
    using the actual LSST PSF model and variance plane from run_subtract().
    flux = sum(phi * diffim) / sum(phi^2)
    fluxErr = sqrt(sum(phi^2 * variance)) / sum(phi^2)
    forced_psf_snr = flux / fluxErr.
  - Also compute a trail-aperture flux/SNR along the line (PCA of (x,y) +
    trail length) for trails >> PSF FWHM.

Writes <out>/per_sighting_forced.csv augmented with x, y, forced_psf_flux,
forced_psf_fluxErr, forced_psf_snr, forced_trail_flux, forced_trail_fluxErr,
forced_trail_snr.
"""
from __future__ import annotations
import argparse, sys, time, concurrent.futures, threading
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
_DSC = f"{_REPO}/ADCNN/data/dataset_creation"
for p in (_REPO, _DSC):
    if p not in sys.path: sys.path.insert(0, p)

from ADCNN.data.dataset_creation.pipetasks import fetch_diffim_inputs, run_subtract  # noqa: E402
from ADCNN.data.dataset_creation.simulate_inject_diffim import format_dataId  # noqa: E402
from lsst.daf.butler import Butler  # noqa: E402
import lsst.geom as geom  # noqa: E402

STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP = "lsst_cells_v2"
REPO = "dp2_prep"

_TLS = threading.local()
def _butler():
    b = getattr(_TLS, "butler", None)
    if b is None:
        b = Butler(REPO, collections=[STAGE3, STAGE2])
        _TLS.butler = b
    return b


def _psf_forced(exposure, x, y):
    """Optimal PSF photometry at (x,y) on the diffim Exposure.
    Returns (flux, fluxErr, snr) or (nan, nan, nan)."""
    try:
        loc = geom.Point2D(float(x), float(y))
        psf = exposure.psf
        psf_img = psf.computeKernelImage(loc)
        phi = psf_img.array.astype(np.float64)
        phi_bbox = psf_img.getBBox()
        img_bbox = exposure.getBBox()
        clipped = phi_bbox.clippedTo(img_bbox)
        if clipped.isEmpty() or clipped.getWidth() < 3 or clipped.getHeight() < 3:
            return float("nan"), float("nan"), float("nan")
        sub = exposure.image[clipped].array.astype(np.float64)
        var = exposure.variance[clipped].array.astype(np.float64)
        dx0 = clipped.getMinX() - phi_bbox.getMinX()
        dy0 = clipped.getMinY() - phi_bbox.getMinY()
        dx1 = dx0 + clipped.getWidth()
        dy1 = dy0 + clipped.getHeight()
        phi_c = phi[dy0:dy1, dx0:dx1]
        phi2 = phi_c * phi_c
        alpha = float(phi2.sum())
        if not np.isfinite(alpha) or alpha <= 0:
            return float("nan"), float("nan"), float("nan")
        # Mask out non-finite variance pixels.
        ok = np.isfinite(sub) & np.isfinite(var) & (var > 0)
        if ok.sum() < 5:
            return float("nan"), float("nan"), float("nan")
        flux = float((phi_c[ok] * sub[ok]).sum() / alpha)
        flux_err = float(np.sqrt((phi2[ok] * var[ok]).sum()) / alpha)
        snr = flux / flux_err if flux_err > 0 else float("nan")
        return flux, flux_err, snr
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _trail_forced(exposure, x, y, trail_length, beta_deg, width=2.0):
    """Aperture along the line of length trail_length, width 'width' px.
    flux = sum(image) in line; var = sum(variance) in line.
    For short trails (< 2 * PSF_FWHM), falls back to PSF photometry."""
    try:
        L = float(trail_length)
        if not np.isfinite(L) or L < 3.0:
            return float("nan"), float("nan"), float("nan")
        theta = np.deg2rad(float(beta_deg))
        c, s = np.cos(theta), np.sin(theta)
        n = max(int(L), 3)
        ts = np.linspace(-L / 2.0, L / 2.0, n)
        # Sample line center + width perpendicular offsets
        ws = np.arange(-int(width // 2), int(width // 2) + 1, 1)
        xs = (x + ts * c).astype(int)
        ys = (y + ts * s).astype(int)
        bbox = exposure.getBBox()
        img = exposure.image.array
        var = exposure.variance.array
        H, W = img.shape
        flux = 0.0; varsum = 0.0; n_px = 0
        for wo in ws:
            xx = xs + int(round(-wo * s))
            yy = ys + int(round(wo * c))
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            v = var[yy[m], xx[m]]
            i = img[yy[m], xx[m]]
            ok = np.isfinite(v) & (v > 0) & np.isfinite(i)
            flux += float(i[ok].sum())
            varsum += float(v[ok].sum())
            n_px += int(ok.sum())
        if n_px < 5 or varsum <= 0:
            return float("nan"), float("nan"), float("nan")
        flux_err = float(np.sqrt(varsum))
        snr = flux / flux_err if flux_err > 0 else float("nan")
        return flux, flux_err, snr
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _process_panel(args):
    visit, detector, sightings = args
    out_rows = []
    try:
        b = _butler()
        ref = b.registry.findDataset(
            "preliminary_visit_image",
            dataId={"instrument": "LSSTCam", "visit": int(visit),
                    "detector": int(detector)},
            collections=[STAGE3, STAGE2],
        )
        pvi, sources, template, _filter, _ntpl = fetch_diffim_inputs(
            b, format_dataId(ref.dataId), skymap=SKYMAP,
            stage3_collection=STAGE3,
        )
        sub = run_subtract(template=template, science=pvi, sources=sources)
        diffim = sub.difference
        for row in sightings:
            pf, pe, ps = _psf_forced(diffim, row["x"], row["y"])
            tf, te, ts = _trail_forced(
                diffim, row["x"], row["y"],
                row.get("trail_length", float("nan")),
                row.get("beta", float("nan")),
            )
            out_rows.append({
                **row,
                "forced_psf_flux": pf, "forced_psf_fluxErr": pe,
                "forced_psf_snr": ps,
                "forced_trail_flux": tf, "forced_trail_fluxErr": te,
                "forced_trail_snr": ts,
            })
        return ("ok", visit, detector, out_rows, len(sightings))
    except Exception as e:
        return ("err", visit, detector, [
            {**row, "forced_psf_flux": float("nan"),
             "forced_psf_fluxErr": float("nan"),
             "forced_psf_snr": float("nan"),
             "forced_trail_flux": float("nan"),
             "forced_trail_fluxErr": float("nan"),
             "forced_trail_snr": float("nan")}
            for row in sightings], repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-sighting", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--parallel", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0,
                    help="for smoke testing")
    a = ap.parse_args()

    ps = pd.read_csv(a.per_sighting)
    tc = pd.read_csv(a.test_csv)
    print(f"[load] per_sighting n={len(ps)} test n={len(tc)}", flush=True)
    # Join: per_sighting has (ObjID, visit, detector, image_id); test.csv has
    # (ObjID, ra, dec, x, y, beta, trail_length, visit, detector). We want the
    # ephemeris (x, y, beta, trail_length) per sighting.
    keep_cols = ["ObjID", "visit", "detector", "x", "y", "beta",
                 "trail_length", "ra", "dec"]
    tcj = tc[[c for c in keep_cols if c in tc.columns]].copy()
    if "trail_length" in ps.columns and "trail_length" in tcj.columns:
        tcj = tcj.drop(columns=["trail_length"])
    m = ps.merge(tcj, on=["ObjID", "visit", "detector"], how="left",
                 suffixes=("", "_eph"))
    nans = m["x"].isna().sum()
    print(f"[join] sightings={len(m)}  no_xy={nans}", flush=True)
    if a.limit:
        m = m.head(a.limit).copy()
        print(f"[limit] {len(m)} sightings", flush=True)

    # Group by (visit, detector) -> list of sighting rows
    panels = []
    for (v, d), g in m.groupby(["visit", "detector"], sort=True):
        rows = g.to_dict("records")
        if any(pd.notna(r.get("x")) for r in rows):
            panels.append((int(v), int(d), rows))
    print(f"[panels] {len(panels)} unique (visit,detector) panels", flush=True)

    t0 = time.time()
    out = []
    ok = err = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(_process_panel, p) for p in panels]
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res[0] == "ok": ok += 1
            else: err += 1
            out.extend(res[3])
            if n % 20 == 0 or n == len(panels):
                print(f"[{n}/{len(panels)}] ok={ok} err={err} "
                      f"{time.time()-t0:.0f}s rows={len(out)}", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(a.out_csv, index=False)
    print(f"[done] wrote {a.out_csv} n={len(df)} "
          f"forced_psf_snr median={df['forced_psf_snr'].median():.2f} "
          f"valid={int(df['forced_psf_snr'].notna().sum())}", flush=True)


if __name__ == "__main__":
    main()
