#!/usr/bin/env python3
"""MATCHED-FILTER trail-length estimator, measured against injected truth.

WHY. The shipped length comes from the SEGMENTATION FOOTPRINT EXTENT, which is contrast-dependent:
on a faint trail the seg net claims only the brightest section and the ends fall below threshold.
Measured on truth, 47.6% of faint (SNR 2-4) 40-56px trails are recovered at <60% of true length --
20.7px for a true 40.1px trail. A faint FAST mover is therefore MEASURED AS A SLOW ONE, the
trail-implied rate disagrees with the two-epoch chord, the dspeed chi2 term fires, and the pair is
rejected for a consistency failure caused by its own mismeasurement.

ESTIMATOR. For a trail of length L at angle beta, the PSF-convolved line template T(L,beta) gives
    S(L,beta) = sum(I * T) / (sigma * ||T||_2)
S peaks at the true length: too short and it misses flux; too long and ||T|| grows as sqrt(L) while
the collected signal stops growing, so S falls. L_hat = argmax S. The PEAK LOCATION does not depend
on flux -- only the precision of locating it does -- so unlike a footprint extent this is
SNR-independent to first order. That is the whole claim, and this script tests it.

The trails are re-injected from the truth table itself (exact sky position, length, PA, magnitude per
object), so no RNG replay is needed and the pixels are identical to what the detector saw.

Usage:  python mf_length.py [n_panels]
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.special import erf
from astropy.wcs import WCS

sys.path.insert(0, os.environ.get("ADCNN_REPO", "."))
from ADCNN.inference.diffim_io import open_diffim
from ADCNN.pipelines.heliolinc.inject_trails import add_trails, PSF_SIGMA_PX

SOLARDAY, EXPTIME, PIX = 86400.0, 30.0, 0.2
STAMP = 96                                   # half-width 48 >= max half-length (56/2) + PSF wings
# 1px steps. A 3px grid put injected lengths 7/10/28/40 EXACTLY on grid points while 14/20/56 fell
# between them -- 21% quantisation at 14px -- contaminating the very per-length comparison this
# script exists to make. Panel IO dominates runtime, so a fine grid is nearly free.
L_GRID = np.arange(4, 80, 1.0)               # candidate trail lengths (px)
B_GRID = np.arange(0, 180, 2.0)              # candidate image-frame angles (deg); beta is NOT known


def sky_trail_to_pixel(w, ra, dec, L_deg, pa_deg):
    cd = np.cos(np.radians(dec))
    dra = 0.5 * L_deg * np.cos(np.radians(pa_deg)) / max(cd, 1e-6)
    ddec = 0.5 * L_deg * np.sin(np.radians(pa_deg))
    (x0, y0), (x1, y1) = w.all_world2pix([[ra - dra, dec - ddec], [ra + dra, dec + ddec]], 0)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), float(np.hypot(x1 - x0, y1 - y0)), \
        float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def build_templates(stamp=STAMP, sigma_px=PSF_SIGMA_PX):
    """(n_template, stamp*stamp) unit-L2 PSF-convolved line templates, built ONCE and reused."""
    c = stamp // 2
    yy, xx = np.mgrid[0:stamp, 0:stamp].astype(np.float32)
    yy -= c; xx -= c
    T, meta = [], []
    for L in L_GRID:
        for b in B_GRID:
            ca, sa = np.cos(np.radians(b)), np.sin(np.radians(b))
            s = xx * ca + yy * sa            # along-track
            p = -xx * sa + yy * ca           # cross-track
            # line segment of half-length L/2, convolved with a Gaussian PSF across the track and
            # softened at the ends (the same profile family the injector stamps)
            # ALONG-TRACK the profile is a TOP-HAT CONVOLVED WITH THE PSF, not a hard-edged
            # segment: a uniformly moving point source smears its ends over ~+-2*sigma. A sharp-ended
            # template correlated against smeared ends peaks SHORT -- measured as a systematic -10.8%
            # underestimate on fast movers (at 56px, 11% = 6px ~ 2 sigma, which is the tell). The
            # top-hat convolved with a Gaussian is exactly the difference of two error functions.
            along = 0.5 * (erf((0.5 * L - s) / (np.sqrt(2) * sigma_px))
                           + erf((0.5 * L + s) / (np.sqrt(2) * sigma_px)))
            t = along * np.exp(-0.5 * (p / sigma_px) ** 2)
            n = np.linalg.norm(t)
            if n <= 0:
                continue
            T.append((t / n).ravel()); meta.append((L, b))
    return np.asarray(T, np.float32), np.asarray(meta, np.float32)


def main(n_panels=40):
    V = "outputs/runs/pa_validate"
    T = pd.read_csv(f"{V}/truth_snr.csv")
    T["detA_ok"] = T.detA_ok.fillna(False)
    man = pd.read_csv("outputs/runs/10k_cadence/run_night_20260706/manifest.csv")
    SIG = float(os.environ.get("MF_SIGMA", PSF_SIGMA_PX))
    TPL, META = build_templates(sigma_px=SIG)
    print(f"[mf] {len(TPL):,} templates ({len(L_GRID)} lengths x {len(B_GRID)} angles), stamp {STAMP}px, tmpl sigma {SIG} (injector {PSF_SIGMA_PX})",
          flush=True)
    rows, done = [], 0
    for (v, d), g in T[T.detA_ok].groupby(["visitA", "detA"]):
        r = man[(man.visit == v) & (man.detector == d)]
        if not len(r):
            continue
        try:
            with open_diffim(r.fits_path.iloc[0], memmap=False) as h:
                img = np.nan_to_num(h[1].data.astype(np.float32)); w = WCS(h[1].header)
        except Exception:
            continue
        H, W = img.shape
        inj, keep = [], []
        for _, p in g.iterrows():
            L_deg = p.rate * (EXPTIME / SOLARDAY)
            x, y, Lpx, beta = sky_trail_to_pixel(w, p.raA, p.decA, L_deg, p.pa)
            if not (STAMP < x < W - STAMP and STAMP < y < H - STAMP):
                continue
            inj.append(dict(x=x, y=y, trail_length=Lpx, beta=beta, mag=p.mag))
            keep.append((p.oid, p.L_px, p.snr_t, p.detA_len, x, y, Lpx))
        if not inj:
            continue
        im = add_trails(np.array(img, copy=True), inj)
        sigma = 1.4826 * np.median(np.abs(im - np.median(im)))
        c = STAMP // 2
        for (oid, Ltrue, snr, seg_len, x, y, Lpx) in keep:
            xi, yi = int(round(x)), int(round(y))
            cut = im[yi - c:yi - c + STAMP, xi - c:xi - c + STAMP]
            if cut.shape != (STAMP, STAMP):
                continue
            S = TPL @ cut.ravel() / max(sigma, 1e-6)
            k = int(np.argmax(S))
            rows.append(dict(oid=oid, L_true=Ltrue, snr_t=snr, seg_len=seg_len,
                             mf_len=float(META[k, 0]), mf_snr=float(S[k])))
        done += 1
        if done % 10 == 0:
            print(f"[mf] {done} panels, {len(rows):,} trails", flush=True)
        if done >= n_panels:
            break
    R = pd.DataFrame(rows)
    R.to_csv(os.environ.get("MF_OUT", f"{V}/mf_length.csv"), index=False)
    print(f"[mf] wrote {V}/mf_length.csv  n={len(R):,}\n")
    R = R[np.isfinite(R.seg_len)]
    print("TRAIL LENGTH: segmentation footprint (shipped) vs MATCHED FILTER, against truth\n")
    print(f"{'true L':>8}{'n':>6}{'seg bias':>11}{'MF bias':>10}{'seg |err|p90':>14}{'MF |err|p90':>13}"
          f"{'seg trunc<60%':>15}{'MF trunc':>10}")
    for L in sorted(R.L_true.round(0).unique()):
        g = R[R.L_true.round(0) == L]
        if len(g) < 15:
            continue
        se = g.seg_len / g.L_true - 1; me = g.mf_len / g.L_true - 1
        print(f"{L:>8.0f}{len(g):>6}{100*se.median():>10.1f}%{100*me.median():>9.1f}%"
              f"{100*se.abs().quantile(.9):>13.1f}%{100*me.abs().quantile(.9):>12.1f}%"
              f"{100*(g.seg_len/g.L_true<0.6).mean():>14.1f}%{100*(g.mf_len/g.L_true<0.6).mean():>9.1f}%")
    print(f"\nBy SNR -- the claim is that the MF peak location is SNR-INDEPENDENT:")
    print(f"{'SNR':>8}{'n':>6}{'seg bias':>11}{'MF bias':>10}{'seg trunc':>12}{'MF trunc':>10}")
    for lo, hi in [(2, 4), (4, 6), (6, 8), (8, 10)]:
        g = R[(R.snr_t >= lo) & (R.snr_t < hi)]
        if len(g) < 15:
            continue
        print(f"{f'{lo}-{hi}':>8}{len(g):>6}{100*(g.seg_len/g.L_true-1).median():>10.1f}%"
              f"{100*(g.mf_len/g.L_true-1).median():>9.1f}%"
              f"{100*(g.seg_len/g.L_true<0.6).mean():>11.1f}%{100*(g.mf_len/g.L_true<0.6).mean():>9.1f}%")
    print(f"\nIMPLIED RATE ERROR (what the linker's dspeed chi2 term actually sees):")
    for lab, col in (("segmentation (shipped)", "seg_len"), ("matched filter", "mf_len")):
        e = (R[col] / R.L_true - 1).abs()
        print(f"  {lab:<24} median {100*e.median():>5.1f}%   p90 {100*e.quantile(.9):>6.1f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
