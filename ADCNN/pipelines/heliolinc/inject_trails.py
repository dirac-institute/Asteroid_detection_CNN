"""Add synthetic asteroid trails directly into a (real) difference-image array (nJy units).

Pilot injection path: instead of the full LSST inject+re-subtract, we add a PSF-convolved trail of the
target integrated magnitude straight into the REAL off-ecliptic diffim (which already carries the real
FP). Trail morphology (light-curve modulation, tapered ends, slight curvature) reuses the validated
profile from realistic_trail.py; the PSF is approximated by a Gaussian of the field seeing. Flux:
diffim is nJy with AB zeropoint 31.4 (mag = 31.4 - 2.5*log10(flux_nJy)).
"""
from __future__ import annotations
import numpy as np

ZP = 31.4          # AB zeropoint for nJy diffims (mag = ZP - 2.5 log10 flux_nJy)
PSF_SIGMA_PX = 1.6  # ~0.75" seeing / 0.2"px / 2.355 ; Gaussian PSF approx for the pilot


def _profile(L, rng):
    """Reuse realistic_trail's along-track positions / curvature / flux weights."""
    try:
        from ADCNN.data.dataset_creation.realistic_trail import _trail_profile
    except Exception:
        # minimal fallback: uniform trail
        n = max(int(np.ceil(L * 2)) + 1, 11)
        s = np.linspace(-0.5 * L, 0.5 * L, n)
        return s, np.zeros_like(s), np.ones_like(s) / len(s)
    return _trail_profile(L, rng)


def add_trails(img, rows, sigma_px=PSF_SIGMA_PX, zp=ZP, seed=0):
    """Add each row's trail into `img` (modified in place and returned). `rows` is an iterable of objects
    with attributes/keys x, y, beta (deg, image frame), trail_length (px), mag. Off-image trails are clipped."""
    if rows is None:
        return img
    H, W = img.shape
    rng = np.random.default_rng(seed)
    hw = int(np.ceil(4 * sigma_px))                     # stamp half-width
    inv2s2 = 1.0 / (2 * sigma_px * sigma_px)
    for r in rows:
        x = float(r["x"]); y = float(r["y"]); beta = np.radians(float(r["beta"]))
        L = max(float(r["trail_length"]), 1.0); mag = float(r["mag"])
        F = 10.0 ** ((zp - mag) / 2.5)                  # total flux, nJy
        s, perp, w = _profile(L, rng)
        ca, sa = np.cos(beta), np.sin(beta)
        # sample-point centres (curvature perp is lateral to the trail)
        xs = x + s * ca - perp * sa
        ys = y + s * sa + perp * ca
        for xc, yc, wi in zip(xs, ys, w):
            ix, iy = int(round(xc)), int(round(yc))
            x0, x1 = max(ix - hw, 0), min(ix + hw + 1, W)
            y0, y1 = max(iy - hw, 0), min(iy + hw + 1, H)
            if x1 <= x0 or y1 <= y0:
                continue
            gx = np.arange(x0, x1) - xc
            gy = np.arange(y0, y1) - yc
            stamp = np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) * inv2s2)
            ssum = stamp.sum()
            if ssum > 0:
                img[y0:y1, x0:x1] += np.float32(wi * F / ssum) * stamp.astype(np.float32)
    return img


def load_inject_map(csv_path):
    """Return {(visit,detector): list-of-row-dicts} from an inject.csv (objID,visit,detector,x,y,trail_length,beta,mag)."""
    import pandas as pd
    d = pd.read_csv(csv_path)
    m = {}
    for (v, det), g in d.groupby(["visit", "detector"]):
        m[(int(v), int(det))] = g.to_dict("records")
    return m
