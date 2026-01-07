from __future__ import annotations
import math
import numpy as np
from pathlib import Path
from typing import Tuple
import cv2

from lsst.daf.butler import Butler
import lsst.geom as geom

# ---------- small utils ----------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------- sky-motion geometry ----------
def vsky_and_pa(ra_rate_cosdec_deg_day: float, dec_rate_deg_day: float) -> Tuple[float, float]:
    """
    Inputs are Sorcha-like components (east=x, north=y) in deg/day.
    Returns:
      vsky_deg_day, position angle in degrees East of North.
    """
    x = float(ra_rate_cosdec_deg_day)
    y = float(dec_rate_deg_day)
    vsky = math.hypot(x, y)
    pa = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return vsky, pa

def detectors_covering_point(butler: Butler, visit: int, ra_deg: float, dec_deg: float):
    where = (
        f"instrument='LSSTCam' AND visit={int(visit)} "
        f"AND visit_detector_region.region OVERLAPS POINT({ra_deg:.9f}, {dec_deg:.9f})"
    )
    return list(butler.registry.queryDatasets("calexp", where=where, findFirst=True))

def sky_to_pixel(calexp, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    sp = geom.SpherePoint(geom.Angle(ra_deg, geom.degrees), geom.Angle(dec_deg, geom.degrees))
    x, y = calexp.wcs.skyToPixel(sp)
    return float(x), float(y)

def draw_one_line(mask, origin, angle, length, true_value=1, line_thickness=500):
    x0, y0 = origin
    x_size = length * np.cos((np.pi / 180) * angle)
    y_size = length * np.sin((np.pi / 180) * angle)
    x1 = x0 - x_size / 2
    y1 = y0 - y_size / 2
    x0 = x0 + x_size / 2
    y0 = y0 + y_size / 2
    one_line_mask = cv2.line(np.zeros(mask.shape), (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=line_thickness)
    mask[one_line_mask != 0] = true_value
    return mask

def psf_fit_flux_sigma(calexp, x, y):
    psf = calexp.getPsf()
    var_full = calexp.variance.array.astype(np.float64)

    p = psf.computeImage(geom.Point2D(x, y)).array.astype(np.float64)
    ph, pw = p.shape

    # center of stamp in full-image coords
    cx = int(round(x))
    cy = int(round(y))

    # stamp bounds in full image
    x0 = cx - pw//2
    y0 = cy - ph//2
    x1 = x0 + pw
    y1 = y0 + ph

    H, W = var_full.shape
    # clip to image bounds
    ix0 = max(x0, 0); iy0 = max(y0, 0)
    ix1 = min(x1, W); iy1 = min(y1, H)

    # corresponding bounds in PSF stamp coords
    px0 = ix0 - x0; py0 = iy0 - y0
    px1 = px0 + (ix1 - ix0); py1 = py0 + (iy1 - iy0)

    p_cut = p[py0:py1, px0:px1]
    v_cut = var_full[iy0:iy1, ix0:ix1]

    # normalize PSF cut to sum=1 (important if clipping happened)
    s = p_cut.sum()
    if not np.isfinite(s) or s <= 0:
        p = None
        v = None
    else:
        p = p_cut / s
        v = v_cut
    if p is None:
        return np.nan
    good = np.isfinite(v) & (v > 0) & np.isfinite(p)
    denom = np.sum((p[good]**2) / v[good])
    if denom <= 0:
        return np.nan
    return float(np.sqrt(1.0 / denom))

def mag_to_snr (mag, calexp, x, y):
    F = calexp.getPhotoCalib().magnitudeToInstFlux(mag)
    sigmaF = psf_fit_flux_sigma(calexp, x, y)
    snr = F / sigmaF
    return snr

def snr_to_mag(snr, calexp, x, y):
    sigmaF = psf_fit_flux_sigma(calexp, x, y)
    F = snr * sigmaF
    mag = calexp.getPhotoCalib().instFluxToMagnitude(F)
    return mag