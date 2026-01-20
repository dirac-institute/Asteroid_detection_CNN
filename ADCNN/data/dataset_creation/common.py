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

import numpy as np
import lsst.geom as geom

def _get_psf_stamp_and_var(calexp, x, y, use_kernel_image=False):
    """
    Returns (p_cut, v_cut) where:
      - p_cut is the PSF template stamp (float64)
      - v_cut is the variance stamp aligned to p_cut (float64)
    The stamp is clipped to image bounds.
    """
    psf = calexp.getPsf()
    var_full = calexp.variance.array.astype(np.float64)

    # PSF template
    if use_kernel_image and hasattr(psf, "computeKernelImage"):
        p = psf.computeKernelImage(geom.Point2D(x, y)).array.astype(np.float64)
    else:
        p = psf.computeImage(geom.Point2D(x, y)).array.astype(np.float64)

    ph, pw = p.shape
    cx = int(round(x))
    cy = int(round(y))

    x0 = cx - pw // 2
    y0 = cy - ph // 2
    x1 = x0 + pw
    y1 = y0 + ph

    H, W = var_full.shape
    ix0 = max(x0, 0); iy0 = max(y0, 0)
    ix1 = min(x1, W); iy1 = min(y1, H)

    px0 = ix0 - x0; py0 = iy0 - y0
    px1 = px0 + (ix1 - ix0); py1 = py0 + (iy1 - iy0)

    p_cut = p[py0:py1, px0:px1]
    v_cut = var_full[iy0:iy1, ix0:ix1]
    return p_cut, v_cut


def sigma_psf_wls(calexp, x, y, *, use_kernel_image=False):
    """
    1-sigma uncertainty of PSF amplitude using inverse-variance weighted LS:
        Var(F) = 1 / sum_i (phi_i^2 / V_i)
    where phi is PSF template normalized to sum(phi)=1.
    """
    p_cut, v_cut = _get_psf_stamp_and_var(calexp, x, y, use_kernel_image=use_kernel_image)

    s = p_cut.sum()
    if not np.isfinite(s) or s <= 0:
        return np.nan
    phi = p_cut / s

    good = np.isfinite(v_cut) & (v_cut > 0) & np.isfinite(phi)
    denom = np.sum((phi[good] ** 2) / v_cut[good])
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.sqrt(1.0 / denom))


def sigma_psf_constvar(calexp, x, y, *, use_kernel_image=False):
    """
    1-sigma uncertainty of PSF amplitude under a constant-variance (unweighted LS) approximation.
    This matches the Bosch-style expression:
        alpha = sum_i phi_i^2
        Var(F) = sum_i (phi_i^2 * V_i) / alpha^2
    where phi is PSF template normalized to sum(phi)=1.

    NOTE: Equivalent to sigma_psf_wls only if V_i is constant across the PSF stamp.
    """
    p_cut, v_cut = _get_psf_stamp_and_var(calexp, x, y, use_kernel_image=use_kernel_image)

    s = p_cut.sum()
    if not np.isfinite(s) or s <= 0:
        return np.nan
    phi = p_cut / s

    good = np.isfinite(v_cut) & (v_cut > 0) & np.isfinite(phi)
    if not np.any(good):
        return np.nan

    alpha = np.sum(phi[good] ** 2)
    if not np.isfinite(alpha) or alpha <= 0:
        return np.nan

    varF = np.sum((phi[good] ** 2) * v_cut[good]) / (alpha ** 2)
    if not np.isfinite(varF) or varF <= 0:
        return np.nan
    return float(np.sqrt(varF))


# ---- Switchable wrapper (drop-in replacement for your old psf_fit_flux_sigma) ----
def psf_fit_flux_sigma(calexp, x, y, *, estimator="wls", use_kernel_image=False):
    """
    estimator:
      - "wls"      : inverse-variance weighted LS (recommended statistically)
      - "constvar" : constant-variance approximation (Bosch-style)
    use_kernel_image:
      - False: psf.computeImage(...)
      - True : psf.computeKernelImage(...) if available, else falls back to computeImage
    """
    if estimator == "wls":
        return sigma_psf_wls(calexp, x, y, use_kernel_image=use_kernel_image)
    elif estimator == "constvar":
        return sigma_psf_constvar(calexp, x, y, use_kernel_image=use_kernel_image)
    else:
        raise ValueError(f"Unknown estimator={estimator!r}. Use 'wls' or 'constvar'.")

def mag_to_snr(mag, calexp, x, y, *, estimator="wls", use_kernel_image=False):
    F = calexp.getPhotoCalib().magnitudeToInstFlux(mag)
    sigmaF = psf_fit_flux_sigma(calexp, x, y, estimator=estimator, use_kernel_image=use_kernel_image)
    return F / sigmaF

def snr_to_mag(snr, calexp, x, y, *, estimator="wls", use_kernel_image=False):
    sigmaF = psf_fit_flux_sigma(calexp, x, y, estimator=estimator, use_kernel_image=use_kernel_image)
    F = snr * sigmaF
    return calexp.getPhotoCalib().instFluxToMagnitude(F)
