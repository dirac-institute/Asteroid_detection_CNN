from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2

import lsst.geom as geom


# ======================================================================================
# Minimal helpers used by ADCNN/data/dataset_creation/simulate_inject.py
# ======================================================================================

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def draw_one_line(
    mask: np.ndarray,
    origin: Tuple[float, float],
    angle_deg: float,
    length: float,
    true_value: int = 1,
    line_thickness: int = 500,
) -> np.ndarray:
    """
    Draw a thick line into `mask` (2D array) and set those pixels to `true_value`.
    OpenCV uses (x, y) integer pixel coordinates.
    """
    x0, y0 = origin
    x_size = float(length) * math.cos(math.radians(angle_deg))
    y_size = float(length) * math.sin(math.radians(angle_deg))

    x1 = x0 - x_size / 2.0
    y1 = y0 - y_size / 2.0
    x2 = x0 + x_size / 2.0
    y2 = y0 + y_size / 2.0

    line = cv2.line(
        np.zeros(mask.shape, dtype=np.uint8),
        (int(round(x2)), int(round(y2))),
        (int(round(x1)), int(round(y1))),
        1,
        thickness=int(line_thickness),
    )
    mask[line != 0] = true_value
    return mask

# ======================================================================================
# SNR calculation
# ======================================================================================

def mag_to_snr(mag, calexp, x, y, *, use_kernel_image=False, l_pix=None, theta_deg=None):
    F = calexp.getPhotoCalib().magnitudeToInstFlux(mag)
    sigmaF = psf_fit_flux_sigma(
        stack_snr=snr, calexp=calexp, x=x, y=y,
        use_kernel_image=use_kernel_image,
        L_pix=l_pix,
        theta_deg=theta_deg,
        n_samples=n_samples
    )
    return F / sigmaF

def snr_to_mag(snr, calexp, x, y, *, use_kernel_image=False, l_pix=None, theta_deg=None, n_samples=20):
    sigmaF = psf_fit_flux_sigma(
        stack_snr=snr, calexp=calexp, x=x, y=y,
        use_kernel_image=use_kernel_image,
        L_pix=l_pix,
        theta_deg=theta_deg,
        n_samples=n_samples
    )
    F = snr * sigmaF
    return calexp.getPhotoCalib().instFluxToMagnitude(F)

def psf_fit_flux_sigma(
    stack_snr: float,
    calexp,
    x: float,
    y: float,
    *,
    L_pix: float,
    theta_deg: float,
    use_kernel_image: bool = False,
    n_samples: int = 20,
):
    """
    Physics-based predictor/inverter:
    Choose the integrated flux F (and corresponding magnitude) such that the *PSF-flux SNR*
    (i.e. what Stack reports as stack_snr) is equal to `stack_snr` for a trailed source.

    Model:
      Stack PSF-flux estimator ~ weighted LS with PSF template phi
      True source image = F * T  (T is PSF⊗line unit-flux template)

    Expected PSF SNR:
      SNR_psf = F * sum(phi*T/V) / sqrt(sum(phi^2/V))

    Solve for F:
      F = stack_snr * sqrt(sum(phi^2/V)) / sum(phi*T/V)
    """
    # PSF stamp + variance stamp (aligned, clipped if near edges)
    p_cut, v_cut = _get_psf_stamp_and_var(calexp, float(x), float(y), use_kernel_image=use_kernel_image)

    s = p_cut.sum()
    if not np.isfinite(s) or s <= 0:
        return np.nan
    phi = (p_cut / s).astype(np.float64)  # unit integrated flux PSF template

    Tcut = _trail_template_cut_for_psf_stamp(
        calexp, float(x), float(y),
        L_pix=float(L_pix), theta_deg=float(theta_deg),
        psf_shape=phi.shape,
        use_kernel_image=use_kernel_image,
        n_samples=int(n_samples),
    )
    if Tcut is None:
        return np.nan

    good = np.isfinite(v_cut) & (v_cut > 0) & np.isfinite(phi) & np.isfinite(Tcut)
    if not np.any(good):
        return np.nan

    denom_phi = np.sum((phi[good] ** 2) / v_cut[good])  # = 1/sigma_psf_wls^2
    overlap = np.sum((phi[good] * Tcut[good]) / v_cut[good])

    if not np.isfinite(denom_phi) or denom_phi <= 0 or not np.isfinite(overlap) or overlap <= 0:
        return np.nan

    return float(np.sqrt(denom_phi)) / float(overlap)
    
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
    
def _trail_template_cut_for_psf_stamp(
    calexp, x, y, *, L_pix: float, theta_deg: float, psf_shape, use_kernel_image: bool, n_samples: int = 81
):
    """
    Build a unit-flux trail template on a large stamp and return the *central cutout*
    with the same shape as the PSF stamp (psf_shape).
    """
    psf = calexp.getPsf()

    if use_kernel_image and hasattr(psf, "computeKernelImage"):
        psf_img = psf.computeKernelImage(geom.Point2D(float(x), float(y))).array.astype(np.float64)
    else:
        psf_img = psf.computeImage(geom.Point2D(float(x), float(y))).array.astype(np.float64)

    s = psf_img.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    psf_img = psf_img / s  # unit integrated flux

    ph, pw = psf_img.shape
    pad = int(np.ceil(L_pix)) + 4
    ah = ph + 2 * pad
    aw = pw + 2 * pad

    T = np.zeros((ah, aw), dtype=np.float64)

    theta = np.deg2rad(theta_deg)
    ux = np.cos(theta)
    uy = np.sin(theta)

    # aim for ~0.25 px spacing along the trail (at least 21 samples)
    step = 0.25
    n = max(21, int(np.ceil(L_pix / step)) + 1)
    s_vals = np.linspace(-0.5*L_pix, 0.5*L_pix, n)


    #s_vals = np.linspace(-0.5 * L_pix, 0.5 * L_pix, int(n_samples))
    for sv in s_vals:
        _add_shifted(T, psf_img, dx=sv * ux, dy=sv * uy)

    Ts = T.sum()
    if not np.isfinite(Ts) or Ts <= 0:
        return None
    T /= Ts  # unit integrated flux trail template

    # central cut to match PSF stamp shape
    out_h, out_w = psf_shape
    cy = ah // 2
    cx = aw // 2
    y0 = cy - out_h // 2
    x0 = cx - out_w // 2
    y1 = y0 + out_h
    x1 = x0 + out_w

    if y0 < 0 or x0 < 0 or y1 > ah or x1 > aw:
        # shouldn't happen unless shapes are pathological
        return T[max(y0,0):min(y1,ah), max(x0,0):min(x1,aw)]

    return T[y0:y1, x0:x1]
    
    
def _add_shifted(acc, psf_img, dx, dy):
    """
    Add psf_img shifted by (dx,dy) pixels into acc, with clipping. dx,dy can be fractional.
    Uses simple bilinear sampling via integer rounding (fast, good enough for this test).
    """
    # For a first test, integer shifts are fine; fractional can be added later if needed.
    sx = int(round(dx))
    sy = int(round(dy))

    ph, pw = psf_img.shape
    ah, aw = acc.shape

    # center align
    # acc is the target stamp centered on (cx,cy), psf_img is centered by PSF rendering
    # We'll paste psf_img into acc with shift (sx,sy).
    x0 = (aw - pw)//2 + sx
    y0 = (ah - ph)//2 + sy
    x1 = x0 + pw
    y1 = y0 + ph

    ax0 = max(x0, 0); ay0 = max(y0, 0)
    ax1 = min(x1, aw); ay1 = min(y1, ah)

    if ax0 >= ax1 or ay0 >= ay1:
        return

    px0 = ax0 - x0; py0 = ay0 - y0
    px1 = px0 + (ax1 - ax0); py1 = py0 + (ay1 - ay0)

    acc[ay0:ay1, ax0:ax1] += psf_img[py0:py1, px0:px1]

