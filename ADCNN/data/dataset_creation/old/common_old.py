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

######################### SNR experiments ##################################################################################
import numpy as np
import lsst.geom as geom

def _extract_var_stamp(calexp, cx, cy, h, w):
    var_full = calexp.variance.array.astype(np.float64)
    H, W = var_full.shape
    x0 = cx - w//2; y0 = cy - h//2
    x1 = x0 + w;    y1 = y0 + h
    ix0 = max(x0, 0); iy0 = max(y0, 0)
    ix1 = min(x1, W); iy1 = min(y1, H)
    return var_full[iy0:iy1, ix0:ix1], (ix0, iy0, ix1, iy1), (x0, y0)

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


def sigmaF_trail_matched(calexp, x, y, L_pix, theta_deg, *,
                         n_samples=200, use_kernel_image=False):
    """
    Compute 1-sigma flux uncertainty for an optimal *trail-matched* filter:
      T = (PSF convolved with a line segment of length L and angle theta), normalized to sum(T)=1
      sigmaF = (sum(T^2 / V))^(-1/2)

    Returns (sigmaF, NEA_trail, stamp_shape)
      where NEA_trail = 1/sum(T^2) for constant variance (useful diagnostic)
    """
    psf = calexp.getPsf()

    # PSF image at position
    if use_kernel_image and hasattr(psf, "computeKernelImage"):
        psf_img = psf.computeKernelImage(geom.Point2D(float(x), float(y))).array.astype(np.float64)
        psf_width = psf.computeKernelImage(geom.Point2D(float(x), float(y))).getWidth()
    else:
        psf_img = psf.computeImage(geom.Point2D(float(x), float(y))).array.astype(np.float64)
        psf_width = psf.computeImage(geom.Point2D(float(x), float(y))).getWidth()

    # normalize PSF to unit flux
    s = psf_img.sum()
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan, psf_img.shape
    psf_img = psf_img / s

    # choose a stamp size big enough to hold the trail+PSF
    # (pad PSF stamp by ~L in both directions)
    ph, pw = psf_img.shape
    
    #L_pix = min([int(psf_width)/2, L_pix])
    
    pad = int(np.ceil(L_pix)) + 4
    ah = ph + 2*pad
    aw = pw + 2*pad

    # build trailed template by averaging shifted PSFs along the line
    T = np.zeros((ah, aw), dtype=np.float64)

    theta = np.deg2rad(theta_deg)
    ux = np.cos(theta)
    uy = np.sin(theta)

    # sample positions along the line segment (centered at 0)
    # spacing so endpoints included
    s_vals = np.linspace(-0.5*L_pix, 0.5*L_pix, int(n_samples))

    for sv in s_vals:
        _add_shifted(T, psf_img, dx=sv*ux, dy=sv*uy)

    # normalize to unit integrated flux (trail template)
    Ts = T.sum()
    if not np.isfinite(Ts) or Ts <= 0:
        return np.nan, np.nan, T.shape
    T /= Ts

    
    # variance stamp at (x,y) with same bbox as T centered on rounded pixel
    cx = int(round(x)); cy = int(round(y))
    V, (ix0, iy0, ix1, iy1), (x0, y0) = _extract_var_stamp(calexp, cx, cy, ah, aw)

    # clip T to the same region as V (if near edges)
    # compute overlap indices
    # T coords corresponding to V region
    tx0 = ix0 - x0
    ty0 = iy0 - y0
    tx1 = tx0 + (ix1 - ix0)
    ty1 = ty0 + (iy1 - iy0)
    Tcut = T[ty0:ty1, tx0:tx1]


    good = np.isfinite(V) & (V > 0) & np.isfinite(Tcut)
    denom = np.sum((Tcut[good]**2) / V[good])
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, np.nan, T.shape

    sigmaF = float(np.sqrt(1.0 / denom))
    nea_trail = float(1.0 / np.sum(Tcut[good]**2))  # constant-variance diagnostic
    return sigmaF, nea_trail, T.shape

###########################################################################################
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


def stack_snr_to_mag(
    stack_snr: float,
    calexp,
    x: float,
    y: float,
    *,
    L_pix: float,
    theta_deg: float,
    use_kernel_image: bool = False,
    n_samples: int = 81,
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

    F = float(stack_snr) * float(np.sqrt(denom_phi)) / float(overlap)
    return calexp.getPhotoCalib().instFluxToMagnitude(F)
#############################################################################################################

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
def psf_fit_flux_sigma(calexp, x, y, *, L_pix=None, theta_deg=None, estimator="trail_matched", use_kernel_image=False):
    """
    estimator:
      - "wls"      : inverse-variance weighted LS (recommended statistically)
      - "constvar" : constant-variance approximation
        - "trail_matched": optimal trail-matched
    use_kernel_image:
      - False: psf.computeImage(...)
      - True : psf.computeKernelImage(...) if available, else falls back to computeImage
    """
    if estimator == "wls":
        return sigma_psf_wls(calexp, x, y, use_kernel_image=use_kernel_image)
    elif estimator == "constvar":
        return sigma_psf_constvar(calexp, x, y, use_kernel_image=use_kernel_image)
    elif estimator == "trail_matched":
        if L_pix is None:
            raise ValueError("L_pix must be provided for 'trail_matched' estimator.")
        if theta_deg is None:
            raise ValueError("theta_deg must be provided for 'trail_matched' estimator.")
        return sigmaF_trail_matched(calexp, x, y, L_pix=L_pix, theta_deg=theta_deg, use_kernel_image=use_kernel_image)[0]
    else:
        raise ValueError(f"Unknown estimator={estimator!r}. Use 'wls', 'constvar' or 'trail_matched'.")

def mag_to_snr(mag, calexp, x, y, *, estimator="wls", use_kernel_image=False, l_pix=None, theta_deg=None):
    F = calexp.getPhotoCalib().magnitudeToInstFlux(mag)
    sigmaF = psf_fit_flux_sigma(calexp, x, y, estimator=estimator, use_kernel_image=use_kernel_image, L_pix=l_pix, theta_deg=theta_deg)
    return F / sigmaF

def snr_to_mag(snr, calexp, x, y, *, estimator="wls", use_kernel_image=False, l_pix=None, theta_deg=None):
    sigmaF = psf_fit_flux_sigma(
        calexp, x, y,
        estimator=estimator,
        use_kernel_image=use_kernel_image,
        L_pix=l_pix,
        theta_deg=theta_deg,
    )
    F = snr * sigmaF
    return calexp.getPhotoCalib().instFluxToMagnitude(F)

