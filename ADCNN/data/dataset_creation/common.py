from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import lsst.geom as geom


# ======================================================================================
# Small utils
# ======================================================================================

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# ======================================================================================
# Drawing helper
# ======================================================================================

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
# PSF helper
# ======================================================================================

def psf_fwhm_arcsec_from_calexp(
    calexp,
    x: float,
    y: float,
    *,
    use_kernel_image: bool = False,
) -> float:
    """
    Estimate PSF FWHM (arcsec) from the PSF image at (x,y) using second moments.
    Assumes an equivalent circular Gaussian for sigma->FWHM conversion.
    """
    psf_img = _compute_psf_image(calexp, x, y, use_kernel_image=use_kernel_image)
    if psf_img is None:
        return np.nan

    s = float(np.sum(psf_img))
    if not np.isfinite(s) or s <= 0:
        return np.nan

    psf_unit = (psf_img / s).astype(np.float64)
    sigma_pix = _estimate_sigma_from_psf(psf_unit)  # pixels
    if not np.isfinite(sigma_pix) or sigma_pix <= 0:
        return np.nan

    fwhm_pix = 2.354820045 * sigma_pix
    pixel_scale = calexp.wcs.getPixelScale().asArcseconds()  # arcsec / pix
    return float(fwhm_pix * pixel_scale)


def start_to_midpoint(x0: float, y0: float, l_pix: float, theta_deg: float) -> tuple[float, float]:
    th = math.radians(theta_deg)
    xm = x0 + 0.5 * l_pix * math.cos(th)
    ym = y0 + 0.5 * l_pix * math.sin(th)
    return xm, ym


# ======================================================================================
# SNR conversion helpers
# ======================================================================================

def mag_to_snr(
    mag,
    calexp,
    x,
    y,
    *,
    use_kernel_image: bool = False,
    l_pix: float,
    theta_deg: float,
    pad_sigma: float = 5.0,
    step: float = 0.15,
) -> float:
    """
    Predict *stack-like* SNR for a trail of length l_pix at (x,y):
        snr_stack ≈ (F / sigmaF_model) * C(L)
    """
    F = float(calexp.getPhotoCalib().magnitudeToInstFlux(mag))
    xm, ym = start_to_midpoint(float(x), float(y), float(l_pix), float(theta_deg))

    sigmaF = psf_fit_flux_sigma(
        calexp=calexp,
        x=xm,
        y=ym,
        L_pix=float(l_pix),
        theta_deg=float(theta_deg),
        use_kernel_image=use_kernel_image,
        pad_sigma=pad_sigma,
        step=step,
    )
    if not np.isfinite(sigmaF) or sigmaF <= 0:
        return float("nan")

    snr_model = F / float(sigmaF)
    return float(snr_model * empirical_stack_snr_correction(l_pix))


def snr_to_mag(
    snr,
    calexp,
    x,
    y,
    *,
    use_kernel_image: bool = False,
    l_pix: float,
    theta_deg: float,
    pad_sigma: float = 5.0,
    step: float = 0.15,
) -> float:
    """
    Convert target *stack SNR* to magnitude using the stack-like model:
        F ≈ stack_snr * sigmaF_model / C(L)
    """
    C = empirical_stack_snr_correction(l_pix)
    if not np.isfinite(C) or C <= 0:
        return float("nan")

    xm, ym = start_to_midpoint(float(x), float(y), float(l_pix), float(theta_deg))
    sigmaF = psf_fit_flux_sigma(
        calexp=calexp,
        x=xm,
        y=ym,
        L_pix=float(l_pix),
        theta_deg=float(theta_deg),
        use_kernel_image=use_kernel_image,
        pad_sigma=pad_sigma,
        step=step,
    )
    if not np.isfinite(sigmaF) or sigmaF <= 0:
        return float("nan")

    F = float(snr) * float(sigmaF) / float(C)
    return float(calexp.getPhotoCalib().instFluxToMagnitude(F))


# ======================================================================================
# Core model: sigmaF for PSF-flux estimator on a trailed source
# ======================================================================================

def psf_fit_flux_sigma(
    calexp,
    x: float,
    y: float,
    *,
    L_pix: float,
    theta_deg: float,
    use_kernel_image: bool = False,
    pad_sigma: float = 5.0,
    step: float = 0.15,
) -> float:
    """
    Predict sigmaF for the *PSF-flux estimator* applied to a trailed source:
        sigmaF = sqrt(sum(phi^2 / V)) / sum(phi*T / V)

    phi: PSF template (unit integrated flux)
    T:   trail template (PSF convolved with uniform line; unit integrated flux)
    V:   per-pixel variance from calexp.variance
    """
    psf_img_native = _compute_psf_image(calexp, x, y, use_kernel_image=use_kernel_image)
    if psf_img_native is None:
        return float("nan")

    psf_sum = float(psf_img_native.sum())
    if not np.isfinite(psf_sum) or psf_sum <= 0:
        return float("nan")

    psf_unit_native = (psf_img_native / psf_sum).astype(np.float64)
    sigma_pix = _estimate_sigma_from_psf(psf_unit_native)
    if not np.isfinite(sigma_pix) or sigma_pix <= 0:
        return float("nan")

    # stamp size
    R = int(math.ceil(pad_sigma * sigma_pix))
    R = max(R, 20)
    S = int(math.ceil(float(L_pix))) + 2 * R + 1
    S = max(S, 33)
    if S % 2 == 0:
        S += 1
    out_shape = (S, S)

    phi_cut, v_cut = _get_psf_stamp_and_var(
        calexp,
        x=float(x),
        y=float(y),
        use_kernel_image=use_kernel_image,
        out_shape=out_shape,
    )
    if phi_cut is None or v_cut is None:
        return float("nan")

    s = float(np.nansum(phi_cut))
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    phi = (phi_cut / s).astype(np.float64)

    T = _trail_template_for_stamp(
        calexp=calexp,
        x=float(x),
        y=float(y),
        L_pix=float(L_pix),
        theta_deg=float(theta_deg),
        out_shape=out_shape,
        use_kernel_image=use_kernel_image,
        step=float(step),
    )
    if T is None:
        return float("nan")

    good = np.isfinite(v_cut) & (v_cut > 0) & np.isfinite(phi) & np.isfinite(T)
    if not np.any(good):
        return float("nan")

    denom_phi = float(np.sum((phi[good] ** 2) / v_cut[good]))
    overlap = float(np.sum((phi[good] * T[good]) / v_cut[good]))
    if not np.isfinite(denom_phi) or denom_phi <= 0 or not np.isfinite(overlap) or overlap <= 0:
        return float("nan")

    return float(np.sqrt(denom_phi) / overlap)


# ======================================================================================
# Internal helpers
# ======================================================================================

def _compute_psf_image(calexp, x: float, y: float, *, use_kernel_image: bool) -> Optional[np.ndarray]:
    psf = calexp.getPsf()
    try:
        if use_kernel_image and hasattr(psf, "computeKernelImage"):
            return psf.computeKernelImage(geom.Point2D(float(x), float(y))).array.astype(np.float64)
        return psf.computeImage(geom.Point2D(float(x), float(y))).array.astype(np.float64)
    except Exception:
        return None


def _estimate_sigma_from_psf(phi_unit: np.ndarray) -> float:
    """Estimate circular sigma (pixels) from second moments of a unit-flux PSF image."""
    H, W = phi_unit.shape
    yy, xx = np.mgrid[0:H, 0:W]
    tot = float(phi_unit.sum())
    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    cy = float((yy * phi_unit).sum() / tot)
    cx = float((xx * phi_unit).sum() / tot)
    dy = yy - cy
    dx = xx - cx
    m2 = float(((dx * dx + dy * dy) * phi_unit).sum() / tot / 2.0)
    if not np.isfinite(m2) or m2 <= 0:
        return np.nan
    return float(math.sqrt(m2))


def _center_crop_pad(img: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Return array of shape out_shape by center-cropping or zero-padding img."""
    H, W = out_shape
    out = np.zeros((H, W), dtype=np.float64)

    ih, iw = img.shape
    cy, cx = ih // 2, iw // 2
    oy, ox = H // 2, W // 2

    ys0 = max(0, cy - oy)
    xs0 = max(0, cx - ox)
    ys1 = min(ih, ys0 + H)
    xs1 = min(iw, xs0 + W)

    yd0 = max(0, oy - cy)
    xd0 = max(0, ox - cx)
    yd1 = yd0 + (ys1 - ys0)
    xd1 = xd0 + (xs1 - xs0)

    out[yd0:yd1, xd0:xd1] = img[ys0:ys1, xs0:xs1]
    return out


def _get_psf_stamp_and_var(
    calexp,
    x: float,
    y: float,
    *,
    use_kernel_image: bool = False,
    out_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (p_cut, v_cut) aligned on out_shape stamp centered at (x,y):
      - p_cut: PSF template stamp (float64), center-cropped/padded
      - v_cut: variance stamp from calexp.variance (float64), NaNs outside bounds
    """
    psf_img = _compute_psf_image(calexp, x, y, use_kernel_image=use_kernel_image)
    if psf_img is None:
        return None, None

    p_cut = _center_crop_pad(psf_img.astype(np.float64), out_shape)

    var_full = calexp.variance.array.astype(np.float64)
    Himg, Wimg = var_full.shape

    H, W = out_shape
    cx = int(round(x))
    cy = int(round(y))
    x0 = cx - W // 2
    y0 = cy - H // 2
    x1 = x0 + W
    y1 = y0 + H

    v_cut = np.full((H, W), np.nan, dtype=np.float64)

    ix0 = max(x0, 0)
    iy0 = max(y0, 0)
    ix1 = min(x1, Wimg)
    iy1 = min(y1, Himg)
    if ix0 >= ix1 or iy0 >= iy1:
        return p_cut, v_cut

    vx0 = ix0 - x0
    vy0 = iy0 - y0
    vx1 = vx0 + (ix1 - ix0)
    vy1 = vy0 + (iy1 - iy0)

    v_cut[vy0:vy1, vx0:vx1] = var_full[iy0:iy1, ix0:ix1]
    return p_cut, v_cut


def _psf_centroid(psf_img: np.ndarray) -> tuple[float, float]:
    ph, pw = psf_img.shape
    yy, xx = np.mgrid[0:ph, 0:pw]
    w = np.maximum(psf_img.astype(np.float64), 0.0)
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0:
        return (pw - 1) / 2.0, (ph - 1) / 2.0
    cx = float((xx * w).sum() / tot)
    cy = float((yy * w).sum() / tot)
    return cx, cy


def _add_shifted_into_center(
    acc: np.ndarray,
    psf_img: np.ndarray,
    dx: float,
    dy: float,
    *,
    psf_cx: float | None = None,
    psf_cy: float | None = None,
    interp=cv2.INTER_LINEAR,
) -> None:
    """Centroid-aware subpixel placement of PSF into acc."""
    ah, aw = acc.shape
    ph, pw = psf_img.shape

    acc_cx = (aw - 1) / 2.0
    acc_cy = (ah - 1) / 2.0

    if psf_cx is None or psf_cy is None:
        psf_cx, psf_cy = _psf_centroid(psf_img)

    target_cx = acc_cx + float(dx)
    target_cy = acc_cy + float(dy)

    x0 = int(np.floor(target_cx - psf_cx))
    y0 = int(np.floor(target_cy - psf_cy))

    fx = float(target_cx - (x0 + psf_cx))
    fy = float(target_cy - (y0 + psf_cy))

    M = np.array([[1.0, 0.0, fx],
                  [0.0, 1.0, fy]], dtype=np.float32)

    shifted = cv2.warpAffine(
        psf_img.astype(np.float32),
        M,
        dsize=(pw, ph),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float64)

    x1 = x0 + pw
    y1 = y0 + ph

    ax0 = max(x0, 0); ay0 = max(y0, 0)
    ax1 = min(x1, aw); ay1 = min(y1, ah)
    if ax0 >= ax1 or ay0 >= ay1:
        return

    px0 = ax0 - x0
    py0 = ay0 - y0
    px1 = px0 + (ax1 - ax0)
    py1 = py0 + (ay1 - ay0)

    acc[ay0:ay1, ax0:ax1] += shifted[py0:py1, px0:px1]


def _trail_template_for_stamp(
    calexp,
    x: float,
    y: float,
    *,
    L_pix: float,
    theta_deg: float,
    out_shape: Tuple[int, int],
    use_kernel_image: bool,
    step: float = 0.15,
) -> Optional[np.ndarray]:
    """Build unit-flux trailed template T = PSF ⊗ (uniform line segment)."""
    psf_img = _compute_psf_image(calexp, x, y, use_kernel_image=use_kernel_image)
    if psf_img is None:
        return None

    s = float(psf_img.sum())
    if not np.isfinite(s) or s <= 0:
        return None

    psf_unit = (psf_img / s).astype(np.float64)
    psf_cx, psf_cy = _psf_centroid(psf_unit)

    H, W = out_shape
    T = np.zeros((H, W), dtype=np.float64)

    theta = math.radians(float(theta_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)

    step = float(step)
    if step <= 0:
        raise ValueError("step must be > 0")

    n = max(int(math.ceil(float(L_pix) / step)) + 1, 21)
    s_vals = np.linspace(-0.5 * float(L_pix), 0.5 * float(L_pix), n)

    for sv in s_vals:
        _add_shifted_into_center(
            acc=T,
            psf_img=psf_unit,
            dx=float(sv * ux),
            dy=float(sv * uy),
            psf_cx=psf_cx,
            psf_cy=psf_cy,
        )

    np.maximum(T, 0.0, out=T)
    Ts = float(T.sum())
    if not np.isfinite(Ts) or Ts <= 0:
        return None
    T /= Ts
    return T


def empirical_stack_snr_correction(L_pix: float) -> float:
    # Fit A (power-saturation)
    a  = 1.12746468
    b  = 0.16618154
    L0 = 26.68594499
    p  = 5.01963910
    L = max(float(L_pix), 0.0)
    x = (L / L0) ** p if L0 > 0 else 0.0
    return a + b * (x / (1.0 + x))
