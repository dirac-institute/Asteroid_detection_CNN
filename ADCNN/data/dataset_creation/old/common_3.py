from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2

import lsst.geom as geom
from lsst.daf.butler import Butler


# ======================================================================================
# Small utils
# ======================================================================================

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


@contextmanager
def suppress_stdout():
    import sys, io
    old = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old


# ======================================================================================
# Sky-motion geometry
# ======================================================================================

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
# SNR conversion helpers (Stack PSF-flux SNR, with centroid-jitter marginalization)
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
    centroid_jitter: bool = True,
    jitter_grid_step: float = 0.25,
    jitter_nsig: float = 3.0,
    iters: int = 2,
) -> float:
    """
    Predict Stack's PSF-flux SNR (stack_snr) for a *trailed* source with given magnitude.

    If centroid_jitter=True, we self-consistently account for centroid uncertainty by
    fixed-point iteration:
        - compute SNR assuming perfect centroid (no jitter) -> snr0
        - recompute sigmaF using expected overlap averaged over centroid jitter at snr0 -> snr1
        - repeat a couple of times (usually 1-2 is enough)
    """
    F = float(calexp.getPhotoCalib().magnitudeToInstFlux(mag))

    # Start with "no jitter" to get an initial snr guess
    sigmaF = psf_fit_flux_sigma(
        calexp=calexp,
        x=float(x),
        y=float(y),
        L_pix=float(l_pix),
        theta_deg=float(theta_deg),
        snr_target=10.0,  # dummy; unused when centroid_jitter=False
        use_kernel_image=use_kernel_image,
        pad_sigma=pad_sigma,
        centroid_jitter=False,
        jitter_grid_step=jitter_grid_step,
        jitter_nsig=jitter_nsig,
    )
    if not np.isfinite(sigmaF) or sigmaF <= 0:
        return float("nan")
    snr = F / sigmaF

    if not centroid_jitter:
        return float(snr)

    # Fixed-point refinement
    iters = max(int(iters), 1)
    for _ in range(iters):
        sigmaF = psf_fit_flux_sigma(
            calexp=calexp,
            x=float(x),
            y=float(y),
            L_pix=float(l_pix),
            theta_deg=float(theta_deg),
            snr_target=float(snr),
            use_kernel_image=use_kernel_image,
            pad_sigma=pad_sigma,
            centroid_jitter=True,
            jitter_grid_step=jitter_grid_step,
            jitter_nsig=jitter_nsig,
        )
        if not np.isfinite(sigmaF) or sigmaF <= 0:
            return float("nan")
        snr = F / sigmaF

    return float(snr)


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
    centroid_jitter: bool = True,
    jitter_grid_step: float = 0.25,
    jitter_nsig: float = 3.0,
) -> float:
    """
    Convert target Stack PSF-flux SNR (stack_snr) to magnitude for a trailed source.

    centroid_jitter=True means: pick a magnitude such that the *expected* stack_snr
    (marginalized over centroid jitter) equals the requested snr.
    """
    snr = float(snr)
    sigmaF = psf_fit_flux_sigma(
        calexp=calexp,
        x=float(x),
        y=float(y),
        L_pix=float(l_pix),
        theta_deg=float(theta_deg),
        snr_target=snr,
        use_kernel_image=use_kernel_image,
        pad_sigma=pad_sigma,
        centroid_jitter=centroid_jitter,
        jitter_grid_step=jitter_grid_step,
        jitter_nsig=jitter_nsig,
    )
    if not np.isfinite(sigmaF) or sigmaF <= 0:
        return float("nan")

    F = snr * float(sigmaF)
    return float(calexp.getPhotoCalib().instFluxToMagnitude(F))


def psf_fit_flux_sigma(
    calexp,
    x: float,
    y: float,
    *,
    L_pix: float,
    theta_deg: float,
    snr_target: float,
    use_kernel_image: bool = False,
    pad_sigma: float = 5.0,
    centroid_jitter: bool = True,
    jitter_grid_step: float = 0.25,
    jitter_nsig: float = 3.0,
) -> float:
    """
    Predict sigmaF such that:  F = stack_snr * sigmaF

    Model:
      Stack PSF-flux estimator ~ weighted LS with PSF template phi
      True source image = F * T  (T is PSF⊗line unit-flux template)

    Expected PSF SNR (evaluated at some centroid):
      SNR_psf = F * sum(phi*T/V) / sqrt(sum(phi^2/V))

    Solve for sigmaF:
      sigmaF = sqrt(sum(phi^2/V)) / sum(phi*T/V)

    If centroid_jitter=True, we replace overlap = sum(phi*T/V) with its expectation
    over centroid offsets (Δx,Δy), using a simple anisotropic Gaussian jitter model
    whose scales depend on (PSF sigma, trail length, snr_target).
    """
    # Native PSF for sigma estimate
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

    # Large common stamp (Fix A)
    R = int(math.ceil(float(pad_sigma) * sigma_pix))
    R = max(R, 20)  # hard floor to avoid truncation if sigma estimate is too small
    S = int(math.ceil(float(L_pix))) + 2 * R + 1
    if S < 33:
        S = 33
    if S % 2 == 0:
        S += 1
    out_shape = (S, S)

    # phi and variance on same stamp
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

    # trail template on same stamp (Fix A + B)
    T = _trail_template_for_stamp(
        calexp=calexp,
        x=float(x),
        y=float(y),
        L_pix=float(L_pix),
        theta_deg=float(theta_deg),
        out_shape=out_shape,
        use_kernel_image=use_kernel_image,
    )
    if T is None:
        return float("nan")

    good = np.isfinite(v_cut) & (v_cut > 0) & np.isfinite(phi) & np.isfinite(T)
    if not np.any(good):
        return float("nan")

    denom_phi = float(np.sum((phi[good] ** 2) / v_cut[good]))
    if not np.isfinite(denom_phi) or denom_phi <= 0:
        return float("nan")

    if centroid_jitter:
        overlap = _expected_overlap_centroid_jitter(
            phi=phi,
            T=T,
            v=v_cut,
            sigma_psf=float(sigma_pix),
            L_pix=float(L_pix),
            theta_deg=float(theta_deg),
            snr_target=float(snr_target),
            nsig=float(jitter_nsig),
            grid_step=float(jitter_grid_step),
        )
    else:
        overlap = float(np.sum((phi[good] * T[good]) / v_cut[good]))

    if not np.isfinite(overlap) or overlap <= 0:
        return float("nan")

    return float(math.sqrt(denom_phi) / overlap)


# ======================================================================================
# Centroid-jitter marginalization (new)
# ======================================================================================

def _centroid_sigmas_pixels(sigma_psf: float, L_pix: float, snr: float) -> tuple[float, float]:
    """
    Simple CRLB-inspired centroid jitter model (pixels), anisotropic for trails.

    Parallel width uses second-moment of a uniform line segment: L/sqrt(12).
    """
    snr = max(float(snr), 1e-3)
    sigma_psf = float(sigma_psf)
    L_pix = float(L_pix)

    sigma_para = math.sqrt(max(sigma_psf**2 + (L_pix**2) / 12.0, 1e-12)) / snr
    sigma_perp = max(sigma_psf, 1e-6) / snr

    # Small floors to avoid degeneracies / pixelization
    sigma_para = max(sigma_para, 0.03)
    sigma_perp = max(sigma_perp, 0.03)
    return sigma_para, sigma_perp


def _overlap_with_shift(phi: np.ndarray, T: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    """
    overlap(dx,dy) = sum( phi_shifted * T / v )
    with phi shifted by (dx,dy) pixels.
    """
    ph, pw = phi.shape
    M = np.array([[1.0, 0.0, float(dx)],
                  [0.0, 1.0, float(dy)]], dtype=np.float32)

    phi_s = cv2.warpAffine(
        phi.astype(np.float32),
        M,
        dsize=(pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float64)

    good = np.isfinite(v) & (v > 0) & np.isfinite(phi_s) & np.isfinite(T)
    if not np.any(good):
        return float("nan")
    return float(np.sum((phi_s[good] * T[good]) / v[good]))


def _expected_overlap_centroid_jitter(
    *,
    phi: np.ndarray,
    T: np.ndarray,
    v: np.ndarray,
    sigma_psf: float,
    L_pix: float,
    theta_deg: float,
    snr_target: float,
    nsig: float = 3.0,
    grid_step: float = 0.25,
) -> float:
    """
    Approximate E[overlap] over centroid offsets with a small weighted grid in (parallel,perp),
    then rotate into (dx,dy).

    This is NOT "fitting": it just propagates centroid uncertainty into expected PSF-flux SNR.
    """
    sigma_para, sigma_perp = _centroid_sigmas_pixels(float(sigma_psf), float(L_pix), float(snr_target))

    grid_step = max(float(grid_step), 0.05)
    nsig = max(float(nsig), 1.0)

    r_para = max(nsig * sigma_para, grid_step)
    r_perp = max(nsig * sigma_perp, grid_step)

    n_para = int(math.ceil(2 * r_para / grid_step)) + 1
    n_perp = int(math.ceil(2 * r_perp / grid_step)) + 1

    para_vals = (np.arange(n_para) - n_para // 2) * grid_step
    perp_vals = (np.arange(n_perp) - n_perp // 2) * grid_step

    theta = math.radians(float(theta_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)
    vx = -uy
    vy = ux

    w_sum = 0.0
    ov_sum = 0.0

    for dp in para_vals:
        for dq in perp_vals:
            w = math.exp(-0.5 * ((dp / sigma_para) ** 2 + (dq / sigma_perp) ** 2))
            dx = dp * ux + dq * vx
            dy = dp * uy + dq * vy

            ov = _overlap_with_shift(phi, T, v, dx=dx, dy=dy)
            if np.isfinite(ov):
                ov_sum += w * ov
                w_sum += w

    if w_sum <= 0:
        return float("nan")
    return float(ov_sum / w_sum)


# ======================================================================================
# Internal helpers (existing)
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
    """
    Estimate circular sigma (pixels) from second moments of a unit-flux PSF image.
    """
    H, W = phi_unit.shape
    yy, xx = np.mgrid[0:H, 0:W]
    tot = float(phi_unit.sum())
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    cy = float((yy * phi_unit).sum() / tot)
    cx = float((xx * phi_unit).sum() / tot)
    dy = yy - cy
    dx = xx - cx
    m2 = float(((dx * dx + dy * dy) * phi_unit).sum() / tot / 2.0)
    if not np.isfinite(m2) or m2 <= 0:
        return float("nan")
    return float(math.sqrt(m2))


def _center_crop_pad(img: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Return array of shape out_shape by center-cropping or zero-padding img.
    """
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
    Returns (p_cut, v_cut) aligned with each other on an out_shape stamp centered at (x,y):
      - p_cut: PSF image on out_shape (center-crop/pad)
      - v_cut: variance on same bbox; NaNs outside image bounds
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


def _trail_template_for_stamp(
    calexp,
    x: float,
    y: float,
    *,
    L_pix: float,
    theta_deg: float,
    out_shape: Tuple[int, int],
    use_kernel_image: bool,
) -> Optional[np.ndarray]:
    """
    Build a unit-flux trailed template T on out_shape by summing subpixel-shifted PSFs.
    """
    psf_img = _compute_psf_image(calexp, x, y, use_kernel_image=use_kernel_image)
    if psf_img is None:
        return None

    s = float(psf_img.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    psf_unit = (psf_img / s).astype(np.float64)

    H, W = out_shape
    T = np.zeros((H, W), dtype=np.float64)

    theta = math.radians(float(theta_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)

    # Sampling along the trail (Fix B)
    step = 0.15 if float(L_pix) <= 50.0 else 0.10
    n = int(math.ceil(float(L_pix) / step)) + 1
    n = max(n, 21)

    s_vals = np.linspace(-0.5 * float(L_pix), 0.5 * float(L_pix), n)

    for sv in s_vals:
        _add_shifted_into_center(acc=T, psf_img=psf_unit, dx=float(sv * ux), dy=float(sv * uy))

    Ts = float(T.sum())
    if not np.isfinite(Ts) or Ts <= 0:
        return None
    T /= Ts
    return T


def _add_shifted_into_center(
    acc: np.ndarray,
    psf_img: np.ndarray,
    dx: float,
    dy: float,
) -> None:
    """
    Add psf_img shifted by (dx,dy) into acc (centered paste) using warpAffine.
    """
    ph, pw = psf_img.shape
    ah, aw = acc.shape

    M = np.array([[1.0, 0.0, float(dx)],
                  [0.0, 1.0, float(dy)]], dtype=np.float32)

    shifted = cv2.warpAffine(
        psf_img.astype(np.float32),
        M,
        dsize=(pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float64)

    x0 = (aw - pw) // 2
    y0 = (ah - ph) // 2
    x1 = x0 + pw
    y1 = y0 + ph

    ax0 = max(x0, 0)
    ay0 = max(y0, 0)
    ax1 = min(x1, aw)
    ay1 = min(y1, ah)
    if ax0 >= ax1 or ay0 >= ay1:
        return

    px0 = ax0 - x0
    py0 = ay0 - y0
    px1 = px0 + (ax1 - ax0)
    py1 = py0 + (ay1 - ay0)

    acc[ay0:ay1, ax0:ax1] += shifted[py0:py1, px0:px1]

