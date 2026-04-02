"""Deterministic fill-until-target variant of the LSST injection dataset builder.

This side-by-side alternative intentionally avoids the expensive up-front
reference pre-check selection used by the original script. It preserves the
scientific behavior of successful samples as closely as possible while changing
only scheduling and filling: candidate refs are ordered deterministically,
processed directly, failed refs are skipped, and parent-owned writers backfill
train/test outputs until the requested successful dataset sizes are reached or
the candidate pool is exhausted.
"""

from __future__ import annotations
import argparse
import contextlib
import io
from pathlib import Path

from common import ensure_dir, draw_one_line, psf_fwhm_arcsec_from_calexp, mag_to_snr, snr_to_mag
from pipetasks import calibrate, isr, fetch_from_butler, source_detect

import random
from typing import List, Sequence

from astroML.crossmatch import crossmatch_angular
from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
from lsst.geom import Point2D
import lsst.geom as geom
from lsst.source.injection.inject_exposure import ExposureInjectTask
from lsst.meas.extensions.psfex.psfexPsfDeterminer import PsfexNoGoodStarsError
import h5py
import concurrent.futures
from astropy.io import ascii
import os
import shutil
from multiprocessing import Value, Lock, Manager
import logging
import pandas as pd
import traceback
import time
import signal
import gc
import resource

completed_counter = Value('i', 0)
counter_lock = Lock()
_WORKER_BUTLERS = {}
TASK_TIMEOUT_SECONDS = 900
PREINJECTION_DETECTION_THRESHOLD = 3.0
ATTEMPT_DIAGNOSTICS = True
MAX_PRE_SOURCES = 12000


def inject(postISRCCD, injection_catalog):
    inject_task = ExposureInjectTask()
    inject_res = inject_task.run([injection_catalog], postISRCCD, postISRCCD.psf, postISRCCD.photoCalib, postISRCCD.wcs)
    return inject_res.output_exposure


def get_worker_butler(repo, coll):
    key = (str(repo), tuple(coll) if isinstance(coll, (list, tuple)) else str(coll))
    butler = _WORKER_BUTLERS.get(key)
    if butler is None:
        butler = Butler(repo, collections=coll)
        _WORKER_BUTLERS[key] = butler
    return butler


def choose_tmp_dir(save_path, seed):
    base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    return os.path.join(base, f"simulate_inject_fill_deterministic_{os.getpid()}_{int(seed)}")


class TaskTimeoutError(TimeoutError):
    pass


def _alarm_timeout_handler(signum, frame):
    raise TaskTimeoutError(f"Task exceeded timeout of {TASK_TIMEOUT_SECONDS} seconds")


def _require_exposure_attr(exposure, name, getter=None, missing_message=None):
    try:
        value = getter(exposure) if getter is not None else getattr(exposure, name)
    except Exception:
        value = None
    if value is None:
        raise RuntimeError(missing_message or f"Exposure is missing {name}.")
    return value


def validate_preliminary_exposure(calexp):
    _require_exposure_attr(calexp, "wcs", missing_message="Exposure is missing a WCS.")
    _require_exposure_attr(calexp, "psf", missing_message="Exposure is missing a PSF.")
    _require_exposure_attr(calexp, "photoCalib", missing_message="Exposure is missing a photoCalib.")
    _require_exposure_attr(calexp, "visitInfo", missing_message="Exposure is missing visitInfo.")
    _require_exposure_attr(calexp, "filter", missing_message="Exposure is missing a filter.")
    _require_exposure_attr(
        calexp,
        "apCorrMap",
        getter=lambda exp: exp.info.getApCorrMap(),
        missing_message="Exposure is missing an aperture correction map.",
    )


def load_preliminary_from_butler_checked(butler, dataId):
    calexp = butler.get("preliminary_visit_image", dataId=dataId)
    validate_preliminary_exposure(calexp)
    background = butler.get("preliminary_visit_image_background", dataId=dataId)
    return calexp, background


def current_rss_mb():
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss_kb) / 1024.0


def format_attempt_diagnostics(result):
    parts = [
        f"status={result.get('status')}",
        f"rank={result.get('rank')}",
        f"image_id={result.get('image_id')}",
        f"split={result.get('split')}",
        f"visit={result.get('dataId', {}).get('visit') if isinstance(result.get('dataId'), dict) else None}",
        f"detector={result.get('dataId', {}).get('detector') if isinstance(result.get('dataId'), dict) else None}",
    ]
    if "compute_s" in result:
        parts.append(f"compute={float(result['compute_s']):.2f}s")
    if "stage_s" in result:
        parts.append(f"stage={float(result['stage_s']):.2f}s")
    if "promote_s" in result:
        parts.append(f"write={float(result['promote_s']):.2f}s")
    if "rss_max_mb" in result:
        parts.append(f"rss_max={float(result['rss_max_mb']):.1f}MB")
    if result.get("rss_log"):
        parts.append(f"rss_log={result['rss_log']}")
    if result.get("error"):
        parts.append(f"error={result['error']}")
    return " | ".join(parts)

def estimate_m5_local_from_psf_var(calexp, x, y, snr=5.0):
    """
    Local m5 at (x,y) using:
      alpha = sum(phi^2)
      sigma_f = sqrt(sum(phi^2 * V)) / alpha
      f_snr = snr * sigma_f
      m_snr = instFluxToMagnitude(f_snr)
    """
    bbox = calexp.getBBox()
    point = geom.Point2D(float(x), float(y))

    psf = calexp.getPsf()
    phi_img = psf.computeKernelImage(point)
    phi = phi_img.array.astype(np.float64)

    var_full = calexp.getMaskedImage().getVariance()
    phi_bbox = phi_img.getBBox()
    phi_bbox_clipped = phi_bbox.clippedTo(bbox)
    V = var_full[phi_bbox_clipped].array.astype(np.float64)

    if phi_bbox_clipped != phi_bbox:
        dx0 = phi_bbox_clipped.getMinX() - phi_bbox.getMinX()
        dy0 = phi_bbox_clipped.getMinY() - phi_bbox.getMinY()
        dx1 = dx0 + phi_bbox_clipped.getWidth()
        dy1 = dy0 + phi_bbox_clipped.getHeight()
        phi = phi[dy0:dy1, dx0:dx1]

    phi2 = phi * phi
    alpha = float(np.sum(phi2))
    if not np.isfinite(alpha) or alpha <= 0:
        return np.nan

    sigma_f = float(np.sqrt(np.sum(phi2 * V)) / alpha)
    f_snr = snr * sigma_f
    return float(calexp.getPhotoCalib().instFluxToMagnitude(f_snr))

def generate_one_line(n_inject, trail_length, mag, beta, ref, dimensions, seed, calexp, mag_mode="psf_mag", psf_template="image", forbidden_mask=None):
    rng = np.random.default_rng(seed)
    injection_catalog = Table(
        names=('injection_id', 'ra', 'dec', 'source_type', 'trail_length', 'mag', 'beta', 'visit', 'detector',
               'integrated_mag', 'PSF_mag', 'SNR', 'physical_filter', 'x', 'y', 'SNR_estimation', 'm5_local', 'm5_detector'),
        dtype=('int64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'int64', 'int64', 'float64',
               'float64', 'float64', 'str', 'int64', 'int64', 'float64', 'float64', 'float64'))

    H, W = int(dimensions.y), int(dimensions.x)
    if forbidden_mask is None:
        forbidden = np.zeros((H, W), dtype=bool)
    else:
        # Make sure we can safely modify it locally
        forbidden = forbidden_mask.astype(bool, copy=True)
    raw = calexp.wcs
    info = calexp.visitInfo
    filter_name = calexp.filter
    m5 = {"u": 23.7, "g": 24.97, "r": 24.52, "i": 24.13, "z": 23.56, "y": 22.55}
    psf_depth = m5[filter_name.bandLabel]
    #a, b = 0.67, 1.16
    a, b = 0.42, 0
    for k in range(n_inject):
        inject_length = rng.uniform(*trail_length)
        if inject_length <= 0:
            length = 1.0
        else:
            length = inject_length
        # Conservative margin: assume R>=20 and S = ceil(L)+2R+1
        R = 30
        S = int(np.ceil(length)) + 2*R + 1
        half = S // 2 + 2  # +2 pixels slack
        #x_pos = rng.uniform(half, dimensions.x - 1 - half)
        #y_pos = rng.uniform(half, dimensions.y - 1 - half)
        angle = rng.uniform(*beta) if inject_length > 0 else 0.0
        x_pos, y_pos, stamp = _try_place_trail_no_overlap(
            rng,
            forbidden,
            dimensions,
            trail_length_px=length,
            angle_deg=angle,
            half_margin=half,
            calexp=calexp,
            psf_template=psf_template,
            max_tries=2000,
        )

        # Mark these pixels as now-occupied so subsequent injections cannot overlap them
        forbidden |= stamp

        m5_local = estimate_m5_local_from_psf_var(calexp, x_pos, y_pos)
        sky_pos = raw.pixelToSky(x_pos, y_pos)
        ra_pos = sky_pos.getRa().asDegrees()
        dec_pos = sky_pos.getDec().asDegrees()
        use_kernel = (psf_template == "kernel")
        fwhm_arcsec = psf_fwhm_arcsec_from_calexp(calexp, x_pos, y_pos, use_kernel_image=use_kernel)
        if not np.isfinite(fwhm_arcsec) or fwhm_arcsec <= 0:
            fwhm_arcsec = {"u": 0.92, "g": 0.87, "r": 0.83, "i": 0.80, "z": 0.78, "y": 0.76}
            fwhm_arcsec = fwhm_arcsec[filter_name.bandLabel]
        pixelScale = raw.getPixelScale().asArcseconds()
        theta_p = fwhm_arcsec / pixelScale
        x = length / theta_p
        upper_limit_mag = psf_depth - 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x)) if mag[1] == 0 else mag[1]
        if mag_mode == "snr":
            snr_edge = 5.0  # your definition of "edge of detection" (change if you want)
            snr_min = float(mag[0])
            snr_max = float(mag[1])
            if snr_max == 0.0:
                snr_min, snr_max = snr_edge, snr_min
            if snr_max < snr_min:
                raise ValueError(f"Bad SNR range: snr_min={snr_min} snr_max={snr_max}")
            snr = float(rng.uniform(snr_min, snr_max))
            snr = max(snr, 0.01)
            psf_magnitude = snr_to_mag(snr, calexp, x_pos, y_pos, l_pix=length, theta_deg=angle, use_kernel_image=use_kernel, snr_definition="detection")
            if inject_length > 0:
                magnitude = psf_magnitude - 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x))
                surface_brightness = magnitude + 2.5 * np.log10(length)
            else:
                magnitude = psf_magnitude
                surface_brightness = magnitude
            #stack_snr = snr
            stack_snr = mag_to_snr(magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel, l_pix=length, theta_deg=angle, snr_definition="measurement")
        elif mag_mode == "psf_mag":
            psf_magnitude = rng.uniform(mag[0], upper_limit_mag)
            if inject_length > 0:
                magnitude = psf_magnitude - 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x))
                surface_brightness = magnitude + 2.5 * np.log10(length)
            else:
                magnitude = psf_magnitude
                surface_brightness = magnitude
            stack_snr = mag_to_snr(magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel, l_pix=length,
                                   theta_deg=angle, snr_definition="measurement")
            snr = mag_to_snr(psf_magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel,
                             l_pix=length, theta_deg=angle, snr_definition="detection")
        elif mag_mode == "surface_brightness":
            surface_brightness = rng.uniform(mag[0], mag[1])
            if inject_length > 0:
                magnitude = surface_brightness - 2.5 * np.log10(length)
                psf_magnitude = magnitude + 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x))
            else:
                magnitude = surface_brightness
                psf_magnitude = magnitude
            snr = mag_to_snr(psf_magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel,
                             l_pix=length, theta_deg=angle, snr_definition="detection")
            stack_snr = mag_to_snr(magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel, l_pix=length,
                                   theta_deg=angle, snr_definition="measurement")
        elif mag_mode == "integrated_mag":
            magnitude = rng.uniform(mag[0], mag[1])
            if inject_length > 0:
                psf_magnitude = magnitude + 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x))
                surface_brightness = magnitude + 2.5 * np.log10(length)
            else:
                psf_magnitude = magnitude
                surface_brightness = magnitude

            snr = mag_to_snr(psf_magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel,
                             l_pix=length, theta_deg=angle, snr_definition="detection")
            stack_snr = mag_to_snr(magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel, l_pix=length,
                                   theta_deg=angle, snr_definition="measurement")
        else:
            raise ValueError(f"Unknown mag_mode: {mag_mode}")
        injection_catalog.add_row([k, ra_pos, dec_pos, "Trail" if inject_length > 0 else "Star", inject_length, surface_brightness, angle, info.id,
                                       int(ref.dataId["detector"]), magnitude, psf_magnitude, snr, str(filter_name.bandLabel),
                                       x_pos, y_pos, stack_snr, m5_local, calexp.info.getSummaryStats().magLim])
    return injection_catalog

def stack_hits_by_footprints(
    post_src,
    calexp_pre,
    calexp_post,
    dimensions,
    truth_id_mask,
    injection_catalog,
    overlap_frac=0.02,
    overlap_minpix=100,
    return_matched_fp_masks = False
):
    H, W = int(dimensions.y), int(dimensions.x)
    N = len(injection_catalog)

    det_flag = np.zeros(N, bool)
    det_mag = np.full(N, np.nan)
    det_magerr = np.full(N, np.nan)
    det_snr = np.full(N, np.nan)
    if return_matched_fp_masks:
        matched_fp_masks = [np.zeros((H, W), bool) for _ in range(N)]

    mags = calexp_post.photoCalib.instFluxToMagnitude(post_src, "base_PsfFlux")

    ys, xs = np.nonzero(truth_id_mask)
    ids = truth_id_mask[ys, xs] - 1

    pix_by_id = [[] for _ in range(N)]
    for y, x, i in zip(ys, xs, ids):
        if 0 <= i < N:
            pix_by_id[i].append((y, x))

    for inj_id in range(N):
        if not pix_by_id[inj_id]:
            continue

        pts = pix_by_id[inj_id]
        yy = np.array([p[0] for p in pts])
        xx = np.array([p[1] for p in pts])

        y0, y1 = yy.min(), yy.max()
        x0, x1 = xx.min(), xx.max()
        truth_count = len(pts)

        th = np.zeros((y1 - y0 + 1, x1 - x0 + 1), bool)
        for y, x in pts:
            th[y - y0, x - x0] = True

        best_ov, best_idx, best_fp = 0, None, None

        for idx in range(len(post_src)):
            fp = post_src[idx].getFootprint()
            n_pix_footprint = fp.getArea()
            required = max(overlap_minpix, int(overlap_frac * n_pix_footprint))
            bb = fp.getBBox()
            if bb.getEndX() < x0 or bb.getBeginX() > x1 or bb.getEndY() < y0 or bb.getBeginY() > y1:
                continue

            fm = np.zeros_like(th)
            ov = 0
            for span in fp.spans:
                y = span.getY()
                if y < y0 or y > y1:
                    continue
                sx0 = max(span.getX0(), x0)
                sx1 = min(span.getX1(), x1)
                if sx0 <= sx1:
                    fm[y - y0, sx0 - x0 : sx1 - x0 + 1] = True
                    ov = int((fm & th).sum())
                    if ov >= required:
                        break

            if ov > best_ov:
                best_ov, best_idx, best_fp = ov, idx, fm
                if ov >= required:
                    break

        if best_idx is not None and best_ov >= required:
            det_flag[inj_id] = True
            det_mag[inj_id] = mags[best_idx, 0]
            det_magerr[inj_id] = mags[best_idx, 1]
            f = float(post_src[best_idx].get("base_PsfFlux_instFlux"))
            fe = float(post_src[best_idx].get("base_PsfFlux_instFluxErr"))
            det_snr[inj_id] = f / fe if (np.isfinite(f) and np.isfinite(fe) and fe > 0) else np.nan
            if return_matched_fp_masks:
                matched_fp_masks[inj_id][y0:y1+1, x0:x1+1] |= best_fp

    injection_catalog["stack_detection"] = det_flag
    injection_catalog["stack_mag"] = det_mag
    injection_catalog["stack_mag_err"] = det_magerr
    injection_catalog["stack_snr"] = det_snr
    if return_matched_fp_masks:
        return injection_catalog, matched_fp_masks
    else:
        return injection_catalog, None

def _catalog_radec_array(cat):
    rows = np.empty((len(cat), 2), dtype=np.float64)
    for idx, record in enumerate(cat):
        try:
            rows[idx, 0] = float(record["coord_ra"])
            rows[idx, 1] = float(record["coord_dec"])
        except Exception:
            coord = record.getCoord()
            rows[idx, 0] = coord.getRa().asRadians()
            rows[idx, 1] = coord.getDec().asRadians()
    return rows


def crossmatch_catalogs (pre, post):
    # Crossmatch POST vs PRE on-sky (inputs in radians, radius in radians)
    #    post-only detections -> "new" sources likely caused by injections
    if len(pre) > 0 and len(post) > 0:
        # arrays of [ra, dec] in radians
        P = _catalog_radec_array(post)
        R = _catalog_radec_array(pre)
        max_sep = np.deg2rad(0.40 / 3600.0)  # 0.40 arcsec -> radians
        dist, ind = crossmatch_angular(P, R, max_sep)
        is_new = np.isinf(dist)  # no match in PRE → likely injected
        new_post = post[is_new].copy()
    else:
        new_post = post.copy()
    return new_post

def footprints_to_label_mask(src_cat, dimensions, dtype=np.uint16):
    """
    Build integer label mask from source footprints.
    0 = background, (idx+1) = source idx in src_cat.
    """
    H, W = int(dimensions.y), int(dimensions.x)
    lab = np.zeros((H, W), dtype=dtype)

    for sid in range(len(src_cat)):
        fp = src_cat[sid].getFootprint()
        label = sid + 1  # 1..N

        # Fill footprint pixels using spans (fast, no per-pixel loops)
        for span in fp.spans:
            y = span.getY()
            if y < 0 or y >= H:
                continue
            x0 = max(span.getX0(), 0)
            x1 = min(span.getX1(), W - 1)
            if x0 <= x1:
                lab[y, x0:x1+1] = label  # overwrite is fine for ignore-mask use
    return lab

def build_forbidden_mask(calexp, pre_injection_src, dimensions):
    """
    Boolean mask of pixels where injections are NOT allowed.
    Combines:
      - calexp.mask planes (if present)
      - footprint pixels of detected sources (pre-injection)
    """
    H, W = int(dimensions.y), int(dimensions.x)
    forbid = np.zeros((H, W), dtype=bool)

    # --- 1) calexp.mask planes ---
    m = calexp.mask
    plane_dict = m.getMaskPlaneDict()

    # Use planes that typically mean "occupied" or should be avoided.
    # Keep this list conservative; you can edit it anytime.
    planes_to_avoid = [
        "DETECTED",
        "DETECTED_NEGATIVE",
        "SAT",
        "BAD",
        "CR",
        "NO_DATA",
        "EDGE",
    ]
    for p in planes_to_avoid:
        if p in plane_dict:
            bit = m.getPlaneBitMask(p)
            forbid |= (m.array & bit) != 0

    # --- 2) existing source footprints (pre-injection) ---
    # This is the most direct "do not overlap sources" mask.
    if pre_injection_src is not None and len(pre_injection_src) > 0:
        lab = footprints_to_label_mask(pre_injection_src, dimensions, dtype=np.uint16)
        forbid |= (lab > 0)

    return forbid


def _try_place_trail_no_overlap(
    rng,
    forbidden,
    dimensions,
    *,
    trail_length_px: float,
    angle_deg: float,
    half_margin: int,
    calexp,
    psf_template: str,
    max_tries: int = 2000,
):
    """
    Rejection-sample (x,y) uniformly, but accept only if the FULL stamped trail
    does not intersect forbidden pixels.

    Returns: (x, y, stamp_bool_mask)
    """
    H, W = int(dimensions.y), int(dimensions.x)

    # Temporary buffer reused per try
    tmp = np.zeros((H, W), dtype=np.uint8)

    for _ in range(max_tries):
        # Uniform over allowed bounding box
        x = float(rng.uniform(half_margin, W - 1 - half_margin))
        y = float(rng.uniform(half_margin, H - 1 - half_margin))

        # Thickness from local PSF width (same idea you already use later)
        use_kernel = (psf_template == "kernel")
        try:
            psf_width = int(calexp.psf.getLocalKernel(Point2D(x, y)).getWidth())
        except Exception:
            psf_width = 7  # safe-ish fallback
        thickness = max(1, int(psf_width // 2))

        # Build a candidate stamp mask for this proposed injection
        tmp.fill(0)
        draw_one_line(
            tmp,
            [x, y],
            angle_deg,
            trail_length_px,
            true_value=1,
            line_thickness=thickness,
        )
        stamp = (tmp != 0)

        # Reject if ANY stamped pixel intersects forbidden
        if (stamp & forbidden).any():
            continue

        # Accept
        return x, y, stamp

    raise RuntimeError(f"Could not place trail without overlap after {max_tries} tries")

def format_dataId(dataId):
    out_dataId = {"instrument": dataId["instrument"],
                  "detector": dataId["detector"],
                  "exposure": dataId["exposure"] if "exposure" in dataId else dataId["visit"],
                  "visit": dataId["exposure"] if "exposure" in dataId else dataId["visit"],
                  "band": dataId["band"]}
    return out_dataId

def dimensions_from_exposure(exposure):
    bbox = exposure.getBBox()
    return geom.Extent2I(int(bbox.getWidth()), int(bbox.getHeight()))

def one_detector_injection(n_inject, trail_length, mag, beta, repo, coll, dimensions, source_type, ref_dataId, seed=None, debug=False, mag_mode="psf_mag", psf_template="image", detection_threshold=5.0, reprocess=False):
    mem_log = []

    def log_mem(phase):
        mem_log.append((phase, current_rss_mb()))

    try:
        if seed is None:
            seed = np.random.randint(0,10000)
        if reprocess:
            raise NotImplementedError("reprocess=True is not supported in simulate_inject_fill_deterministic.py")
        butler = get_worker_butler(repo, coll)
        ref = butler.registry.findDataset(source_type, dataId=ref_dataId)
        formatted_dataId = format_dataId(ref.dataId)
        calexp, background = load_preliminary_from_butler_checked(butler, dataId=formatted_dataId)
        log_mem("loaded")
        pre_injection_fixed_src = source_detect(
            calexp,
            background,
            threshold=PREINJECTION_DETECTION_THRESHOLD,
        )
        log_mem("pre_fixed")
        if len(pre_injection_fixed_src) > MAX_PRE_SOURCES:
            raise RuntimeError(
                f"Too many pre-injection sources: {len(pre_injection_fixed_src)} > {MAX_PRE_SOURCES}"
            )
        if float(detection_threshold) == float(PREINJECTION_DETECTION_THRESHOLD):
            pre_injection_eval_src = pre_injection_fixed_src
        else:
            pre_injection_eval_src = source_detect(
                calexp,
                background,
                threshold=detection_threshold,
            )
            if len(pre_injection_eval_src) > MAX_PRE_SOURCES:
                raise RuntimeError(
                    f"Too many eval pre-injection sources: {len(pre_injection_eval_src)} > {MAX_PRE_SOURCES}"
                )
        log_mem("pre_eval")
        local_dimensions = dimensions_from_exposure(calexp)
        forbidden = build_forbidden_mask(calexp, pre_injection_fixed_src, local_dimensions)
        log_mem("forbidden")
        injection_catalog = generate_one_line(
            n_inject,
            trail_length,
            mag,
            beta,
            ref,
            local_dimensions,
            seed,
            calexp,
            mag_mode=mag_mode,
            psf_template=psf_template,
            forbidden_mask=forbidden,
        )
        del forbidden
        gc.collect()
        log_mem("generated")
        injected_calexp = inject(calexp, injection_catalog)
        log_mem("injected")
        if pre_injection_fixed_src is not pre_injection_eval_src:
            del pre_injection_fixed_src
            gc.collect()
            log_mem("dropped_pre_fixed")
        post_injection_Src = source_detect(injected_calexp, background, threshold=detection_threshold)
        log_mem("post_detect")
        mask = np.zeros((local_dimensions.y, local_dimensions.x), dtype=np.uint16)
        for i, row in enumerate(injection_catalog):
            psf_width = injected_calexp.psf.getLocalKernel(Point2D(row["x"], row["y"])).getWidth()
            mask = draw_one_line(mask, [row["x"], row["y"]], row["beta"], row["trail_length"], true_value=i + 1,
                                 line_thickness=int(psf_width/2))
        injection_catalog, matched_fp_mask = stack_hits_by_footprints(post_src=crossmatch_catalogs (pre_injection_eval_src, post_injection_Src),
                                                                       calexp_pre=calexp,
                                                                       calexp_post=injected_calexp,
                                                                       dimensions=local_dimensions,
                                                                       truth_id_mask=mask,
                                                                       injection_catalog=injection_catalog,
                                                                       overlap_minpix=1,
                                                                       overlap_frac=0.0,
                                                                       return_matched_fp_masks=debug)
        log_mem("stack_hits")

        real_labels = footprints_to_label_mask(pre_injection_eval_src, local_dimensions, dtype=np.uint16)
        log_mem("real_labels")
        img = injected_calexp.image.array.astype("float32")
        mask_out = mask.astype("bool")
        del post_injection_Src
        del pre_injection_eval_src
        if "pre_injection_fixed_src" in locals():
            del pre_injection_fixed_src
        del background
        del calexp
        gc.collect()
        log_mem("prepared_return")
        if not debug:
            return True, img, mask_out, real_labels, injection_catalog, mem_log
        else:
            det_mask = None
            m = injected_calexp.mask
            if "DETECTED" in m.getMaskPlaneDict():
                det_bit = m.getPlaneBitMask("DETECTED")
                det_mask = (m.array & det_bit) != 0
            det_neg_mask = None
            if "DETECTED_NEGATIVE" in m.getMaskPlaneDict():
                detn_bit = m.getPlaneBitMask("DETECTED_NEGATIVE")
                det_neg_mask = (m.array & detn_bit) != 0
            matched_fp_masks = np.any(np.stack(matched_fp_mask, axis=-1), axis=-1)
            return True, img, mask_out, real_labels, injection_catalog, det_mask, matched_fp_masks, mem_log
    except BaseException as e:
        tb = traceback.format_exc()
        if mem_log:
            rss_text = ", ".join(f"{phase}={rss:.1f}MB" for phase, rss in mem_log)
            tb = f"{tb}\n[rss]\n{rss_text}"
        return False, ref_dataId, repr(e), tb


def worker(args):
    rank, image_id, split_name, dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta, source_type, global_seed, mag_mode, psf_template, detection_threshold = args
    seed = (int(global_seed) * 1_000_003 + int(dataId["visit"]) * 1_003 + int(dataId["detector"])) & 0xFFFFFFFF
    try:
        t0 = time.perf_counter()
        previous_handler = signal.signal(signal.SIGALRM, _alarm_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, TASK_TIMEOUT_SECONDS)
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr_buffer):
                res = one_detector_injection(
                    number,
                    trail_length,
                    magnitude,
                    beta,
                    repo,
                    coll,
                    dims,
                    source_type,
                    dataId,
                    seed,
                    mag_mode=mag_mode,
                    psf_template=psf_template,
                    detection_threshold=detection_threshold,
                )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        t1 = time.perf_counter()
        if res[0] is False:
            tb = res[3]
            stderr_text = stderr_buffer.getvalue().strip()
            if stderr_text:
                tb = f"{tb}\n[stderr]\n{stderr_text}"
            return {
                "status": "err",
                "rank": rank,
                "image_id": image_id,
                "split": split_name,
                "dataId": res[1],
                "error": res[2],
                "traceback": tb,
                "compute_s": t1 - t0,
            }
        _, img, mask, real_labels, catalog, mem_log = res
        t2 = time.perf_counter()
        with lock:
            with h5py.File(h5path, "a") as f:
                f["images"][image_id] = img
                f["masks"][image_id] = mask
                f["real_labels"][image_id] = real_labels
            df = catalog.to_pandas()
            df["image_id"] = image_id
            file_exists = os.path.exists(csvpath)
            df.to_csv(csvpath, mode=("a" if file_exists else "w"), header=(not file_exists), index=False)
        t3 = time.perf_counter()
        return {
            "status": "ok",
            "rank": rank,
            "image_id": image_id,
            "split": split_name,
            "dataId": dataId,
            "compute_s": t1 - t0,
            "stage_s": t2 - t1,
            "promote_s": t3 - t2,
            "rss_max_mb": max((rss for _, rss in mem_log), default=0.0),
            "rss_log": ", ".join(f"{phase}={rss:.1f}MB" for phase, rss in mem_log),
        }

    except BaseException as e:
        tb = traceback.format_exc()
        stderr_text = stderr_buffer.getvalue().strip() if "stderr_buffer" in locals() else ""
        if stderr_text:
            tb = f"{tb}\n[stderr]\n{stderr_text}"
        return {
            "status": "err",
            "rank": rank,
            "image_id": image_id,
            "split": split_name,
            "dataId": dataId,
            "error": repr(e),
            "traceback": tb,
        }

def _key_from_dataId(d):
    return (int(d["visit"]), int(d["detector"]))


def get_allowed_detector_ids(butler, instrument: str, physical_type: str = "E2V") -> set[int]:
    camera = butler.get("camera", dataId={"instrument": instrument})
    allowed = set()
    for det in camera:
        if det.getType().name != "SCIENCE":
            continue
        if str(det.getPhysicalType()) == str(physical_type):
            allowed.add(int(det.getId()))
    return allowed


def select_candidate_refs_deterministic(
    *,
    repo: str,
    collections: str | Sequence[str],
    where: str,
    instrument: str = "LSSTCam",
    seed: int = 123,
    verbose: bool = False,
) -> List:
    """Return the full deterministic candidate order for preliminary_visit_image refs."""
    b = Butler(repo, collections=collections)
    allowed_detector_ids = get_allowed_detector_ids(b, instrument=instrument, physical_type="E2V")
    print(f"E2V detector filter: {len(allowed_detector_ids)} science detectors allowed", flush=True)

    refs_by_key = {}
    all_pvi_iter = b.registry.queryDatasets(
        "preliminary_visit_image",
        instrument=instrument,
        where=where,
        collections=collections,
        findFirst=True,
    )
    for ref in all_pvi_iter:
        key = _key_from_dataId(ref.dataId)
        if int(key[1]) not in allowed_detector_ids:
            continue
        refs_by_key.setdefault(key, ref)

    ordered_refs = [refs_by_key[key] for key in sorted(refs_by_key)]
    rng = random.Random(int(seed))
    rng.shuffle(ordered_refs)

    if verbose:
        print(f"Deterministic candidate refs available: {len(ordered_refs)}", flush=True)
    return ordered_refs


def compute_target_total(random_subset: int, n_candidates: int) -> int:
    if int(random_subset) > 0:
        return min(int(random_subset), int(n_candidates))
    return min(200, int(n_candidates))


def compute_split_targets(target_total: int, train_test_split: float, test_only: bool, seed: int):
    if 0 < train_test_split < 1:
        test_target = int((1 - train_test_split) * int(target_total))
        rng_split = np.random.default_rng(int(seed) + 1)
        test_ordinals = set(rng_split.choice(np.arange(int(target_total)), test_target, replace=False).tolist())
    else:
        test_target = 0
        test_ordinals = set()
    train_target = int(target_total) - int(test_target)
    if test_only:
        return 0, int(test_target), set(range(int(test_target)))
    return train_target, test_target, test_ordinals


def choose_output_dimensions(butler, refs, coll, seed, sample_size=5):
    sample_size = min(int(sample_size), len(refs))
    if sample_size <= 0:
        raise RuntimeError("No refs available to choose output dimensions.")

    rng = random.Random(int(seed) + 17)
    sample_indices = sorted(rng.sample(range(len(refs)), sample_size))
    counts = {}
    dims_by_key = {}
    for idx in sample_indices:
        dims = butler.get("preliminary_visit_image.dimensions", dataId=refs[idx].dataId)
        key = (int(dims.y), int(dims.x))
        counts[key] = counts.get(key, 0) + 1
        dims_by_key[key] = dims

    best_key = max(sorted(counts), key=lambda key: counts[key])
    return dims_by_key[best_key], sample_indices, counts


def init_output_file(path, n_rows, dims, chunks, compression=None):
    kwargs = {"chunks": chunks, "maxshape": (n_rows, dims.y, dims.x)}
    if compression is not None:
        kwargs.update(compression)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", shape=(n_rows, dims.y, dims.x), dtype="float32", **kwargs)
        f.create_dataset("masks", shape=(n_rows, dims.y, dims.x), dtype="bool", **kwargs)
        f.create_dataset("real_labels", shape=(n_rows, dims.y, dims.x), dtype="uint16", **kwargs)


def resize_output_file(path, n_rows):
    with h5py.File(path, "a") as f:
        for key in ("images", "masks", "real_labels"):
            f[key].resize((n_rows, f[key].shape[1], f[key].shape[2]))


def build_output_slots(target_total, test_ordinals, test_only, h5train_path, h5test_path, train_csv_path, test_csv_path):
    slots = []
    train_image_id = 0
    test_image_id = 0
    if test_only:
        for ordinal in range(int(target_total)):
            slots.append({
                "ordinal": ordinal,
                "split": "test",
                "image_id": test_image_id,
                "h5path": h5test_path,
                "csvpath": test_csv_path,
            })
            test_image_id += 1
    else:
        for ordinal in range(int(target_total)):
            if ordinal in test_ordinals:
                slots.append({
                    "ordinal": ordinal,
                    "split": "test",
                    "image_id": test_image_id,
                    "h5path": h5test_path,
                    "csvpath": test_csv_path,
                })
                test_image_id += 1
            else:
                slots.append({
                    "ordinal": ordinal,
                    "split": "train",
                    "image_id": train_image_id,
                    "h5path": h5train_path,
                    "csvpath": train_csv_path,
                })
                train_image_id += 1
    return slots

def run_parallel_injection(repo, coll, save_path, number, trail_length, magnitude, beta, where, parallel=4,
                           random_subset=0, train_test_split=0, seed=123, chunks=None, test_only=False, mag_mode="psf_mag",
                           psf_template="image", stack_detection_threshold=5.0):
    butler = Butler(repo, collections=coll)
    h5train_path = os.path.join(save_path, "train.h5")
    h5test_path = os.path.join(save_path, "test.h5")

    refs = select_candidate_refs_deterministic(
        repo=repo,
        collections=coll,
        where=where,
        instrument="LSSTCam",
        seed=seed,
    )
    print("Deterministic candidate refs:", len(refs), flush=True)
    if len(refs) == 0:
        raise RuntimeError("No candidate preliminary_visit_image refs found.")

    target_total = compute_target_total(random_subset, len(refs))
    train_target, test_target, test_ordinals = compute_split_targets(target_total, train_test_split, test_only, seed)
    final_target_total = test_target if test_only else target_total
    print(f"Target successful outputs: total={final_target_total} train={train_target} test={test_target}", flush=True)

    dims, dim_sample_indices, dim_counts = choose_output_dimensions(butler, refs, coll, seed, sample_size=5)
    print(
        f"Chosen output dimensions: ({dims.y}, {dims.x}) from sample_indices={dim_sample_indices} counts={dim_counts}",
        flush=True,
    )
    if chunks is not None:
        h5_chunks = (1, min(int(chunks), dims.y), min(int(chunks), dims.x))
    else:
        h5_chunks = (1, dims.y, dims.x)

    train_csv_path = os.path.join(save_path, "train.csv")
    test_csv_path = os.path.join(save_path, "test.csv")
    for path in (h5train_path, h5test_path, train_csv_path, test_csv_path):
        if os.path.exists(path):
            os.remove(path)

    if train_target > 0:
        init_output_file(h5train_path, train_target, dims, h5_chunks)
    if test_target > 0:
        init_output_file(
            h5test_path,
            test_target,
            dims,
            h5_chunks,
            compression={"compression": "gzip", "compression_opts": 4, "shuffle": True},
        )

    slots = build_output_slots(final_target_total, test_ordinals, test_only, h5train_path, h5test_path, train_csv_path, test_csv_path)
    success_count = 0
    attempts = 0
    timing = {"compute_s": 0.0, "stage_s": 0.0, "promote_s": 0.0, "successes": 0, "failures": 0, "rss_max_mb": 0.0}
    manager = Manager()
    lock = manager.Lock()
    next_candidate_for_slot = [slot_idx for slot_idx in range(len(slots))]
    next_slot_to_activate = 0
    filled_slots = set()

    def make_task(slot_idx):
        slot = slots[slot_idx]
        candidate_idx = next_candidate_for_slot[slot_idx]
        if candidate_idx >= len(refs):
            return None
        ref = refs[candidate_idx]
        next_candidate_for_slot[slot_idx] += len(slots)
        return [
            candidate_idx,
            slot["image_id"],
            slot["split"],
            ref.dataId,
            repo,
            coll,
            dims,
            lock,
            slot["h5path"],
            slot["csvpath"],
            number,
            trail_length,
            magnitude,
            beta,
            "preliminary_visit_image",
            seed,
            mag_mode,
            psf_template,
            stack_detection_threshold,
        ]

    def log_timing():
        success_denom = max(1, timing["successes"])
        attempt_denom = max(1, attempts)
        print(
            f"[timing] attempts={attempts} successes={timing['successes']} failures={timing['failures']} "
            f"avg_compute={timing['compute_s']/attempt_denom:.2f}s "
            f"avg_stage={timing['stage_s']/success_denom:.2f}s "
            f"avg_promote={timing['promote_s']/success_denom:.2f}s "
            f"peak_rss={timing['rss_max_mb']:.1f}MB",
            flush=True,
        )

    if parallel > 1:
        max_workers = max(1, int(parallel))
        max_inflight = max_workers
        future_to_slot = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            max_tasks_per_child=1,
        ) as ex:
            while next_slot_to_activate < len(slots) and len(future_to_slot) < max_inflight:
                task = make_task(next_slot_to_activate)
                if task is None:
                    next_slot_to_activate += 1
                    continue
                fut = ex.submit(worker, task)
                future_to_slot[fut] = next_slot_to_activate
                next_slot_to_activate += 1

            while future_to_slot:
                done, _ = concurrent.futures.wait(
                    future_to_slot.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    slot_idx = future_to_slot.pop(fut)
                    result = fut.result()
                    attempts += 1
                    timing["compute_s"] += float(result.get("compute_s", 0.0))
                    if ATTEMPT_DIAGNOSTICS:
                        print(f"[diag] {format_attempt_diagnostics(result)}", flush=True)
                    if result["status"] == "ok":
                        filled_slots.add(slot_idx)
                        success_count += 1
                        timing["successes"] += 1
                        timing["stage_s"] += float(result.get("stage_s", 0.0))
                        timing["promote_s"] += float(result.get("promote_s", 0.0))
                        timing["rss_max_mb"] = max(timing["rss_max_mb"], float(result.get("rss_max_mb", 0.0)))
                        print(f"[{success_count}/{final_target_total}] done", flush=True)
                    else:
                        timing["failures"] += 1
                        print(
                            f"[{success_count}/{final_target_total}] ERROR: rank={result['rank']} dataId={result['dataId']} error={result['error']}",
                            flush=True,
                        )
                        if result.get("traceback"):
                            print(result["traceback"], flush=True)
                        retry_task = make_task(slot_idx)
                        if retry_task is not None:
                            retry_fut = ex.submit(worker, retry_task)
                            future_to_slot[retry_fut] = slot_idx

                if attempts % 25 == 0:
                    log_timing()

                if success_count >= final_target_total:
                    for fut in future_to_slot:
                        fut.cancel()
                    break

                while (
                    next_slot_to_activate < len(slots)
                    and len(future_to_slot) < max_inflight
                ):
                    task = make_task(next_slot_to_activate)
                    if task is None:
                        next_slot_to_activate += 1
                        continue
                    fut = ex.submit(worker, task)
                    future_to_slot[fut] = next_slot_to_activate
                    next_slot_to_activate += 1
    else:
        while next_slot_to_activate < len(slots):
            slot_idx = next_slot_to_activate
            next_slot_to_activate += 1
            while True:
                task = make_task(slot_idx)
                if task is None:
                    break
                result = worker(task)
                attempts += 1
                timing["compute_s"] += float(result.get("compute_s", 0.0))
                if ATTEMPT_DIAGNOSTICS:
                    print(f"[diag] {format_attempt_diagnostics(result)}", flush=True)
                if result["status"] == "ok":
                    filled_slots.add(slot_idx)
                    success_count += 1
                    timing["successes"] += 1
                    timing["stage_s"] += float(result.get("stage_s", 0.0))
                    timing["promote_s"] += float(result.get("promote_s", 0.0))
                    timing["rss_max_mb"] = max(timing["rss_max_mb"], float(result.get("rss_max_mb", 0.0)))
                    print(f"[{success_count}/{final_target_total}] done", flush=True)
                    break
                timing["failures"] += 1
                print(
                    f"[{success_count}/{final_target_total}] ERROR: rank={result['rank']} dataId={result['dataId']} error={result['error']}",
                    flush=True,
                )
                if result.get("traceback"):
                    print(result["traceback"], flush=True)
            if attempts % 25 == 0 and attempts > 0:
                log_timing()
            if success_count >= final_target_total:
                break

    if success_count < final_target_total:
        print(
            f"Candidate pool exhausted before reaching target: successes={success_count}/{final_target_total}, "
            f"attempts={attempts}, remaining_shortfall={final_target_total - success_count}",
            flush=True,
        )
    if attempts > 0:
        success_denom = max(1, timing["successes"])
        attempt_denom = max(1, attempts)
        print(
            f"[timing-final] attempts={attempts} successes={timing['successes']} failures={timing['failures']} "
            f"avg_compute={timing['compute_s']/attempt_denom:.2f}s "
            f"avg_stage={timing['stage_s']/success_denom:.2f}s "
            f"avg_promote={timing['promote_s']/success_denom:.2f}s "
            f"peak_rss={timing['rss_max_mb']:.1f}MB",
            flush=True,
        )

def rng_for_task(seed: int, dataId: dict) -> np.random.Generator:
    # stable across runs
    s = (int(seed) * 1_000_003
         + int(dataId["visit"]) * 1_003
         + int(dataId["detector"])) & 0xFFFFFFFF
    return np.random.default_rng(s)

def main():
    ap = argparse.ArgumentParser("Build a SIMULATED (injected) dataset")
    ap.add_argument("--repo", type=str, default="dp2_prep")
    ap.add_argument("--collections", type=str, default="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2")
    ap.add_argument("--save-path", default="../DATA/")
    ap.add_argument("--where",
                    default="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') ")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--random-subset", type=int, default=10)
    ap.add_argument("--train-test-split", type=float, default=0.1)
    ap.add_argument("--trail-length-min", type=float, default=6)
    ap.add_argument("--trail-length-max", type=float, default=60)
    ap.add_argument("--mag-min", type=float, default=19)
    ap.add_argument("--mag-max", type=float, default=24)
    ap.add_argument("--mag-mode", choices=["psf_mag", "snr", "surface_brightness", "integrated_mag"], default="psf_mag")
    ap.add_argument("--psf-template", choices=["image", "kernel"], default="kernel",
                    help="PSF template source: image=computeImage; kernel=computeKernelImage (if available)")
    ap.add_argument("--beta-min", type=float, default=0)
    ap.add_argument("--beta-max", type=float, default=180)
    ap.add_argument("--number", type=int, default=20)
    ap.add_argument("--stack-detection-threshold", type=float, default=5.0, help="SNR threshold for the stack (default 5.0)")
    ap.add_argument("--seed", type=int, default=13379)
    ap.add_argument("--chunks", type=int, default=None, help="HDF5 chunk size (square). Example: 128 -> chunks=(1,128,128). None = contiguous")
    ap.add_argument("--test-only", action="store_true", default=False, help="Only generate test set")
    args = ap.parse_args()

    ensure_dir(args.save_path)
    logger = logging.getLogger("lsst")
    logger.setLevel(logging.ERROR)
    # call into your pasted function
    run_parallel_injection(
        repo=args.repo,
        coll=args.collections,
        save_path=args.save_path,
        number=args.number,
        trail_length=[args.trail_length_min, args.trail_length_max],
        magnitude=[args.mag_min, args.mag_max],
        mag_mode=args.mag_mode,
        beta=[args.beta_min, args.beta_max],
        parallel=args.parallel,
        where=args.where,
        random_subset=args.random_subset,
        train_test_split=args.train_test_split,
        chunks=args.chunks,
        test_only=args.test_only,
        seed=args.seed,
        psf_template=args.psf_template,
        stack_detection_threshold=args.stack_detection_threshold,
    )

if __name__ == "__main__":
    main()
