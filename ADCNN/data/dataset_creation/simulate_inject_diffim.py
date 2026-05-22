"""Diffim-based variant of simulate_inject.py.

Same outputs (HDF5 with images / masks / real_labels + a per-injection CSV),
same drawn-line truth, same crossmatch-and-recover pre/post logic. The only
substantive change is the *image*: instead of saving the injected calexp,
we save the difference image produced by subtracting the matching
template_coadd from the injected PVI.

Flow per (visit, detector):
    1. fetch PVI + single_visit_star_footprints (kernel candidates) + same-band
       overlapping template_coadd
    2. AlardLupton subtract on the CLEAN PVI -> clean diffim
    3. DetectAndMeasure on the clean diffim -> "pre_injection_Src" (real
       residuals: variable stars, dipoles, kernel mismatches — all
       non-asteroid stuff)
    4. forbidden mask = PVI mask planes + clean-diffim source footprints
    5. generate injection catalog and inject into a CLONE of the PVI
    6. AlardLupton subtract on the INJECTED PVI with the SAME `sources`
       (same kernel-candidate set => same kernel solve) -> injected diffim
    7. DetectAndMeasure on the injected diffim -> "post_injection_Src"
    8. drawn-line truth mask, crossmatch pre/post, stack_hits_by_footprints,
       footprints_to_label_mask -> identical to direct-image flow
    9. write injected diffim to HDF5 (1 channel float32) + masks + real_labels

The template fetch requires the stage3 collection to be in the chain along
with stage2.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import logging
import os
import random
import traceback
import warnings
from multiprocessing import Lock, Manager, Value
from pathlib import Path
from typing import Any, List, Sequence


# ======================================================================================
# Silence known-harmless noise at module-import time so it applies to BOTH the
# master process and every forked worker (workers don't run main()).
#
# All four classes of message below were inspected against the running 850-
# pair pilot; none indicate a real failure. The actual worker failures
# surface as `[N/850] ERROR` lines from worker()'s own try/except.
# ======================================================================================
def _silence_known_noise() -> None:
    # 1. DetectAndMeasureTask runs DipoleFitPlugin on every diffim detection.
    #    Most aren't real dipoles; "DipoleFitPlugin failed on record N: bad
    #    dipole fit" is `self.log.warning(...)` from
    #    lsst/ip/diffim/dipoleFitTask.py:1271. The catch: measBase plugins
    #    use `logging.getLogger(plugin_name)` where plugin_name is e.g.
    #    "ip_diffim_DipoleFit" (registered at @measBase.register(...) on
    #    line 1001 of dipoleFitTask.py). That logger is TOP-LEVEL, not
    #    under lsst.*, so silencing only `lsst` doesn't reach it.
    for name in (
        "lsst",
        "lsst.ip.diffim",
        "lsst.detectAndMeasure",
        "lsst.meas.algorithms",
        "ip_diffim_DipoleFit",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Safety net: disable WARNING and below process-wide. Doesn't affect our
    # own print()s or worker traceback dumps (those don't use logging).
    # ERROR / CRITICAL still pass.
    logging.disable(logging.WARNING)

    # 2. lsst.meas.algorithms.maskStreaks emits RuntimeWarnings on degenerate
    #    streak chi^2 inputs.
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning,
        module=r"lsst\.meas\.algorithms\.maskStreaks",
    )
    # 3. astropy unit conversion divide-by-zero edge in flux<->mag.
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning,
        module=r"astropy\.units\.quantity",
    )
    # 4. sklearn complains when R^2 is computed on <2 samples deep in the stack.
    try:
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    except Exception:
        pass


_silence_known_noise()


import h5py
import numpy as np
import pandas as pd
from astroML.crossmatch import crossmatch_angular
from astropy.table import Table

import lsst.geom as geom
from lsst.daf.butler import Butler
from lsst.geom import Point2D
from lsst.pipe.base import NoWorkFound, UnprocessableDataError, UpstreamFailureNoWorkFound
from lsst.source.injection.inject_exposure import ExposureInjectTask

# Stack-side control-flow exceptions that subclass BaseException (NOT Exception)
# in lsst.pipe.base._status. They signal "this quantum has no work / can't be
# processed" rather than a code bug — for our use case we want to log and
# skip the pair, not crash the whole 850-pair run. The four-pair pilot lost
# pair 198/850 to a NoWorkFound("Insufficient Template Coverage. (8.2% < 10%)")
# until this was added.
_SKIP_EXCEPTIONS = (
    Exception,
    NoWorkFound,
    UnprocessableDataError,
    UpstreamFailureNoWorkFound,
)

from common import (
    draw_one_line,
    ensure_dir,
    mag_to_snr,
    psf_fwhm_arcsec_from_calexp,
    snr_to_mag,
)
from pipetasks import (
    catalog_to_pandas,
    fetch_diffim_inputs,
    run_detect_diffim,
    run_subtract,
)


completed_counter = Value('i', 0)
counter_lock = Lock()


# ======================================================================================
# Injection (identical semantics to simulate_inject.py)
# ======================================================================================

def inject(pvi_clone, injection_catalog):
    """ExposureInjectTask onto a CLONE of the PVI. Returns the injected
    Exposure. The input catalog is mutated by the task — snapshot any cols
    you need before calling.

    If the env var ADCNN_REALISTIC_TRAIL=1 is set (via --realistic-trail), the
    stock uniform-Box trail renderer is replaced once per worker process with the
    light-curve / tapered / curved renderer in ADCNN.data.dataset_creation.
    realistic_trail (physical priors only; leakage-free)."""
    if os.environ.get("ADCNN_REALISTIC_TRAIL") == "1":
        try:
            from ADCNN.data.dataset_creation import realistic_trail
        except ImportError:
            import realistic_trail  # script runs from its own dir; ADCNN may not be on path
        realistic_trail.install(verbose=False)
    inject_task = ExposureInjectTask()
    inject_res = inject_task.run(
        [injection_catalog],
        pvi_clone,
        pvi_clone.psf,
        pvi_clone.photoCalib,
        pvi_clone.wcs,
    )
    return inject_res.output_exposure


def estimate_m5_local_from_psf_var(calexp, x, y, snr=5.0):
    """Local m5 at (x,y); identical to simulate_inject.py."""
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


def generate_one_line(n_inject, trail_length, mag, beta, ref, dimensions, seed,
                     calexp, mag_mode="psf_mag", psf_template="image",
                     forbidden_mask=None):
    """Identical to simulate_inject.py.generate_one_line — the science
    exposure used for PSF/photometry calls is the PVI here (which is the
    same kind of object as the calexp it replaced).

    Note: SNR/m5 estimates use the science-frame variance, so columns like
    SNR_estimation and m5_local are PVI-frame, NOT diffim-frame. The diffim
    pixel noise is roughly sqrt(science_var + matched_template_var),
    somewhat larger. Treat these as approximate.
    """
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
        forbidden = forbidden_mask.astype(bool, copy=True)
    raw = calexp.wcs
    info = calexp.visitInfo
    filter_name = calexp.filter
    m5 = {"u": 23.7, "g": 24.97, "r": 24.52, "i": 24.13, "z": 23.56, "y": 22.55}
    psf_depth = m5[filter_name.bandLabel]
    a, b = 0.42, 0
    for k in range(n_inject):
        inject_length = rng.uniform(*trail_length)
        if inject_length <= 0:
            length = 1.0
        else:
            length = inject_length
        R = 30
        S = int(np.ceil(length)) + 2 * R + 1
        half = S // 2 + 2
        angle = rng.uniform(*beta)
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
        forbidden |= stamp

        m5_local = estimate_m5_local_from_psf_var(calexp, x_pos, y_pos)
        sky_pos = raw.pixelToSky(x_pos, y_pos)
        ra_pos = sky_pos.getRa().asDegrees()
        dec_pos = sky_pos.getDec().asDegrees()
        use_kernel = (psf_template == "kernel")
        fwhm_arcsec = psf_fwhm_arcsec_from_calexp(calexp, x_pos, y_pos, use_kernel_image=use_kernel)
        if not np.isfinite(fwhm_arcsec) or fwhm_arcsec <= 0:
            fwhm_arcsec_table = {"u": 0.92, "g": 0.87, "r": 0.83, "i": 0.80, "z": 0.78, "y": 0.76}
            fwhm_arcsec = fwhm_arcsec_table[filter_name.bandLabel]
        pixelScale = raw.getPixelScale().asArcseconds()
        theta_p = fwhm_arcsec / pixelScale
        x = length / theta_p
        upper_limit_mag = psf_depth - 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x)) if mag[1] == 0 else mag[1]
        if mag_mode == "snr":
            snr_edge = 5.0
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


# ======================================================================================
# Truth / labels / crossmatch (verbatim copies — same semantics, operating on
# diffim source catalogs whose API matches SourceCatalog from
# SingleFrameDetectAndMeasureTask)
# ======================================================================================

def stack_hits_by_footprints(
    post_src,
    calexp_pre,
    calexp_post,
    dimensions,
    truth_id_mask,
    injection_catalog,
    overlap_frac=0.02,
    overlap_minpix=100,
    return_matched_fp_masks=False,
):
    H, W = int(dimensions.y), int(dimensions.x)
    N = len(injection_catalog)

    det_flag = np.zeros(N, bool)
    det_mag = np.full(N, np.nan)
    det_magerr = np.full(N, np.nan)
    det_snr = np.full(N, np.nan)
    if return_matched_fp_masks:
        matched_fp_masks = [np.zeros((H, W), bool) for _ in range(N)]

    # Use the science (PVI) photoCalib for flux→mag — diffim Exposures don't
    # carry a usable photoCalib in this stack.
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
                    fm[y - y0, sx0 - x0: sx1 - x0 + 1] = True
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
                matched_fp_masks[inj_id][y0:y1 + 1, x0:x1 + 1] |= best_fp

    injection_catalog["stack_detection"] = det_flag
    injection_catalog["stack_mag"] = det_mag
    injection_catalog["stack_mag_err"] = det_magerr
    injection_catalog["stack_snr"] = det_snr
    if return_matched_fp_masks:
        return injection_catalog, matched_fp_masks
    else:
        return injection_catalog, None


def crossmatch_catalogs(pre, post):
    """Same on-sky crossmatch as the direct-image pipeline. Operates on
    diffim source catalogs (DetectAndMeasureTask.diaSources). Matches on
    coord_ra/coord_dec at 0.4 arcsec radius; returns post sources that
    have NO match in pre (i.e. likely caused by the injection)."""
    if len(pre) > 0 and len(post) > 0:
        P = post.asAstropy().to_pandas()[["coord_ra", "coord_dec"]].values
        R = pre.asAstropy().to_pandas()[["coord_ra", "coord_dec"]].values
        max_sep = np.deg2rad(0.40 / 3600.0)
        dist, ind = crossmatch_angular(P, R, max_sep)
        is_new = np.isinf(dist)
        new_post = post[is_new].copy()
    else:
        new_post = post.copy()
    return new_post


def footprints_to_label_mask(src_cat, dimensions, dtype=np.uint16):
    """0 = background, (idx+1) = source idx in src_cat."""
    H, W = int(dimensions.y), int(dimensions.x)
    lab = np.zeros((H, W), dtype=dtype)

    for sid in range(len(src_cat)):
        fp = src_cat[sid].getFootprint()
        label = sid + 1
        for span in fp.spans:
            y = span.getY()
            if y < 0 or y >= H:
                continue
            x0 = max(span.getX0(), 0)
            x1 = min(span.getX1(), W - 1)
            if x0 <= x1:
                lab[y, x0:x1 + 1] = label
    return lab


def build_forbidden_mask(calexp, pre_injection_src, dimensions):
    """PVI mask planes ∪ pre-injection source footprints. For diffim mode,
    pre_injection_src is the clean-diffim source catalog (residuals of real
    non-asteroid stuff). The mask planes come from the PVI we'll inject
    into."""
    H, W = int(dimensions.y), int(dimensions.x)
    forbid = np.zeros((H, W), dtype=bool)

    m = calexp.mask
    plane_dict = m.getMaskPlaneDict()
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

    if pre_injection_src is not None and len(pre_injection_src) > 0:
        lab = footprints_to_label_mask(pre_injection_src, dimensions, dtype=np.uint16)
        forbid |= (lab > 0)

    return forbid


def _try_place_trail_no_overlap(rng, forbidden, dimensions, *, trail_length_px,
                                angle_deg, half_margin, calexp, psf_template,
                                max_tries=2000):
    H, W = int(dimensions.y), int(dimensions.x)
    tmp = np.zeros((H, W), dtype=np.uint8)

    for _ in range(max_tries):
        x = float(rng.uniform(half_margin, W - 1 - half_margin))
        y = float(rng.uniform(half_margin, H - 1 - half_margin))

        try:
            psf_width = int(calexp.psf.getLocalKernel(Point2D(x, y)).getWidth())
        except Exception:
            psf_width = 7
        thickness = max(1, int(psf_width // 2))

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

        if (stamp & forbidden).any():
            continue

        return x, y, stamp

    raise RuntimeError(f"Could not place trail without overlap after {max_tries} tries")


def format_dataId(dataId):
    out_dataId = {"instrument": dataId["instrument"],
                  "detector": dataId["detector"],
                  "exposure": dataId["exposure"] if "exposure" in dataId else dataId["visit"],
                  "visit": dataId["exposure"] if "exposure" in dataId else dataId["visit"],
                  "band": dataId["band"]}
    return out_dataId


# ======================================================================================
# Per-detector diffim injection (the meat — replaces simulate_inject.one_detector_injection)
# ======================================================================================

def one_detector_injection(n_inject, trail_length, mag, beta, repo, coll, dimensions,
                           source_type, ref_dataId, skymap, stage3_collection,
                           seed=None, debug=False, mag_mode="psf_mag",
                           psf_template="image", detection_threshold=5.0,
                           measueTrails=False):
    try:
        if seed is None:
            seed = np.random.randint(0, 10000)
        butler = Butler(repo, collections=coll)
        ref = butler.registry.findDataset(source_type, dataId=ref_dataId)

        # 1. Inputs: PVI, kernel-candidate sources, template.
        pvi, sources, template, physical_filter, _n_tmpl = fetch_diffim_inputs(
            butler,
            format_dataId(ref.dataId),
            skymap=skymap,
            stage3_collection=stage3_collection,
        )

        # 2. Clean diffim subtraction.
        sub_clean = run_subtract(template=template, science=pvi, sources=sources)
        diffim_clean = sub_clean.difference

        # 3. Pre-injection sources = real residuals on the clean diffim.
        det_clean = run_detect_diffim(
            science=pvi,
            matchedTemplate=sub_clean.matchedTemplate,
            difference=diffim_clean,
            threshold=detection_threshold,
            measueTrails=measueTrails,
        )
        pre_injection_Src = det_clean.diaSources

        # 4. Forbidden mask from PVI mask + clean-diffim residual footprints.
        forbidden = build_forbidden_mask(pvi, pre_injection_Src, dimensions)

        # 5. Generate injection catalog using PVI photometry/PSF/WCS.
        injection_catalog = generate_one_line(
            n_inject, trail_length, mag, beta, ref, dimensions, seed, pvi,
            mag_mode=mag_mode, psf_template=psf_template,
            forbidden_mask=forbidden,
        )

        # n_inject==0: real-empty-background mode. Skip the inject/re-subtract/
        # re-detect path (ExposureInjectTask rejects empty catalogs); the
        # "injected" diffim IS the clean diffim, truth mask is all zeros, and
        # the panel still carries real_labels from pre-injection residuals.
        if n_inject == 0:
            diffim_inj = diffim_clean
            mask = np.zeros((dimensions.y, dimensions.x), dtype=np.uint16)
            real_labels = footprints_to_label_mask(pre_injection_Src, dimensions, dtype=np.uint16)
            matched_fp_mask = None
        else:
            # 6. Inject into a CLONE so the clean PVI stays around for later
            # (and so the same `sources` give the same kernel-candidate set on
            # both subtractions).
            pvi_injected = inject(pvi.clone(), injection_catalog)

            # 7. Injected diffim subtraction with the SAME sources.
            sub_inj = run_subtract(template=template, science=pvi_injected, sources=sources)
            diffim_inj = sub_inj.difference

            # 8. Post-injection sources on the injected diffim.
            det_inj = run_detect_diffim(
                science=pvi_injected,
                matchedTemplate=sub_inj.matchedTemplate,
                difference=diffim_inj,
                threshold=detection_threshold,
                measueTrails=measueTrails,
            )
            post_injection_Src = det_inj.diaSources

            # 9. Drawn-line truth (identical to direct-image flow).
            mask = np.zeros((dimensions.y, dimensions.x), dtype=np.uint16)
            for i, row in enumerate(injection_catalog):
                psf_width = pvi_injected.psf.getLocalKernel(Point2D(row["x"], row["y"])).getWidth()
                mask = draw_one_line(
                    mask, [row["x"], row["y"]], row["beta"], row["trail_length"],
                    true_value=i + 1, line_thickness=int(psf_width / 2),
                )

            # 10. Crossmatch pre vs post; mark recovered injections by footprint
            # overlap with drawn truth.
            injection_catalog, matched_fp_mask = stack_hits_by_footprints(
                post_src=crossmatch_catalogs(pre_injection_Src, post_injection_Src),
                calexp_pre=pvi,
                calexp_post=pvi_injected,
                dimensions=dimensions,
                truth_id_mask=mask,
                injection_catalog=injection_catalog,
                overlap_minpix=1,
                overlap_frac=0.0,
                return_matched_fp_masks=debug,
            )

            # real_labels = footprints of pre-injection diffim residuals.
            real_labels = footprints_to_label_mask(pre_injection_Src, dimensions, dtype=np.uint16)

        # 11. Image written to HDF5 = the injected DIFFIM (1-channel float32).
        if not debug:
            return True, diffim_inj.image.array.astype("float32"), mask.astype("bool"), real_labels, injection_catalog
        else:
            det_mask = None
            mplanes = diffim_inj.mask.getMaskPlaneDict()
            if "DETECTED" in mplanes:
                det_bit = diffim_inj.mask.getPlaneBitMask("DETECTED")
                det_mask = (diffim_inj.mask.array & det_bit) != 0
            det_neg_mask = None
            if "DETECTED_NEGATIVE" in mplanes:
                detn_bit = diffim_inj.mask.getPlaneBitMask("DETECTED_NEGATIVE")
                det_neg_mask = (diffim_inj.mask.array & detn_bit) != 0
            matched_fp_masks = (
                np.any(np.stack(matched_fp_mask, axis=-1), axis=-1)
                if matched_fp_mask is not None else None
            )
            return True, diffim_inj.image.array.astype("float32"), mask.astype("bool"), real_labels, injection_catalog, det_mask, matched_fp_masks
    except _SKIP_EXCEPTIONS as e:
        return False, ref_dataId, repr(e), traceback.format_exc()


# ======================================================================================
# Worker / pool — same shape as simulate_inject.py
# ======================================================================================

def worker(args):
    (idx, dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length,
     magnitude, beta, source_type, global_seed, mag_mode, psf_template,
     detection_threshold, measueTrails, skymap, stage3_collection) = args
    seed = (int(global_seed) * 1_000_003 + int(dataId["visit"]) * 1_003 + int(dataId["detector"])) & 0xFFFFFFFF
    try:
        res = one_detector_injection(
            number, trail_length, magnitude, beta, repo, coll, dims, source_type,
            dataId, skymap=skymap, stage3_collection=stage3_collection, seed=seed,
            mag_mode=mag_mode, psf_template=psf_template,
            detection_threshold=detection_threshold, measueTrails=measueTrails,
        )
        if res[0] is False:
            return ("err", res[1], res[2], res[3])
        _, img, mask, real_labels, catalog = res
        with lock:
            with h5py.File(h5path, "a") as f:
                f["images"][idx] = img
                f["masks"][idx] = mask
                if "real_labels" in f:
                    f["real_labels"][idx] = real_labels

            df = catalog_to_pandas(catalog, measueTrails=measueTrails)
            df["image_id"] = idx
            file_exists = os.path.exists(csvpath)
            df.to_csv(csvpath, mode=("a" if file_exists else "w"),
                      header=(not file_exists), index=False)
        return ("ok", idx)

    except _SKIP_EXCEPTIONS:
        tb = traceback.format_exc()
        return ("err", idx, dataId, tb)


def _key_from_dataId(d):
    return (int(d["visit"]), int(d["detector"]))


def reservoir_sample(iterable, k: int, seed: int = 123):
    rng = random.Random(int(seed))
    sample = []
    for i, item in enumerate(iterable, 1):
        if i <= k:
            sample.append(item)
        else:
            j = rng.randrange(i)
            if j < k:
                sample[j] = item
    return sample


# ======================================================================================
# Pair selection: PVI + same-band overlapping template_coadd
# ======================================================================================

def select_good_refs_random_check(
    *,
    repo: str,
    collections: str | Sequence[str],
    where: str,
    skymap: str,
    stage3_collection: str | None,
    instrument: str = "LSSTCam",
    k: int = 200,
    seed: int = 123,
    pool_size: int = 5000,
    max_checks: int = 200000,
    check_refs: bool = True,
    exclude_keys: set | None = None,
    verbose: bool = False,
) -> List:
    """Like the direct-image variant but with one extra validation step:
    a same-band template_coadd must overlap the PVI region.

    Stack quirks observed in build_manifest.py:
      - `band=:band` bind on template_coadd queries silently no-ops; filter
        Python-side via `r.dataId.get("band") == band`.
      - Use `patch.region OVERLAPS :region`, NOT `template_coadd.region`.
      - The PVI's `ref.dataId.required` does NOT include `band` (it's an
        implied dimension); read it from `expandDataId(...).get("band")`.
    """
    b = Butler(repo, collections=collections)

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
        refs_by_key.setdefault(key, ref)

    if exclude_keys:
        n_before = len(refs_by_key)
        refs_by_key = {k: v for k, v in refs_by_key.items() if k not in exclude_keys}
        if verbose:
            print(f"Excluded {n_before - len(refs_by_key)} (visit,detector) pairs "
                  f"present in exclude_keys (e.g. test sets); {len(refs_by_key)} remain",
                  flush=True)
    ordered_refs = [refs_by_key[key] for key in sorted(refs_by_key)]
    rng = random.Random(int(seed))
    rng.shuffle(ordered_refs)

    if not check_refs:
        out = sorted(ordered_refs[:int(k)], key=lambda r: _key_from_dataId(r.dataId))
        if verbose:
            print(f"Selected refs without checks: {len(out)} (requested k={k})", flush=True)
        return out

    initial_pool = min(len(ordered_refs), max(int(k), int(pool_size)))
    dims_x = []
    dims_y = []
    small_n = min(100, initial_pool)
    for i in range(small_n):
        try:
            dims = b.get(
                "preliminary_visit_image.dimensions",
                dataId=ordered_refs[i].dataId,
                collections=collections,
            )
            dims_x.append(int(dims.x))
            dims_y.append(int(dims.y))
        except Exception:
            continue

    dim_x = None
    dim_y = None
    if len(dims_x) > 0:
        dim_x = np.bincount(np.array(dims_x, dtype=int)).argmax()
        dim_y = np.bincount(np.array(dims_y, dtype=int)).argmax()

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n_workers = max(1, int(slurm_cpus) - 1)
        except ValueError:
            n_workers = max(1, (os.cpu_count() or 1) - 1)
    else:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    template_collections = (
        [stage3_collection] if isinstance(stage3_collection, str) else list(stage3_collection)
    ) if stage3_collection is not None else None

    def _validate_ref(ref):
        local_b = Butler(repo, collections=collections)
        try:
            svsf = local_b.registry.findDataset(
                "single_visit_star_footprints",
                dataId=ref.dataId,
                collections=collections,
            )
            if svsf is None:
                return False

            try:
                pc = local_b.get("preliminary_visit_image.photoCalib", dataId=ref.dataId, collections=collections)
            except Exception:
                return False
            if pc is None:
                return False

            if dim_x is not None and dim_y is not None:
                try:
                    dims_local = local_b.get(
                        "preliminary_visit_image.dimensions",
                        dataId=ref.dataId,
                        collections=collections,
                    )
                    if int(dims_local.x) != int(dim_x) or int(dims_local.y) != int(dim_y):
                        return False
                except Exception:
                    return False

            # NEW: same-band template_coadd overlap check.
            try:
                expanded = local_b.registry.expandDataId(ref.dataId)
                region = expanded.region
                band = expanded.get("band")
            except Exception:
                return False
            if region is None or band is None:
                return False

            t_query_kwargs = dict(
                where="skymap = :skymap AND patch.region OVERLAPS :region",
                bind={"skymap": skymap, "region": region},
                findFirst=True,
            )
            if template_collections is not None:
                t_query_kwargs["collections"] = template_collections
            all_t = list(local_b.registry.queryDatasets("template_coadd", **t_query_kwargs))
            if not [r for r in all_t if r.dataId.get("band") == band]:
                return False

            return True
        except Exception:
            return False

    good = []
    checks = 0
    next_start = 0
    refill_size = max(1, int(pool_size))
    batch_size = max(1, min(256, refill_size))

    while len(good) < int(k) and checks < int(max_checks) and next_start < len(ordered_refs):
        pool_end = min(len(ordered_refs), next_start + (initial_pool if next_start == 0 else refill_size))
        pool = ordered_refs[next_start:pool_end]
        next_start = pool_end

        for batch_start in range(0, len(pool), batch_size):
            if len(good) >= int(k) or checks >= int(max_checks):
                break
            batch = pool[batch_start:batch_start + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_workers, len(batch))) as ex:
                results = list(ex.map(_validate_ref, batch))

            for ref, ok in zip(batch, results):
                if len(good) >= int(k) or checks >= int(max_checks):
                    break
                checks += 1
                if ok:
                    good.append(ref)

    good = sorted(good, key=lambda r: _key_from_dataId(r.dataId))

    if verbose:
        print(
            f"Selected good refs: {len(good)} (requested k={k}), "
            f"from pool_size={pool_size}, checks={checks}",
            flush=True,
        )

    return good


def run_parallel_injection(repo, coll, save_path, number, trail_length, magnitude, beta, where,
                           skymap, stage3_collection, parallel=4, random_subset=0,
                           train_test_split=0, seed=123, chunks=None, test_only=False,
                           mag_mode="psf_mag", psf_template="image",
                           stack_detection_threshold=5.0, measueTrails=False,
                           exclude_keys=None, check_refs=True):
    butler = Butler(repo, collections=coll)
    h5train_path = os.path.join(save_path, "train.h5")
    h5test_path = os.path.join(save_path, "test.h5")

    refs = select_good_refs_random_check(
        repo=repo,
        collections=coll,
        where=where,
        skymap=skymap,
        stage3_collection=stage3_collection,
        instrument="LSSTCam",
        k=int(random_subset) if int(random_subset) > 0 else 200,
        seed=seed,
        pool_size=5000,
        max_checks=200000,
        exclude_keys=exclude_keys,
        check_refs=check_refs,
        verbose=True,
    )
    print("Selected datasets:", len(refs))

    global total_tasks
    total_tasks = len(refs)
    rng_split = np.random.default_rng(seed + 1)
    test_index = rng_split.choice(np.arange(len(refs)), int((1 - train_test_split) * len(refs)),
                                  replace=False) if 0 < train_test_split < 1 else []
    if test_only:
        total_tasks = len(test_index)
    dims = butler.get("preliminary_visit_image.dimensions", dataId=refs[0].dataId)
    if chunks is not None:
        chunks = (1, min(int(chunks), dims.y), min(int(chunks), dims.x))
    if not test_only:
        with h5py.File(h5train_path, "w") as f:
            f.create_dataset("images", shape=(len(refs) - len(test_index), dims.y, dims.x), dtype="float32", chunks=chunks)
            f.create_dataset("masks", shape=(len(refs) - len(test_index), dims.y, dims.x), dtype="bool", chunks=chunks)
            f.create_dataset("real_labels", shape=(len(refs) - len(test_index), dims.y, dims.x), dtype="uint16", chunks=chunks)
    if len(test_index) > 0:
        with h5py.File(h5test_path, "w") as f:
            f.create_dataset(
                "images",
                shape=(len(test_index), dims.y, dims.x),
                dtype="float32",
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            f.create_dataset(
                "masks",
                shape=(len(test_index), dims.y, dims.x),
                dtype="bool",
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            f.create_dataset(
                "real_labels",
                shape=(len(test_index), dims.y, dims.x),
                dtype="uint16",
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
    manager = Manager()
    lock = manager.Lock()
    count_train = 0
    count_test = 0
    tasks = []
    for i, ref in enumerate(refs):
        if i in test_index:
            h5path = h5test_path
            csvpath = os.path.join(save_path, "test.csv")
            count = count_test
            count_test += 1
            tasks.append([count, ref.dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta,
                          "preliminary_visit_image", seed, mag_mode, psf_template, stack_detection_threshold, measueTrails,
                          skymap, stage3_collection])
        elif not test_only:
            h5path = h5train_path
            csvpath = os.path.join(save_path, "train.csv")
            count = count_train
            count_train += 1
            tasks.append([count, ref.dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta,
                          "preliminary_visit_image", seed, mag_mode, psf_template, stack_detection_threshold, measueTrails,
                          skymap, stage3_collection])
    if parallel > 1:
        completed = 0
        total_tasks = len(tasks)

        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(worker, t) for t in tasks]

            for fut in concurrent.futures.as_completed(futs):
                completed += 1
                # Catch BaseException here so a worker that raised something
                # like NoWorkFound (BaseException-only subclass) before our
                # try/except in worker() can run can't kill the whole job.
                try:
                    out = fut.result()
                except BaseException as e:
                    print(f"[{completed}/{total_tasks}] WORKER CRASH: {type(e).__name__}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
                    continue

                if out[0] == "ok":
                    print(f"[{completed}/{total_tasks}] done", flush=True)
                else:
                    _, idx, dataId, tb = out
                    print(f"[{completed}/{total_tasks}] ERROR: idx={idx} dataId={dataId}", flush=True)
                    print(tb, flush=True)
    else:
        for task in tasks:
            worker(task)


def rng_for_task(seed: int, dataId: dict) -> np.random.Generator:
    s = (int(seed) * 1_000_003
         + int(dataId["visit"]) * 1_003
         + int(dataId["detector"])) & 0xFFFFFFFF
    return np.random.default_rng(s)


# ======================================================================================
# CLI
# ======================================================================================

def main():
    ap = argparse.ArgumentParser("Build a SIMULATED (injected) DIFFIM dataset")
    ap.add_argument("--repo", type=str, default="dp2_prep")
    # NOTE: needs both stage3 (template_coadd) and stage2 (PVI / sources).
    ap.add_argument(
        "--collections", nargs="+",
        default=[
            "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3",
            "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
        ],
        help="Butler collection chain. Must include the stage3 collection "
             "carrying template_coadd and the stage2 collection carrying "
             "preliminary_visit_image / single_visit_star_footprints.",
    )
    ap.add_argument(
        "--stage3-collection",
        default="LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3",
        help="Subset of --collections used to query template_coadd "
             "(passed through to GetTemplateTask).",
    )
    ap.add_argument("--skymap", default="lsst_cells_v2")
    ap.add_argument("--save-path", default="../DATA/")
    ap.add_argument("--where",
                    default="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') ")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--random-subset", type=int, default=10)
    ap.add_argument("--train-test-split", type=float, default=0.1)
    ap.add_argument("--trail-length-min", type=float, default=6)
    ap.add_argument("--trail-length-max", type=float, default=60)
    ap.add_argument("--mag-min", type=float, default=22.5)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--mag-mode", choices=["psf_mag", "snr", "surface_brightness", "integrated_mag"], default="psf_mag")
    ap.add_argument("--psf-template", choices=["image", "kernel"], default="kernel")
    ap.add_argument("--beta-min", type=float, default=0)
    ap.add_argument("--beta-max", type=float, default=180)
    ap.add_argument("--number", type=int, default=20)
    ap.add_argument("--stack-detection-threshold", type=float, default=5.0)
    ap.add_argument("--measueTrails", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--test-only", action="store_true", default=False)
    ap.add_argument("--skip-prevalidation", action="store_true", default=False,
                    help="Skip the slow per-pair template/source pre-validation. "
                         "Generation skips failed pairs anyway (writes no CSV rows), and "
                         "CSV-driven training excludes the resulting empty h5 slots -- so "
                         "this is empty-tensor-safe and much faster to start. Oversample "
                         "--random-subset slightly to offset ~5-15%% skipped pairs.")
    ap.add_argument("--exclude-pairs-csv", nargs="*", default=None,
                    help="CSV file(s) with visit,detector columns whose pairs must "
                         "NOT be selected for injection (leakage guard against test "
                         "sets). e.g. the test_5sigma and test_real catalogs.")
    ap.add_argument("--realistic-trail", action="store_true", default=False,
                    help="Render trails with the realistic (light-curve/tapered/"
                         "curved) renderer instead of the uniform galsim.Box. "
                         "Leakage-free: physical priors only.")
    args = ap.parse_args()

    if args.realistic_trail:
        os.environ["ADCNN_REALISTIC_TRAIL"] = "1"
        print("[main] realistic trail renderer ENABLED", flush=True)

    ensure_dir(args.save_path)
    logger = logging.getLogger("lsst")
    logger.setLevel(logging.ERROR)

    # leakage guard: never inject into a (visit,detector) that is in a test set.
    exclude_keys = None
    if args.exclude_pairs_csv:
        ek = set()
        for p in args.exclude_pairs_csv:
            df = pd.read_csv(p)
            ek |= {(int(v), int(d)) for v, d in zip(df["visit"], df["detector"])}
        exclude_keys = ek
        print(f"[main] excluding {len(exclude_keys)} (visit,detector) pairs from "
              f"{len(args.exclude_pairs_csv)} csv(s)", flush=True)

    coll = args.collections if len(args.collections) > 1 else args.collections[0]

    run_parallel_injection(
        repo=args.repo,
        coll=coll,
        save_path=args.save_path,
        number=args.number,
        trail_length=[args.trail_length_min, args.trail_length_max],
        magnitude=[args.mag_min, args.mag_max],
        mag_mode=args.mag_mode,
        beta=[args.beta_min, args.beta_max],
        parallel=args.parallel,
        where=args.where,
        skymap=args.skymap,
        stage3_collection=args.stage3_collection,
        random_subset=args.random_subset,
        train_test_split=args.train_test_split,
        chunks=args.chunks,
        test_only=args.test_only,
        seed=args.seed,
        psf_template=args.psf_template,
        stack_detection_threshold=args.stack_detection_threshold,
        measueTrails=args.measueTrails,
        exclude_keys=exclude_keys,
        check_refs=not args.skip_prevalidation,
    )


if __name__ == "__main__":
    main()
