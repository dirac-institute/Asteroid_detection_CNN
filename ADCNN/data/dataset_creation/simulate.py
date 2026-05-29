"""Build the SIMULATED (injected-trail) difference-image datasets.

ONE deterministic entry point produces every set used by the two-stage detector:
    train  (+ val)   stage-1 segmentation training  (val -> model selection)
    train2 (+ val2)  stage-2 cutout-CNN training     (val2 -> FP-filter threshold)
    test             held-out evaluation
A single panel universe is selected for the --where region, partitioned ONCE into these
five mutually-disjoint sets (seeded shuffle -> contiguous slices), and cached in
``<save-path>/split.json``. Each requested set (``--sets train train2 test``) is then injected
with the SAME validated per-panel core. Determinism: a fixed ``--seed`` fixes the panel
selection, the partition, AND each panel's injections (seeded by seed,visit,detector), so every
rerun selects the same panels and injects identical trails. A panel that fails to subtract is
simply dropped (failures are repeatable). ``split.json`` is reused across runs so partial builds
(e.g. ``--sets test``) stay consistent with a full build.

Each set writes ``<save-path>/<set>.h5`` (images / masks / real_labels) + ``<set>.csv``
(per-injection truth). The saved *image* is the difference image produced by subtracting the
matching template_coadd from the injected PVI.

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
       footprints_to_label_mask -> per-injection truth + stack-detection labels
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
import math
from multiprocessing import Lock, Manager, Value
from types import SimpleNamespace
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

# InsufficientKernelSourcesError is raised by AlardLuptonSubtractTask when a panel has too few PSF-
# matching kernel sources (config minKernelSources=3 — not enough good isolated stars to solve the AL
# kernel). It must be caught IN-PROCESS: it is an lsst.pipe.base AlgorithmError whose __init__ takes
# keyword-only required args, so Python's default exception pickling (which reconstructs via positional
# args) raises TypeError on unpickle. If it escaped a worker, the parent could not unpickle it and the
# whole ProcessPool would die with BrokenProcessPool. Catching it here makes it a clean panel skip.
try:
    from lsst.ip.diffim.subtractImages import InsufficientKernelSourcesError as _InsufficientKernelSrc
    _EXTRA_SKIP_EXCEPTIONS = (_InsufficientKernelSrc,)
except Exception:
    _EXTRA_SKIP_EXCEPTIONS = ()

# LSST-stack control-flow exceptions that legitimately mean "skip this panel" (template-coverage
# shortfall, too-few kernel sources, no work for this quantum). NOT a place for bare Exception: that
# would hide genuine bugs behind a silently-short dataset. Unexpected errors are handled by worker()'s
# safety net, which surfaces them as visible errors without killing the pool.
_SKIP_EXCEPTIONS = (
    NoWorkFound,
    UnprocessableDataError,
    UpstreamFailureNoWorkFound,
) + _EXTRA_SKIP_EXCEPTIONS

from ADCNN.utils.helpers import draw_one_line
from ADCNN.data.dataset_creation.photometry import (
    ensure_dir,
    mag_to_snr,
    psf_fwhm_arcsec_from_calexp,
    snr_to_mag,
)
from ADCNN.data.dataset_creation.butler_tasks import (
    catalog_to_pandas,
    fetch_diffim_inputs,
    run_detect_diffim,
    run_subtract,
)


completed_counter = Value('i', 0)
counter_lock = Lock()


# ======================================================================================
# Injection
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
    """Local m5 at (x,y)."""
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
    """Generate the per-detector injection catalog. The science exposure used
    for PSF/photometry calls is the PVI.

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
# Per-detector diffim injection (the meat)
# ======================================================================================

def one_detector_injection(n_inject, trail_length, mag, beta, repo, coll, dimensions,
                           source_type, ref_dataId, skymap, stage3_collection,
                           seed=None, debug=False, mag_mode="psf_mag",
                           psf_template="image", detection_threshold=5.0,
                           measure_trails=False, stack_detection_thresholds=None,
                           injection_detection_threshold=None):
    """Inject + difference a single detector, then label it at the stack-detection threshold(s).

    The expensive work — the clean and injected PSF-matching subtractions and the trail injection —
    runs exactly once per panel; only the DIA detection + footprint labelling varies with sigma, so
    that is the only step repeated. ``stack_detection_thresholds`` selects the return shape:

      - ``None``  -> single-sigma: returns ``(True, image, mask, real_labels, catalog)`` (or a
                     7-tuple when ``debug``).
      - a list    -> multi-sigma: the injection placement is fixed at one reference sigma so the
                     IMAGE is identical across the sweep, and detection is re-run per sigma. Returns
                     ``(True, image, mask, {sigma: (real_labels, catalog)})``.
    """
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

        # Build at THIS panel's own pixel dimensions (LSSTCam mixes ITL 4072x4000 and e2v 4096x4004
        # CCDs); the h5 writer later pads every panel to the common max frame so both geometries are
        # kept. (The passed-in `dimensions` is only the h5 target; injection/labels use the real size.)
        _bb = pvi.getBBox()
        dimensions = SimpleNamespace(x=int(_bb.getWidth()), y=int(_bb.getHeight()))

        # 2. Clean diffim subtraction (ONCE; shared by every requested sigma).
        sub_clean = run_subtract(template=template, science=pvi, sources=sources)
        diffim_clean = sub_clean.difference

        single = stack_detection_thresholds is None
        sigmas = [float(detection_threshold)] if single else [float(s) for s in stack_detection_thresholds]
        # Injection placement / forbidden mask are fixed at ONE reference sigma so the injected image
        # is identical across the sweep (default = the single sigma, or the deepest for a multi build).
        inj_thr = (float(injection_detection_threshold) if injection_detection_threshold is not None
                   else (float(detection_threshold) if single else max(sigmas)))

        # Clean-diffim detections, cached by sigma: the forbidden mask and each sigma's real-residual
        # labels come from these, so a sigma shared between them is detected only once.
        _clean: dict = {}
        def detect_clean(thr):
            thr = float(thr)
            if thr not in _clean:
                _clean[thr] = run_detect_diffim(
                    science=pvi, matchedTemplate=sub_clean.matchedTemplate,
                    difference=diffim_clean, threshold=thr, measure_trails=measure_trails,
                ).diaSources
            return _clean[thr]

        # 3-5. Forbidden mask (reference sigma) -> injection catalog (placement identical for all sigmas).
        forbidden = build_forbidden_mask(pvi, detect_clean(inj_thr), dimensions)
        injection_catalog = generate_one_line(
            n_inject, trail_length, mag, beta, ref, dimensions, seed, pvi,
            mag_mode=mag_mode, psf_template=psf_template, forbidden_mask=forbidden,
        )

        # 6-9. n_inject==0: real-empty-background mode. Skip the inject/re-subtract path
        # (ExposureInjectTask rejects empty catalogs); the "injected" diffim IS the clean diffim and
        # the truth mask is all zeros (the panel still carries real_labels from residuals).
        if n_inject == 0:
            diffim_inj = diffim_clean
            pvi_injected = None
            mask = np.zeros((dimensions.y, dimensions.x), dtype=np.uint16)
            def detect_inj(thr):
                return None
        else:
            # Inject into a CLONE (clean PVI kept; same `sources` -> same kernel candidates on both
            # subtractions). Inject + re-subtract + draw the truth mask ONCE.
            pvi_injected = inject(pvi.clone(), injection_catalog)
            sub_inj = run_subtract(template=template, science=pvi_injected, sources=sources)
            diffim_inj = sub_inj.difference
            mask = np.zeros((dimensions.y, dimensions.x), dtype=np.uint16)
            for i, row in enumerate(injection_catalog):
                psf_width = pvi_injected.psf.getLocalKernel(Point2D(row["x"], row["y"])).getWidth()
                mask = draw_one_line(
                    mask, [row["x"], row["y"]], row["beta"], row["trail_length"],
                    true_value=i + 1, line_thickness=int(psf_width / 2),
                )
            _inj: dict = {}
            def detect_inj(thr):
                thr = float(thr)
                if thr not in _inj:
                    _inj[thr] = run_detect_diffim(
                        science=pvi_injected, matchedTemplate=sub_inj.matchedTemplate,
                        difference=diffim_inj, threshold=thr, measure_trails=measure_trails,
                    ).diaSources
                return _inj[thr]

        # 10-11. Per-sigma labelling: stack-detection (post-injection footprint overlap with the drawn
        # truth) + real-residual labels (pre-injection footprints). The catalog is copied per sigma so
        # each carries its own stack_detection columns.
        per_sigma = {}
        for s in sigmas:
            pre_s = detect_clean(s)
            real_labels_s = footprints_to_label_mask(pre_s, dimensions, dtype=np.uint16)
            if n_inject == 0:
                per_sigma[s] = (real_labels_s, injection_catalog, None)
                continue
            cat_s = injection_catalog.copy()
            cat_s, matched_fp_mask = stack_hits_by_footprints(
                post_src=crossmatch_catalogs(pre_s, detect_inj(s)),
                calexp_pre=pvi, calexp_post=pvi_injected, dimensions=dimensions,
                truth_id_mask=mask, injection_catalog=cat_s,
                overlap_minpix=1, overlap_frac=0.0,
                return_matched_fp_masks=debug and single,
            )
            per_sigma[s] = (real_labels_s, cat_s, matched_fp_mask)

        img = diffim_inj.image.array.astype("float32")
        mask_b = mask.astype("bool")
        if not single:
            return True, img, mask_b, {s: (rl, cat) for s, (rl, cat, _) in per_sigma.items()}

        # single-sigma: 5-tuple, or 7-tuple when debug is set.
        real_labels, catalog, matched_fp_mask = per_sigma[sigmas[0]]
        if not debug:
            return True, img, mask_b, real_labels, catalog
        det_mask = None
        mplanes = diffim_inj.mask.getMaskPlaneDict()
        if "DETECTED" in mplanes:
            det_bit = diffim_inj.mask.getPlaneBitMask("DETECTED")
            det_mask = (diffim_inj.mask.array & det_bit) != 0
        matched_fp_masks = (
            np.any(np.stack(matched_fp_mask, axis=-1), axis=-1)
            if matched_fp_mask is not None else None
        )
        return True, img, mask_b, real_labels, catalog, det_mask, matched_fp_masks
    except _SKIP_EXCEPTIONS as e:
        return False, ref_dataId, repr(e), traceback.format_exc()


# ======================================================================================
# Worker / pool
# ======================================================================================

def _pad(a, shape):
    """Place a 2-D panel array top-left in a zero/false `shape`=(H,W) frame (pad), or crop if larger.
    LSSTCam mixes ITL (4072x4000) and e2v (4096x4004) CCDs; padding every panel to the common max
    frame keeps BOTH geometries in one fixed-size h5 instead of dropping the minority vendor. The
    injected trails live in the real top-left region, so pixel coordinates are unchanged."""
    H, W = int(shape[0]), int(shape[1])
    if a.shape == (H, W):
        return a
    out = np.zeros((H, W), dtype=a.dtype)
    h, w = min(a.shape[0], H), min(a.shape[1], W)
    out[:h, :w] = a[:h, :w]
    return out


def _sigma_tag(s):
    """5.0 -> '5', 4.5 -> '4.5'  (suffix for per-sigma h5 datasets / CSV columns)."""
    s = float(s)
    return str(int(s)) if s.is_integer() else str(s)


_STACK_COLS = ("stack_detection", "stack_mag", "stack_mag_err", "stack_snr")


def worker(args):
    (counters, dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length,
     magnitude, beta, source_type, global_seed, mag_mode, psf_template,
     detection_threshold, measure_trails, skymap, stage3_collection) = args
    # A list in the threshold slot selects the multi-sigma build (one panel, labelled at each sigma).
    multi = isinstance(detection_threshold, (list, tuple))
    seed = (int(global_seed) * 1_000_003 + int(dataId["visit"]) * 1_003 + int(dataId["detector"])) & 0xFFFFFFFF
    try:
        res = one_detector_injection(
            number, trail_length, magnitude, beta, repo, coll, dims, source_type,
            dataId, skymap=skymap, stage3_collection=stage3_collection, seed=seed,
            mag_mode=mag_mode, psf_template=psf_template, measure_trails=measure_trails,
            detection_threshold=(5.0 if multi else detection_threshold),
            stack_detection_thresholds=(detection_threshold if multi else None),
        )
        if res[0] is False:
            return ("err", res[1], res[2], res[3])
        # Index assigned under the lock, only on success -> successful panels land contiguously
        # (0..n-1) with NO gaps; the h5 is truncated to the final count after the pool drains. A
        # panel that fails to subtract is simply dropped (its failure is repeatable, so the set of
        # panels + their injections stays deterministic across reruns).
        with lock:
            idx = int(counters[h5path]); counters[h5path] = idx + 1
            if not multi:
                _, img, mask, real_labels, catalog = res
                with h5py.File(h5path, "a") as f:
                    tgt = f["images"].shape[1:]                 # pad this panel to the common h5 frame
                    f["images"][idx] = _pad(img, tgt)
                    f["masks"][idx] = _pad(mask, tgt)
                    if "real_labels" in f:
                        f["real_labels"][idx] = _pad(real_labels, tgt)
                df = catalog_to_pandas(catalog, measure_trails=measure_trails)
            else:
                _, img, mask, per_sigma = res
                tags = sorted(per_sigma, reverse=True)             # e.g. [5.0, 4.0, 3.0]
                with h5py.File(h5path, "a") as f:
                    tgt = f["images"].shape[1:]                    # pad this panel to the common h5 frame
                    f["images"][idx] = _pad(img, tgt)
                    f["masks"][idx] = _pad(mask, tgt)
                    # plain real_labels = deepest sigma (= injection-reference = training sigma) -> the
                    # segmentation model's DIA-mask input channel, matching how it was trained.
                    f["real_labels"][idx] = _pad(per_sigma[tags[0]][0], tgt)
                    for s in tags:
                        f[f"real_labels_{_sigma_tag(s)}sigma"][idx] = _pad(per_sigma[s][0], tgt)
                # ONE row-block: shared truth columns once + the per-sigma stack_* columns suffixed.
                df = catalog_to_pandas(per_sigma[tags[0]][1], measure_trails=measure_trails)
                df = df.drop(columns=[c for c in _STACK_COLS if c in df.columns])
                for s in tags:
                    cs = catalog_to_pandas(per_sigma[s][1], measure_trails=measure_trails)
                    for c in _STACK_COLS:
                        if c in cs.columns:
                            df[f"{c}_{_sigma_tag(s)}sigma"] = cs[c].values
            df["image_id"] = idx
            file_exists = os.path.exists(csvpath)
            df.to_csv(csvpath, mode=("a" if file_exists else "w"),
                      header=(not file_exists), index=False)
        return ("ok", idx)

    except _SKIP_EXCEPTIONS:
        tb = traceback.format_exc()
        return ("err", -1, dataId, tb)
    except Exception as e:
        # SAFETY NET: never let an exception cross the ProcessPool boundary as a pickled object. Several
        # LSST AlgorithmError/RepeatableQuantumError subclasses have keyword-only __init__ and FAIL to
        # unpickle in the parent, which silently kills the whole pool (BrokenProcessPool). Re-raise as a
        # plain (picklable) RuntimeError carrying the type+message: the build then prints + counts it as
        # a visible "worker error" (a flood still trips the per-set >=50% guard), and the pool survives.
        raise RuntimeError(f"{type(e).__module__}.{type(e).__name__}: {str(e)[:300]}") from None


def _key_from_dataId(d):
    return (int(d["visit"]), int(d["detector"]))


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
    filter_dims: bool = True,
    exclude_keys: set | None = None,
    min_ecliptic_lat: float = 0.0,
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

    # Ecliptic-latitude cut: keep only panels whose VISIT boresight is far from the ecliptic, so
    # the diffim background carries no real moving objects (the dense main belt sits within ~±20°).
    # Combined with `exclude_keys` (the real-asteroid catalog) this gives a contamination-clean
    # synthetic set AND keeps the real test panels disjoint for later evaluation.
    if min_ecliptic_lat and float(min_ecliptic_lat) > 0:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from lsst.sphgeom import LonLat
        vlonlat = {}
        for r in b.registry.queryDimensionRecords("visit", instrument=instrument, where=where):
            c = LonLat(r.region.getBoundingCircle().getCenter())
            vlonlat[int(r.id)] = (c.getLon().asDegrees(), c.getLat().asDegrees())
        if vlonlat:
            vids = list(vlonlat)
            ras = np.array([vlonlat[v][0] for v in vids])
            decs = np.array([vlonlat[v][1] for v in vids])
            eclat = SkyCoord(ras * u.deg, decs * u.deg, frame="icrs").barycentrictrueecliptic.lat.deg
            allowed = {vids[i] for i in range(len(vids)) if abs(float(eclat[i])) > float(min_ecliptic_lat)}
            n_before = len(refs_by_key)
            refs_by_key = {kk: vv for kk, vv in refs_by_key.items() if kk[0] in allowed}
            if verbose:
                print(f"Ecliptic cut |lat|>{min_ecliptic_lat} deg: kept {len(refs_by_key)}/{n_before} panels "
                      f"({len(allowed)}/{len(vids)} visits are off-ecliptic)", flush=True)

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

            if filter_dims and dim_x is not None and dim_y is not None:
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


# ======================================================================================
# Dataset orchestration: select a panel universe ONCE, partition it into disjoint named
# sets, and inject each requested set with the SAME validated per-panel core (`worker` ->
# `one_detector_injection`). Deterministic: a fixed --seed fixes the universe selection, the
# partition, AND each panel's injections (seeded by seed,visit,detector), so reruns reproduce
# byte-identical datasets. The partition is cached in split.json so partial builds (e.g. only
# the test set) stay consistent with a full build.
# ======================================================================================

# train and train2 each carry a held-out val set for evaluation:
#   val  -> stage-1 segmentation-model selection ;  val2 -> stage-2 CNN threshold.
# The fixed order makes the partition a deterministic sequence of contiguous slices.
_SET_ORDER = ("train", "val", "train2", "val2", "test")
_GROUPS = {"train": ("train", "val"), "train2": ("train2", "val2"), "test": ("test",)}


def _partition(keys, sizes, seed):
    """Deterministic, disjoint partition of the (visit,detector) `keys` into the named sets:
    one seeded shuffle, then contiguous slices -> no panel can land in two sets."""
    rng = np.random.default_rng(int(seed))
    shuffled = [keys[i] for i in rng.permutation(len(keys))]
    parts, i = {}, 0
    for name in _SET_ORDER:
        n = int(sizes.get(name, 0))
        parts[name] = shuffled[i:i + n]
        i += n
    return parts


def _build_set(dataids, *, name, repo, coll, dims, save_path, number, trail_length, magnitude,
               beta, mag_mode, psf_template, measure_trails, seed, skymap, stage3_collection,
               parallel, chunks, stack_detection_threshold=None,
               stack_detection_thresholds=None, compress=False, target=None):
    """Inject + difference + truth-label every panel in `dataids` into <save_path>/<name>.{h5,csv}
    via the per-panel `worker`/`one_detector_injection`. The h5 is created RESIZABLE and truncated to
    the count actually built, so a panel that fails to subtract leaves NO empty slot (its failure is
    repeatable, so the panel set + injections stay deterministic).

    `stack_detection_thresholds` (a list) labels each panel — injected once — at every sigma, writing
    ONE h5 (shared images/masks + `real_labels_<sigma>sigma` per sigma) and ONE csv (shared truth
    columns + `stack_detection_<sigma>sigma` per sigma). `stack_detection_threshold` (a scalar) is the
    single-sigma path.

    `compress` gzips the h5 (read-only eval sets — test — like the original test pipeline: the diffims
    are mostly background so they pack ~10x, keeping the multi-sigma test set tiny; the big TRAIN sets
    stay uncompressed for fast training-loop reads).

    `target` (build-to-target cap): `dataids` is the over-allocated slice (target x headroom); the build
    STOPS once `target` panels are successfully built (cancelling the rest), so skip-prevalidation panel
    failures don't leave the set short AND the on-disk size is bounded (no overshoot). In-flight workers
    at the cap may add up to ~`parallel` extra panels (harmless)."""
    multi = stack_detection_thresholds is not None
    thr_arg = [float(s) for s in stack_detection_thresholds] if multi else stack_detection_threshold
    # Single-sigma: one `real_labels`. Multi-sigma: a plain `real_labels` (= the deepest/injection-
    # reference sigma) that the segmentation model consumes as its DIA-mask input channel exactly as in
    # training, PLUS a per-sigma `real_labels_<s>sigma` for the notebook's stack-FP-per-panel counts.
    label_dsets = ([("real_labels", "uint16")]
                   + ([(f"real_labels_{_sigma_tag(s)}sigma", "uint16") for s in thr_arg] if multi else []))
    h5_path = os.path.join(save_path, f"{name}.h5")
    csv_path = os.path.join(save_path, f"{name}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    ch = (1, min(int(chunks), dims.y), min(int(chunks), dims.x)) if chunks else None
    ds_kw = {}
    if compress:
        if ch is None:                       # gzip requires chunking
            ch = (1, min(512, dims.y), min(512, dims.x))
        ds_kw = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(h5_path, "w") as f:
        mx = (None, dims.y, dims.x)
        for nm, dt in ([("images", "float32"), ("masks", "bool")] + label_dsets):
            f.create_dataset(nm, shape=(len(dataids), dims.y, dims.x), maxshape=mx, dtype=dt, chunks=ch, **ds_kw)
    manager = Manager()
    lock = manager.Lock()
    counters = manager.dict()
    counters[h5_path] = 0
    tasks = [[counters, did, repo, coll, dims, lock, h5_path, csv_path, number, trail_length,
              magnitude, beta, "preliminary_visit_image", seed, mag_mode, psf_template,
              thr_arg, measure_trails, skymap, stage3_collection]
             for did in dataids]
    print(f"[build:{name}] {len(tasks)} panels -> {h5_path}"
          + (f" (sigmas={thr_arg})" if multi else ""), flush=True)
    err = 0
    if parallel > 1:
        # Plain process pool. worker() never lets an exception escape unpickled (it returns a clean
        # skip or re-raises as a picklable RuntimeError — see worker()), so a bad panel is just counted
        # in `err` and dropped; it cannot break the pool. The only thing that still can is genuine
        # infrastructure trouble (node OOM / eviction): for that we retry the leftover panels a few
        # times in a fresh pool, then leave the remainder to SLURM --requeue.
        from concurrent.futures.process import BrokenProcessPool
        pending = list(tasks)
        capped = False
        for attempt in range(1, 4):
            if not pending or capped:
                break
            batch, pending = pending, []
            try:
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=parallel, max_tasks_per_child=40) as ex:
                    futmap = {ex.submit(worker, t): t for t in batch}
                    left = set(futmap)
                    for fut in concurrent.futures.as_completed(futmap):
                        left.discard(fut)
                        try:
                            if fut.result()[0] != "ok":
                                err += 1
                        except BrokenProcessPool:
                            pending.append(futmap[fut])
                        except BaseException as e:
                            err += 1
                            print(f"[{name}] worker error: {type(e).__name__}: {e}", flush=True)
                        if target and int(counters[h5_path]) >= int(target):
                            capped = True
                            for f in left:
                                f.cancel()
                            break
                        done = len(batch) - len(left)
                        if done % 50 == 0:
                            print(f"[{name} {done}/{len(batch)}] built={int(counters[h5_path])} "
                                  f"err={err}", flush=True)
                    if not capped:
                        pending.extend(futmap[f] for f in left)
            except BrokenProcessPool:
                pass
            if pending and not capped and attempt < 3:
                print(f"[{name}] pool broke (infrastructure); retrying {len(pending)} panels "
                      f"(attempt {attempt + 1}/3)", flush=True)
        if pending and not capped:
            err += len(pending)
            print(f"[{name}] DROPPED {len(pending)} panels after retries — SLURM --requeue will rerun", flush=True)
    else:
        for t in tasks:
            worker(t)

    # Truncate to the number actually built -> no empty tail slots from panels that failed.
    n = int(counters[h5_path])
    with h5py.File(h5_path, "a") as f:
        for ds in ["images", "masks"] + [nm for nm, _ in label_dsets]:
            if f[ds].shape[0] != n:
                f[ds].resize(n, axis=0)
    print(f"[build:{name}] DONE built={n}/{len(dataids)} err={err} -> {h5_path} (+ {csv_path})", flush=True)
    return n


def gather_shards(save_path, name, cleanup=True):
    """Concatenate the per-shard files a sharded build produced — <name>.shard*.{h5,csv} — into the
    single <name>.{h5,csv} that the trainer/eval read, renumbering image_id to 0..N-1. This is the
    recombine step for a set whose build was spread across a SLURM array (see make_datasets_fleet.sh);
    train is read directly from its shards via --data-sources, so only the small sets are gathered.
    Generic over datasets (single-sigma real_labels and multi-sigma real_labels_<s>sigma alike).

    Memory- and disk-safe: copies in small panel-blocks (bounded RAM) and, when cleanup is set, deletes
    each shard the moment it has been consumed, so the growing output and the not-yet-consumed shards
    never both need to fit at once — the gather succeeds even with little free space."""
    import glob
    import re as _re
    shards = sorted(glob.glob(os.path.join(save_path, f"{name}.shard*.h5")),
                    key=lambda p: int(_re.search(r"shard(\d+)\.h5$", p).group(1)))
    if not shards:
        raise SystemExit(f"[gather] no {name}.shard*.h5 under {save_path}")
    with h5py.File(shards[0], "r") as f0:
        dsets = [(k, f0[k].dtype) for k in f0.keys()]
        Y, X = int(f0["images"].shape[1]), int(f0["images"].shape[2])
        compressed = f0["images"].compression is not None
    counts = []
    for s in shards:
        with h5py.File(s, "r") as f:
            counts.append(int(f["images"].shape[0]))
    total = sum(counts)
    ds_kw = (dict(compression="gzip", compression_opts=4, shuffle=True,
                  chunks=(1, min(512, Y), min(512, X))) if compressed else {})
    out_h5 = os.path.join(save_path, f"{name}.h5")
    out_csv = os.path.join(save_path, f"{name}.csv")
    # CSV first (cheap) — read every shard csv before any shard is deleted, renumber image_id, concat.
    frames, off = [], 0
    for s, n in zip(shards, counts):
        df = pd.read_csv(s[:-3] + ".csv")
        df["image_id"] = df["image_id"].astype(int) + off
        frames.append(df)
        off += n
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    # h5: copy in 32-panel blocks and delete each shard right after it is consumed.
    with h5py.File(out_h5, "w") as fo:
        for k, dt in dsets:
            fo.create_dataset(k, shape=(total, Y, X), dtype=dt, **ds_kw)
        off = 0
        for s, n in zip(shards, counts):
            if n:
                with h5py.File(s, "r") as f:
                    for k, _dt in dsets:
                        for a in range(0, n, 32):
                            b = min(a + 32, n)
                            fo[k][off + a:off + b] = f[k][a:b]
                off += n
            if cleanup:
                for p in (s, s[:-3] + ".csv"):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    print(f"[gather] {name}: {len(shards)} shards -> {total} panels -> {out_h5}", flush=True)
    return total


# ======================================================================================
# CLI
# ======================================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Build the simulated (injected-trail) diffim datasets — train(+val), "
                    "train2(+val2), test — from one deterministic panel partition. See module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- Butler / region (needs stage3=template_coadd + stage2=PVI/sources) ---
    ap.add_argument("--repo", type=str, default="dp2_prep")
    ap.add_argument("--collections", nargs="+",
                    default=["LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3",
                             "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"],
                    help="Butler chain: the stage3 collection (template_coadd) + the stage2 "
                         "collection (preliminary_visit_image / single_visit_star_footprints).")
    ap.add_argument("--stage3-collection", default="LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3",
                    help="subset of --collections used to query template_coadd")
    ap.add_argument("--skymap", default="lsst_cells_v2")
    ap.add_argument("--where",
                    default="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') ")
    # --- output + which sets to build ---
    ap.add_argument("--save-path", default="../DATA/",
                    help="root dir; writes <set>.h5/<set>.csv per set + split.json")
    ap.add_argument("--sets", nargs="+", choices=list(_GROUPS), default=list(_GROUPS),
                    help="which groups to BUILD now: train -> {train,val}; train2 -> {train2,val2}; "
                         "test -> {test}. The partition always covers all five sets (cached in "
                         "split.json), so building only a subset stays consistent with a full build.")
    ap.add_argument("--n-train", type=int, default=1500, help="stage-1 segmentation training panels")
    ap.add_argument("--n-val", type=int, default=150, help="held-out val for stage-1 model selection")
    ap.add_argument("--n-train2", type=int, default=500, help="stage-2 cutout-CNN training panels")
    ap.add_argument("--n-val2", type=int, default=100, help="held-out val2 for the stage-2 CNN threshold")
    ap.add_argument("--n-test", type=int, default=300, help="held-out evaluation panels")
    ap.add_argument("--repartition", action="store_true",
                    help="re-select the panel universe + rewrite split.json. Otherwise an existing "
                         "split.json is reused (so partial/repeat builds are consistent + deterministic).")
    # --- injection core knobs ---
    ap.add_argument("--trail-length-min", type=float, default=6)
    ap.add_argument("--trail-length-max", type=float, default=60)
    ap.add_argument("--mag-min", type=float, default=22.5)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--mag-mode", choices=["psf_mag", "snr", "surface_brightness", "integrated_mag"], default="psf_mag")
    ap.add_argument("--psf-template", choices=["image", "kernel"], default="kernel")
    ap.add_argument("--beta-min", type=float, default=0)
    ap.add_argument("--beta-max", type=float, default=180)
    ap.add_argument("--number", type=int, default=20)
    ap.add_argument("--stack-detection-threshold", type=float, default=5.0,
                    help="stack DIA detection sigma for the stack_detection label (train family + val, "
                         "and the single-sigma test build when --test-sigmas is not given)")
    ap.add_argument("--test-sigmas", nargs="*", type=float, default=None,
                    help="label the TEST set at several stack-detection sigmas at once (e.g. 5 4 3): each "
                         "panel is injected once and detected at every sigma. Writes one gzip'd test.h5 "
                         "(shared images/masks, plain real_labels = deepest sigma for the model input, plus "
                         "real_labels_<s>sigma) and one test.csv (stack_detection_<s>sigma per sigma).")
    ap.add_argument("--realistic-trail", action="store_true", default=False,
                    help="render trails with the realistic (light-curve/tapered/curved) renderer "
                         "instead of the uniform galsim.Box (leakage-free: physical priors only)")
    ap.add_argument("--measure_trails", action="store_true", default=False)
    # --- determinism / scale ---
    ap.add_argument("--seed", type=int, default=2026,
                    help="ONE seed: panel universe selection + partition + per-panel injections")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=None)
    ap.add_argument("--skip-prevalidation", action="store_true", default=False,
                    help="skip the slow per-panel template/source pre-validation when selecting the "
                         "universe (faster; unbuildable panels are simply skipped at build time)")
    # --- multi-node sharding (one build spread across a SLURM job array) ---
    ap.add_argument("--plan-only", action="store_true", default=False,
                    help="select + partition the universe, write split.json, then EXIT (no build), so a "
                         "fleet of array tasks can reuse one deterministic partition.")
    ap.add_argument("--gather", action="store_true", default=False,
                    help="GATHER mode: concatenate <name>.shard*.{h5,csv} -> <name>.{h5,csv} for each "
                         "--only-sets name (renumbering image_id), then EXIT. No Butler/build. Run after a "
                         "sharded build to recombine each small set into the single file the trainer reads.")
    ap.add_argument("--only-sets", nargs="*", default=None,
                    help="build exactly these set NAMES (train/val/train2/val2/test), overriding the "
                         "--sets group expansion. For per-set parallel jobs.")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="split the built set into this many strided shards across array tasks; this task "
                         "builds shard --shard into <name>.shard<shard>.{h5,csv} (per-shard target = ceil/N). "
                         "n_shards=1 writes a single <name>.{h5,csv}.")
    ap.add_argument("--shard", type=int, default=0, help="which shard index this task builds (0..n_shards-1)")
    ap.add_argument("--exclude-pairs-csv", nargs="*", default=None,
                    help="CSV(s) with visit/FieldID + detector columns to keep OUT of the universe "
                         "(leakage guard against EXTERNAL sets, e.g. the real-asteroid catalog)")
    ap.add_argument("--min-ecliptic-lat", type=float, default=0.0,
                    help="select only panels whose visit is > this |ecliptic latitude| (deg) from the "
                         "ecliptic, to avoid real-asteroid (esp. main-belt) contamination. 0 = no cut.")
    args = ap.parse_args()

    import json
    if args.realistic_trail:
        os.environ["ADCNN_REALISTIC_TRAIL"] = "1"
        print("[main] realistic trail renderer ENABLED", flush=True)
    logging.getLogger("lsst").setLevel(logging.ERROR)
    ensure_dir(args.save_path)
    if args.gather:   # recombine a sharded build's per-shard files into single per-set files, then exit
        if not args.only_sets:
            raise SystemExit("[gather] --gather requires --only-sets <names>")
        for nm in args.only_sets:
            gather_shards(args.save_path, nm)
        print("GATHER DONE", flush=True)
        return
    coll = args.collections if len(args.collections) > 1 else args.collections[0]
    sizes = {"train": args.n_train, "val": args.n_val, "train2": args.n_train2,
             "val2": args.n_val2, "test": args.n_test}
    # Over-allocate the panel universe by ADCNN_ALLOC_HEADROOM so that panels which can't be built (no
    # overlapping template / too few kernel sources, ~20%) don't leave a set short; the build then CAPS
    # each set at its target count, so the on-disk size stays bounded. Set the headroom to 1.0 to disable
    # over-allocation (e.g. when the targets are already sized for the expected drop rate).
    HEADROOM = float(os.environ.get("ADCNN_ALLOC_HEADROOM", "1.5"))
    alloc = {k: (int(round(sizes[k] * HEADROOM)) if sizes[k] > 0 else 0) for k in _SET_ORDER}
    split_path = os.path.join(args.save_path, "split.json")

    # --- 1. panel partition (deterministic; cached in split.json) ---
    if os.path.exists(split_path) and not args.repartition:
        meta = json.loads(open(split_path).read())
        parts = {k: [(int(v), int(d)) for v, d in meta["sets"][k]] for k in _SET_ORDER}
        print(f"[split] reusing {split_path}: "
              + ", ".join(f"{k}={len(parts[k])}" for k in _SET_ORDER), flush=True)
    else:
        exclude_keys = set()
        for p in (args.exclude_pairs_csv or []):
            df = pd.read_csv(p)
            vcol = "visit" if "visit" in df.columns else "FieldID"   # real-mover catalogs use FieldID
            sub = df[[vcol, "detector"]].dropna()
            exclude_keys |= {(int(v), int(d)) for v, d in zip(sub[vcol], sub["detector"])}
        if exclude_keys:
            print(f"[split] excluding {len(exclude_keys)} (visit,detector) panels from "
                  f"{len(args.exclude_pairs_csv)} catalog(s) (real-asteroid leakage guard)", flush=True)
        n_universe = sum(alloc.values())
        refs = select_good_refs_random_check(
            repo=args.repo, collections=coll, where=args.where, skymap=args.skymap,
            stage3_collection=args.stage3_collection, instrument="LSSTCam",
            k=n_universe, seed=args.seed, exclude_keys=exclude_keys,
            min_ecliptic_lat=args.min_ecliptic_lat,
            check_refs=not args.skip_prevalidation, filter_dims=False, verbose=True)
        keys = [_key_from_dataId(r.dataId) for r in refs]
        if len(keys) < n_universe:
            print(f"[split] WARNING: universe has {len(keys)} < allocated {n_universe} panels; "
                  "sets may be short — widen --where, lower --n-*, or lower --min-ecliptic-lat.", flush=True)
        parts = _partition(keys, alloc, args.seed)   # partition the OVER-ALLOCATED universe
        meta = {"seed": args.seed, "where": args.where,
                "min_ecliptic_lat": args.min_ecliptic_lat,
                "exclude_pairs_csv": args.exclude_pairs_csv or [],
                "targets": sizes, "alloc": {k: len(parts[k]) for k in _SET_ORDER},
                "sizes": {k: len(parts[k]) for k in _SET_ORDER},
                "sets": {k: [[int(v), int(d)] for (v, d) in parts[k]] for k in _SET_ORDER}}
        with open(split_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[split] wrote {split_path}: "
              + ", ".join(f"{k}={len(parts[k])}" for k in _SET_ORDER), flush=True)

    # --plan-only: the deterministic partition (split.json) is now written/reused; stop here so a fleet
    # of sharded array tasks can all reuse this ONE partition without racing to (re)select it.
    if args.plan_only:
        print(f"[main] PLAN ONLY -> {split_path} written/reused; exiting before build", flush=True)
        return

    # --- 2. build the requested sets ---
    # `--only-sets a b c` builds exactly those set names (for sharded/parallel jobs); else expand groups.
    build = list(args.only_sets) if args.only_sets else [s for g in args.sets for s in _GROUPS[g]]
    all_keys = [k for s in _SET_ORDER for k in parts[s]]
    if not all_keys:
        raise SystemExit("[main] empty partition; nothing to build")
    butler = Butler(args.repo, collections=coll)
    # Common h5 frame = MAX panel dims over every distinct detector in the universe (LSSTCam mixes
    # ITL 4072x4000 and e2v 4096x4004 CCDs). Each panel is built at its own size then padded to this
    # frame, so BOTH geometries are kept rather than dropping the minority vendor. Dims are a detector
    # property, so one (visit,detector) per distinct detector covers all geometries present.
    import concurrent.futures as _cf
    _uniq = {}
    for (v, d) in all_keys:
        _uniq.setdefault(int(d), (int(v), int(d)))

    def _dims_of(vd):
        v, d = vd
        try:
            lb = Butler(args.repo, collections=coll)
            dd = lb.get("preliminary_visit_image.dimensions",
                        dataId={"instrument": "LSSTCam", "visit": int(v), "detector": int(d)})
            return int(dd.y), int(dd.x)
        except Exception:
            return None
    with _cf.ThreadPoolExecutor(max_workers=16) as _ex:
        _ds = [r for r in _ex.map(_dims_of, _uniq.values()) if r]
    if not _ds:
        raise SystemExit("[main] could not read panel dimensions for any detector")
    dims = SimpleNamespace(y=max(r[0] for r in _ds), x=max(r[1] for r in _ds))
    print(f"[main] h5 frame padded to max (y,x)=({dims.y},{dims.x}) over {len(_ds)}/{len(_uniq)} "
          f"distinct detectors", flush=True)
    test_sigma = args.stack_detection_threshold
    common = dict(repo=args.repo, coll=coll, dims=dims, save_path=args.save_path,
                  number=args.number, trail_length=[args.trail_length_min, args.trail_length_max],
                  magnitude=[args.mag_min, args.mag_max], beta=[args.beta_min, args.beta_max],
                  mag_mode=args.mag_mode, psf_template=args.psf_template,
                  measure_trails=args.measure_trails, seed=args.seed, skymap=args.skymap,
                  stage3_collection=args.stage3_collection, parallel=args.parallel, chunks=args.chunks)
    nsh = max(1, int(args.n_shards))   # multi-node sharding: this task builds 1 of nsh
    sh = int(args.shard)
    built_counts = {}
    for name in _SET_ORDER:
        if name not in build:
            continue
        if not parts[name]:
            print(f"[build:{name}] no panels in partition; skipping", flush=True)
            continue
        # Shard: this task takes a strided slice of the over-allocated panels and builds 1/nsh of the
        # target into <name>.shard<sh>.{h5,csv}; the parts are read from the SHARED split.json so every
        # shard is disjoint and consistent. nsh==1 -> the normal single-file build.
        panels = parts[name][sh::nsh] if nsh > 1 else parts[name]
        out_name = f"{name}.shard{sh}" if nsh > 1 else name
        tgt = int(math.ceil(sizes[name] / nsh)) if nsh > 1 else sizes[name]   # per-shard build-to-target cap
        dataids = [{"instrument": "LSSTCam", "visit": int(v), "detector": int(d)} for (v, d) in panels]
        if name == "test" and args.test_sigmas:
            # test labelled at several sigmas at once -> compressed h5 (shared images/masks +
            # real_labels_<sigma>sigma) + csv (stack_detection_<sigma>sigma per sigma).
            built_counts[out_name] = _build_set(dataids, name=out_name, stack_detection_thresholds=args.test_sigmas,
                                                compress=True, target=tgt, **common)
        elif name == "test":
            built_counts[out_name] = _build_set(dataids, name=out_name, stack_detection_threshold=test_sigma,
                                                compress=True, target=tgt, **common)
        else:
            built_counts[out_name] = _build_set(dataids, name=out_name, stack_detection_threshold=args.stack_detection_threshold,
                                                compress=False, target=tgt, **common)
        # Guard per set: abort if a build comes up far short of its target (a sign something is wrong on
        # this node), so it can't silently feed a too-small dataset into training. Skip the check for tiny
        # per-shard targets, where a single unbuildable panel is normal.
        if tgt >= 10 and built_counts[out_name] < 0.50 * tgt:
            raise SystemExit(f"[main] ABORT: {out_name} built {built_counts[out_name]} < 50% of target {tgt}; "
                             "check the log")
    print("DATASETS DONE built:", built_counts, flush=True)


if __name__ == "__main__":
    main()
