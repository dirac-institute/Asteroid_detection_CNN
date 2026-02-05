from __future__ import annotations
import argparse
from pathlib import Path

from common import ensure_dir, draw_one_line, psf_fwhm_arcsec_from_calexp, mag_to_snr, snr_to_mag

from astroML.crossmatch import crossmatch_angular
from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
from lsst.geom import Point2D
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.source.injection.inject_exposure import ExposureInjectTask
from lsst.meas.extensions.psfex.psfexPsfDeterminer import PsfexNoGoodStarsError
import h5py
import concurrent.futures
from multiprocessing import Lock, Semaphore, Manager
from astropy.io import ascii
import os
from multiprocessing import Value, Lock
import logging
import pandas as pd
import traceback

completed_counter = Value('i', 0)
counter_lock = Lock()


def characterizeCalibrate(postISRCCD):
    char_config = CharacterizeImageTask.ConfigClass()
    char_config.doApCorr = True
    char_config.doDeblend = True
    #char_config.doApCorr = False
    #char_config.doDeblend = False
    char_task = CharacterizeImageTask(config=char_config)
    char_result = char_task.run(postISRCCD)

    calib_config = CalibrateTask.ConfigClass()
    calib_config.doAstrometry = False
    calib_config.doPhotoCal = False
    calib_task = CalibrateTask(config=calib_config, icSourceSchema=char_result.sourceCat.schema)
    calib_result = calib_task.run(postISRCCD, background=char_result.background, icSourceCat=char_result.sourceCat)
    return calib_result.outputExposure, calib_result.sourceCat


def inject(postISRCCD, injection_catalog):
    inject_task = ExposureInjectTask()
    inject_res = inject_task.run([injection_catalog], postISRCCD, postISRCCD.psf, postISRCCD.photoCalib, postISRCCD.wcs)
    return inject_res.output_exposure


def generate_one_line(n_inject, trail_length, mag, beta, ref, dimensions, seed, calexp, mag_mode="psf_mag", psf_template="image", forbidden_mask=None):
    rng = np.random.default_rng(seed)
    injection_catalog = Table(
        names=('injection_id', 'ra', 'dec', 'source_type', 'trail_length', 'mag', 'beta', 'visit', 'detector',
               'integrated_mag', 'PSF_mag', 'SNR', 'physical_filter', 'x', 'y', 'stack_model_SNR'),
        dtype=('int64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'int64', 'int64', 'float64',
               'float64', 'float64', 'str', 'int64', 'int64', 'float64'))

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

        # Mark these pixels as now-occupied so subsequent injections cannot overlap them
        forbidden |= stamp

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
            if snr_max <= snr_min:
                raise ValueError(f"Bad SNR range: snr_min={snr_min} snr_max={snr_max}")
            snr = float(rng.uniform(snr_min, snr_max))
            snr = max(snr, 0.01)
            psf_magnitude = snr_to_mag(snr, calexp, x_pos, y_pos, l_pix=length, theta_deg=angle, use_kernel_image=use_kernel, snr_definition="detection")
            if trail_length > 0:
                magnitude = psf_magnitude - 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x))
                surface_brightness = magnitude + 2.5 * np.log10(length)
            else:
                magnitude = psf_magnitude
                surface_brightness = magnitude
            stack_snr = mag_to_snr(magnitude, calexp, x_pos, y_pos, use_kernel_image=use_kernel, l_pix=length, theta_deg=angle, snr_definition="measurement")
        elif mag_mode == "psf_mag":
            psf_magnitude = rng.uniform(mag[0], upper_limit_mag)
            if trail_length > 0:
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
            if trail_length > 0:
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
            if trail_length > 0:
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
                                       x_pos, y_pos, stack_snr])
    return injection_catalog

def stack_hits_by_footprints(
    post_src,
    calexp_pre,
    dimensions,
    truth_id_mask,
    injection_catalog,
    overlap_frac=0.02,
    overlap_minpix=100,
):
    H, W = int(dimensions.y), int(dimensions.x)
    N = len(injection_catalog)

    det_flag = np.zeros(N, bool)
    det_mag = np.full(N, np.nan)
    det_magerr = np.full(N, np.nan)
    det_snr = np.full(N, np.nan)
    matched_fp_masks = [np.zeros((H, W), bool) for _ in range(N)]

    mags = calexp_pre.photoCalib.instFluxToMagnitude(post_src, "base_PsfFlux")

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
            matched_fp_masks[inj_id][y0:y1+1, x0:x1+1] |= best_fp

    injection_catalog["stack_detection"] = det_flag
    injection_catalog["stack_mag"] = det_mag
    injection_catalog["stack_mag_err"] = det_magerr
    injection_catalog["stack_snr"] = det_snr
    return injection_catalog, matched_fp_masks

def crossmatch_catalogs (pre, post):
    # Crossmatch POST vs PRE on-sky (inputs in radians, radius in radians)
    #    post-only detections -> "new" sources likely caused by injections
    if len(pre) > 0 and len(post) > 0:
        # arrays of [ra, dec] in radians
        P = post.asAstropy().to_pandas()[["coord_ra", "coord_dec"]].values
        R = pre.asAstropy().to_pandas()[["coord_ra", "coord_dec"]].values
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

def one_detector_injection(n_inject, trail_length, mag, beta, repo, coll, dimensions, source_type, ref_dataId, seed=None, debug=False, mag_mode="psf_mag", psf_template="image"):
    try:
        if seed is None:
            seed = np.random.randint(0,10000)
        butler = Butler(repo, collections=coll)
        ref = butler.registry.findDataset(source_type, dataId=ref_dataId)
        calexp = butler.get("preliminary_visit_image", dataId=ref.dataId)
        pre_injection_Src = butler.get("single_visit_star_footprints", dataId=ref.dataId)
        forbidden = build_forbidden_mask(calexp, pre_injection_Src, dimensions)
        injection_catalog = generate_one_line(n_inject, trail_length, mag, beta, ref, dimensions, seed, calexp, mag_mode=mag_mode, psf_template=psf_template, forbidden_mask=forbidden)
        injected_calexp, post_injection_Src = characterizeCalibrate(inject(calexp, injection_catalog))
        mask = np.zeros((dimensions.y, dimensions.x), dtype=int)
        for i, row in enumerate(injection_catalog):
            psf_width = injected_calexp.psf.getLocalKernel(Point2D(row["x"], row["y"])).getWidth()
            mask = draw_one_line(mask, [row["x"], row["y"]], row["beta"], row["trail_length"], true_value=i + 1,
                                 line_thickness=int(psf_width/2))
        injection_catalog, matched_fp_mask = stack_hits_by_footprints(post_src=crossmatch_catalogs (pre_injection_Src, post_injection_Src),
                                                                       calexp_pre=calexp,
                                                                       dimensions=dimensions,
                                                                       truth_id_mask=mask,
                                                                       injection_catalog=injection_catalog,
                                                                       overlap_minpix=1,
                                                                       overlap_frac=0.0,)

        real_labels = footprints_to_label_mask(pre_injection_Src, dimensions, dtype=np.uint16)
        if not debug:
            return True, injected_calexp.image.array.astype("float32"), mask.astype("bool"), real_labels, injection_catalog
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
            return True, injected_calexp.image.array.astype("float32"), mask.astype("bool"), real_labels, injection_catalog, det_mask, matched_fp_masks
    except Exception as e:
        return False, ref_dataId, repr(e), traceback.format_exc()
        return {"ok": False, "dataId": ref_dataId, "error": repr(e), "traceback": traceback.format_exc()}


def worker(args):
    idx, dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta, source_type, global_seed, mag_mode, psf_template = args
    seed = (int(global_seed) * 1_000_003 + int(dataId["visit"]) * 1_003 + int(dataId["detector"])) & 0xFFFFFFFF
    try:
        res = one_detector_injection(number, trail_length, magnitude, beta, repo, coll, dims, source_type, dataId, seed,
                                     mag_mode=mag_mode, psf_template=psf_template)
        if res[0] is False:
            return ("err", res[1], res[2], res[3])
        _, img, mask, real_labels, catalog = res
        with lock:
            with h5py.File(h5path, "a") as f:
                f["images"][idx] = img
                f["masks"][idx] = mask
                if "real_labels" in f:
                    f["real_labels"][idx] = real_labels

            df = catalog.to_pandas()
            df["image_id"] = idx
            file_exists = os.path.exists(csvpath)
            df.to_csv(csvpath, mode=("a" if file_exists else "w"),
                      header=(not file_exists), index=False)
        return ("ok", idx)

    except Exception:
        tb = traceback.format_exc()
        return ("err", idx, dataId, tb)


def run_parallel_injection(repo, coll, save_path, number, trail_length, magnitude, beta, where, parallel=4,
                           random_subset=0, train_test_split=0, seed=123, chunks=None, test_only=False, bad_visits_file=None, mag_mode="psf_mag",
                           psf_template="image"):
    butler = Butler(repo, collections=coll)
    h5train_path = os.path.join(save_path, "train.h5")
    h5test_path = os.path.join(save_path, "test.h5")

    refs = list(set(butler.registry.queryDatasets("preliminary_visit_image", where=where, instrument="LSSTComCam", findFirst=True)))
    refs = sorted(refs, key=lambda r: str(r.dataId["visit"]*1000+r.dataId["detector"]))
    if bad_visits_file is not None:
        bad_df = pd.read_csv(bad_visits_file)
        bad_set = set(zip(bad_df["visit"].astype(int), bad_df["detector"].astype(int)))
        refs = [r for r in refs if (int(r.dataId["visit"]), int(r.dataId["detector"])) not in bad_set]
    if random_subset > 0:
        rng_subset = np.random.default_rng(seed)
        refs = list(rng_subset.choice(refs, random_subset, replace=False))
    global total_tasks
    total_tasks = len(refs)  # Needed for progress display
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
                          "preliminary_visit_image", seed, mag_mode, psf_template])
        elif not test_only:
            h5path = h5train_path
            csvpath = os.path.join(save_path, "train.csv")
            count = count_train
            count_train += 1
            tasks.append([count, ref.dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta,
                          "preliminary_visit_image", seed, mag_mode, psf_template])
    if parallel > 1:
        completed = 0
        total_tasks = len(tasks)

        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(worker, t) for t in tasks]

            for fut in concurrent.futures.as_completed(futs):
                completed += 1
                out = fut.result()

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
    # stable across runs
    s = (int(seed) * 1_000_003
         + int(dataId["visit"]) * 1_003
         + int(dataId["detector"])) & 0xFFFFFFFF
    return np.random.default_rng(s)

def main():
    ap = argparse.ArgumentParser("Build a SIMULATED (injected) dataset")
    ap.add_argument("--repo", type=str, default="/repo/main")
    ap.add_argument("--collections", type=str, default="LSSTComCam/runs/DRP/DP1/w_2025_17/DM-50530")
    ap.add_argument("--save-path", default="../../../DATA/")
    ap.add_argument("--where", default="")
    ap.add_argument("--bad-visits-file", type=str, default="./bad_visits.csv",)
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
    ap.add_argument("--seed", type=int, default=123)
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
        bad_visits_file=args.bad_visits_file,
        psf_template=args.psf_template,
    )

if __name__ == "__main__":
    main()