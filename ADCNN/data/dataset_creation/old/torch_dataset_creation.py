from astroML.crossmatch import crossmatch_angular
from lsst.daf.butler import Butler
import numpy as np
from astropy.table import Table
import cv2
from lsst.geom import Point2D
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.source.injection.inject_exposure import ExposureInjectTask
from lsst.meas.extensions.psfex.psfexPsfDeterminer import PsfexNoGoodStarsError
import h5py
import concurrent.futures
from multiprocessing import Lock, Semaphore, Manager
from astropy.io import ascii
import gc
import psutil
import tracemalloc
import os
import pandas as pd
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from multiprocessing import Value, Lock

completed_counter = Value('i', 0)
counter_lock = Lock()


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

def mem(label=""):
    gc.collect()
    print(f"\n=== {label} ===")
    print(f"Resident memory: {(psutil.Process(os.getpid()).memory_info().rss / 1024**2):.1f} MB")
    top_stats = tracemalloc.take_snapshot().statistics('lineno')
    print("[Top alloc lines]")
    for stat in top_stats[:5]:
        print(stat)

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

def characterizeCalibrate(postISRCCD, verbose=False):
    char_config = CharacterizeImageTask.ConfigClass()
    char_config.doApCorr = True
    char_config.doDeblend = True
    char_task = CharacterizeImageTask(config=char_config)
    if verbose:
        char_result = char_task.run(postISRCCD)
    else:
        with suppress_stdout():
            char_result = char_task.run(postISRCCD)
    calib_config = CalibrateTask.ConfigClass()
    calib_config.doAstrometry = False
    calib_config.doPhotoCal = False
    calib_task = CalibrateTask(config=calib_config, icSourceSchema=char_result.sourceCat.schema)
    if verbose:
        calib_result = calib_task.run(postISRCCD, background=char_result.background, icSourceCat=char_result.sourceCat)
    else:
        with suppress_stdout():
            calib_result = calib_task.run(postISRCCD, background=char_result.background, icSourceCat=char_result.sourceCat)
    return calib_result.outputExposure, calib_result.sourceCat

def inject(postISRCCD, injection_catalog):
    inject_task = ExposureInjectTask()
    inject_res = inject_task.run([injection_catalog], postISRCCD, postISRCCD.psf, postISRCCD.photoCalib, postISRCCD.wcs)
    return inject_res.output_exposure

def generate_one_line(n_inject, trail_length, mag, beta, butler, ref, dimensions, source_type):
    injection_catalog = Table(names=('injection_id', 'ra', 'dec', 'source_type', 'trail_length', 'mag', 'beta', 'visit', 'detector', 'integrated_mag', 'PSF_mag', 'physical_filter', 'x', 'y'),
                               dtype=('int64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'int64', 'int64', 'float64', 'float64', 'str', 'int64', 'int64'))
    raw = butler.get(source_type + ".wcs", dataId=ref.dataId)
    info = butler.get(source_type + ".visitInfo", dataId=ref.dataId)
    filter_name = butler.get(source_type + ".filter", dataId=ref.dataId)
    fwhm = {"u": 0.92, "g": 0.87, "r": 0.83, "i": 0.80, "z": 0.78, "y": 0.76}
    m5 = {"u": 23.7, "g": 24.97, "r": 24.52, "i": 24.13, "z": 23.56, "y": 22.55}
    psf_depth = m5[filter_name.bandLabel]
    pixelScale = raw.getPixelScale().asArcseconds()
    theta_p = fwhm[filter_name.bandLabel] * pixelScale
    a, b = 0.67, 1.16
    for k in range(n_inject):
        x_pos = np.random.uniform(1, dimensions.x - 1)
        y_pos = np.random.uniform(1, dimensions.y - 1)
        sky_pos = raw.pixelToSky(x_pos, y_pos)
        ra_pos = sky_pos.getRa().asDegrees()
        dec_pos = sky_pos.getDec().asDegrees()
        inject_length = np.random.uniform(*trail_length)
        x = inject_length / (24 * theta_p)
        upper_limit_mag = psf_depth - 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x)) if mag[1] == 0 else mag[1]
        magnitude = np.random.uniform(mag[0], upper_limit_mag)
        surface_brightness = magnitude + 2.5 * np.log10(inject_length)
        psf_magnitude = magnitude + 1.25 * np.log10(1 + (a * x ** 2) / (1 + b * x))
        angle = np.random.uniform(*beta)
        injection_catalog.add_row([k, ra_pos, dec_pos, "Trail", inject_length, surface_brightness, angle, info.id, int(ref.dataId["detector"]), magnitude, psf_magnitude, str(filter_name.bandLabel), x_pos, y_pos])
    return injection_catalog

def stack_hits(pre_injection_Src, post_injection_Src, calexp, dimensions, mask, injection_catalog):
    """
    Label injected sources that were detected by the LSST pipeline ("stack").
    Strategy:
      - Crossmatch post-injection sources to pre-injection sources by sky coords (radians), radius ~ 0.04".
      - New-only detections = rows in post that do NOT match pre.
      - Convert those new detections to pixel coords; rasterize into a mask with compact IDs 1..K.
      - Overlap with injected ground-truth mask (values 1..N) to mark which injections were detected.
    Outputs columns on injection_catalog: stack_detection (bool), stack_mag, stack_mag_err.
    """
    #Compute PSF mags for POST catalog (injected scene)
    mags = calexp.photoCalib.instFluxToMagnitude(post_injection_Src, 'base_PsfFlux')  # (N,2)
    post = post_injection_Src.asAstropy().to_pandas()
    post["magnitude"] = mags[:, 0]
    post["magnitude_err"] = mags[:, 1]

    # Keep primary
    post = post[post.get("parent", 0) == 0].copy()
    pre = pre_injection_Src.asAstropy().to_pandas()
    pre = pre[pre.get("parent", 0) == 0].copy()

    # Drop obvious bad fluxes
    if "base_PsfFlux_flag" in post.columns:
        post = post[post["base_PsfFlux_flag"] == False].copy()

    # Crossmatch POST vs PRE on-sky (inputs in radians, radius in radians)
    #    post-only detections -> "new" sources likely caused by injections
    if len(pre) > 0 and len(post) > 0:
        # arrays of [ra, dec] in radians
        P = post[["coord_ra", "coord_dec"]].values
        R = pre[["coord_ra", "coord_dec"]].values
        max_sep = np.deg2rad(0.04 / 3600.0)  # 0.04 arcsec -> radians
        dist, ind = crossmatch_angular(P, R, max_sep)
        is_new = np.isinf(dist)  # no match in PRE → likely injected
        new_post = post[is_new].copy()
    else:
        new_post = post.copy()

    if len(new_post) == 0:
        # Initialize output columns to "not detected"
        injection_catalog["stack_detection"] = False
        injection_catalog["stack_mag"] = np.nan
        injection_catalog["stack_mag_err"] = np.nan
        return injection_catalog

    # coords → pixels (radians if degrees=False)
    ra_rad = new_post["coord_ra"].to_numpy()
    dec_rad = new_post["coord_dec"].to_numpy()
    xx, yy  = calexp.wcs.skyToPixelArray(ra_rad, dec_rad, degrees=False)

    # image width/height as plain Python ints
    H = int(getattr(dimensions, "y", dimensions[0]))
    W = int(getattr(dimensions, "x", dimensions[1]))

    # round, clip, flatten → 1D index arrays with native index dtype
    x_pix = np.rint(xx).astype(np.intp, copy=False)
    y_pix = np.rint(yy).astype(np.intp, copy=False)
    x_pix = np.clip(x_pix, 0, W - 1).ravel()
    y_pix = np.clip(y_pix, 0, H - 1).ravel()

    # K as a real Python int (avoid 0-D/array scalars)
    K = int(x_pix.shape[0])

    stack_mask = np.zeros((H, W), dtype=np.int32)
    compact_ids = np.arange(1, K + 1, dtype=np.int32)

    # write with tuple indexing; x/y are 1D np.intp → no deprecation
    stack_mask[(y_pix, x_pix)] = compact_ids

    # Overlap with injected GT mask (values 1..N, where N == len(injection_catalog))
    # mask != 0 means which injected object id is present (id = mask-1)
    hits = (mask != 0) & (stack_mask != 0)
    if not hits.any():
        injection_catalog["stack_detection"] = False
        injection_catalog["stack_mag"] = np.nan
        injection_catalog["stack_mag_err"] = np.nan
        return injection_catalog

    inj_ids = mask[hits].astype(np.int64) - 1          # 0-based indices into injection_catalog
    stk_ids = stack_mask[hits].astype(np.int64) - 1    # 0-based indices into new_post rows (COMPACT)

    # Build lookup arrays for magnitudes aligned with the COMPACT order
    new_post_reset = new_post.reset_index(drop=True)   # 0..K-1
    mag_lookup     = new_post_reset["magnitude"].to_numpy()
    magerr_lookup  = new_post_reset["magnitude_err"].to_numpy()

    # Initialize output columns and fill where hits happen
    if "stack_detection" not in injection_catalog.colnames:
        injection_catalog["stack_detection"] = False
        injection_catalog["stack_mag"] = np.nan
        injection_catalog["stack_mag_err"] = np.nan

    # Set flags/mags for all overlaps
    injection_catalog["stack_detection"][inj_ids] = True
    injection_catalog["stack_mag"][inj_ids] = mag_lookup[stk_ids]
    injection_catalog["stack_mag_err"][inj_ids] = magerr_lookup[stk_ids]

    return injection_catalog


def one_detector_injection(n_inject, trail_length, mag, beta, repo, coll, dimensions, source_type, ref_dataId):
    try:
        butler = Butler(repo, collections=coll)
        ref = butler.registry.findDataset(source_type, dataId=ref_dataId)
        injection_catalog = generate_one_line(n_inject, trail_length, mag, beta, butler, ref, dimensions, source_type)
        calexp = butler.get("calexp", dataId=ref.dataId)
        pre_injection_Src = butler.get("src", dataId=ref.dataId)
        injected_calexp, post_injection_Src = characterizeCalibrate(inject(calexp, injection_catalog))
        mask = np.zeros((dimensions.y, dimensions.x), dtype=int)
        for i, row in enumerate(injection_catalog):
            psf_width = injected_calexp.psf.getLocalKernel(Point2D(row["x"], row["y"])).getWidth()
            mask = draw_one_line(mask, [row["x"], row["y"]], row["beta"], row["trail_length"], true_value=i+1, line_thickness=psf_width)
        injection_catalog = stack_hits (pre_injection_Src, post_injection_Src, calexp, dimensions, mask, injection_catalog)
        return injected_calexp.image.array.astype("float32"), mask.astype("bool"), injection_catalog
    except Exception as e:
        print(f"[WARNING] Skipping {ref_dataId} due to failure: {e}")
        return None

def worker(args):
    idx, dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta, source_type = args
    res = one_detector_injection(number, trail_length, magnitude, beta, repo, coll, dims, source_type, dataId)
    with counter_lock:
            completed_counter.value += 1
            print(f"[{completed_counter.value}/{total_tasks}] done", flush=True)
    if res is None:
        return
    img, mask, catalog = res
    with lock:
        with h5py.File(h5path, "a") as f:
            f["images"][idx] = img
            f["masks"][idx] = mask
        df = catalog.to_pandas()
        df["image_id"] = idx
        df.to_csv(csvpath, mode=("w" if idx == 0 else "a") , header=(idx == 0), index=False)

def run_parallel_injection(repo, coll, save_path, number, trail_length, magnitude, beta, where, parallel=4, random_subset=0, train_test_split=0):
    
    
    butler = Butler(repo, collections=coll)
    h5train_path = os.path.join(save_path, "train.h5")
    h5test_path = os.path.join(save_path, "test.h5")
    
    refs = list(set(butler.registry.queryDatasets("calexp", where=where, instrument="LSSTComCam", findFirst=True)))
    if random_subset>0:
        refs = list(np.random.choice(refs, random_subset, replace=False))
    global total_tasks
    total_tasks = len(refs)  # Needed for progress display

    test_index = np.random.choice(np.arange(len(refs)), int((1-train_test_split)*len(refs)), replace=False) if 0 < train_test_split < 1 else []
    dims = butler.get("calexp.dimensions", dataId=refs[0].dataId)
    with h5py.File(h5train_path, "w") as f:
        f.create_dataset("images", shape=(len(refs)-len(test_index), dims.y, dims.x), dtype="float32")
        f.create_dataset("masks", shape=(len(refs)-len(test_index), dims.y, dims.x), dtype="bool")
    if len(test_index)>0:
        with h5py.File(h5test_path, "w") as f:
            f.create_dataset("images", shape=(len(test_index), dims.y, dims.x), dtype="float32")
            f.create_dataset("masks", shape=(len(test_index), dims.y, dims.x), dtype="bool")
    manager = Manager()
    lock = manager.Lock()
    count_train = 0
    count_test = 0
    tasks = []
    for i,ref in enumerate(refs):
        if i in test_index:
            h5path = h5test_path
            csvpath = os.path.join(save_path, "test.csv")
            count=count_test
            count_test+=1
        else:
            h5path = h5train_path
            csvpath = os.path.join(save_path, "train.csv")
            count=count_train
            count_train+=1
        tasks.append ([count, ref.dataId, repo, coll, dims, lock, h5path, csvpath, number, trail_length, magnitude, beta, "calexp"])
    if parallel>1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            list(executor.map(worker, tasks))
    else:
        for task in tasks:
            worker (task)

# Example run
if __name__ == "__main__":
    where = "instrument='LSSTComCam' AND skymap='lsst_cells_v1' AND day_obs>=20241101 AND day_obs<=20241127 AND exposure.observation_type='science' AND band in ('u','g','r','i','z','y') AND (exposure not in (2024110600163, 2024110800318, 2024111200185, 2024111400039, 2024111500225, 2024111500226, 2024111500239, 2024111500240, 2024111500242, 2024111500288, 2024111500289, 2024111800077, 2024111800078, 2024112300230, 2024112400094, 2024112400225, 2024112600327))"
    
    run_parallel_injection(
        repo="/repo/main",
        coll="LSSTComCam/runs/DRP/DP1/w_2025_03/DM-48478",
        save_path="./",
        number=20,
        trail_length=[6, 60],
        magnitude=[19, 24],
        beta=[0, 180],
        parallel=10,
        where=where,
        random_subset=850, 
        train_test_split=0.94117
    )
