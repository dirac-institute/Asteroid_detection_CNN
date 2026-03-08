from __future__ import annotations
import argparse
from pathlib import Path
import random
from typing import List, Sequence

from ADCNN.data.dataset_creation.common import ensure_dir
from ADCNN.data.dataset_creation.simulate_inject import one_detector_injection, select_good_refs_random_check

from lsst.daf.butler import Butler
import numpy as np
import concurrent.futures
from multiprocessing import Lock, Semaphore, Manager
import os
from multiprocessing import Value, Lock
import logging
import pandas as pd
import traceback

completed_counter = Value('i', 0)
counter_lock = Lock()

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
            df = catalog.to_pandas()
            df["image_id"] = idx
            file_exists = os.path.exists(csvpath)
            df.to_csv(csvpath, mode=("a" if file_exists else "w"),
                      header=(not file_exists), index=False)
        return ("ok", idx)

    except Exception:
        tb = traceback.format_exc()
        return ("err", idx, dataId, tb)

def _key_from_dataId(d):
    return (int(d["visit"]), int(d["detector"]))


def reservoir_sample(iterable, k: int, seed: int = 123):
    """Uniform sample of size k from an iterable without materializing it."""
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


def select_good_refs_random_check(
    *,
    repo: str,
    collections: str | Sequence[str],
    where: str,
    instrument: str = "LSSTCam",
    k: int = 200,
    seed: int = 123,
    pool_size: int = 5000,
    max_checks: int = 200000,
    verbose: bool = True,
) -> List:
    """
    Deterministically:
      1) build a random pool of `pool_size` preliminary_visit_image refs (reservoir sample)
      2) shuffle the pool with `seed`
      3) iterate, accept refs that satisfy:
           - single_visit_star_footprints exists
           - preliminary_visit_image.photoCalib component loads and is not None
      4) stop when `k` good refs collected or `max_checks` reached

    Notes:
      - Fully deterministic given (repo, collections, where, instrument, seed, pool_size, max_checks)
      - Serial IO (no parallelism)
    """
    b = Butler(repo, collections=collections)

    # Step 1: sample from all preliminary_visit_image refs (no full materialization)
    all_pvi_iter = b.registry.queryDatasets(
        "preliminary_visit_image",
        instrument=instrument,
        where=where,
        collections=collections,
        findFirst=True,
    )
    pool = reservoir_sample(all_pvi_iter, k=int(pool_size), seed=int(seed))

    # Step 2: deterministic shuffle order
    rng = random.Random(int(seed))
    rng.shuffle(pool)

    good = []
    seen = set()

    checks = 0
    for pvi in pool:
        if len(good) >= int(k) or checks >= int(max_checks):
            break
        checks += 1

        key = _key_from_dataId(pvi.dataId)
        if key in seen:
            continue
        seen.add(key)

        # A) require star footprints to exist (fast registry check)
        svsf = b.registry.findDataset(
            "single_visit_star_footprints",
            dataId=pvi.dataId,
            collections=collections,
        )
        if svsf is None:
            continue

        # B) require photoCalib to load and not be None
        try:
            pc = b.get(pvi.makeComponentRef("photoCalib"))
        except Exception:
            continue
        if pc is None:
            continue

        good.append(pvi)

        if verbose and (len(good) % 25 == 0 or len(good) == k):
            print(f"good={len(good)}/{k} checks={checks}", flush=True)

    good = sorted(good, key=lambda r: _key_from_dataId(r.dataId))

    if verbose:
        print(f"Selected good refs: {len(good)} (requested k={k}), from pool_size={pool_size}, checks={checks}", flush=True)

    return good

def run_parallel_injection(repo, coll, save_path, number, trail_length, magnitude, beta, where, parallel=4,
                           random_subset=0, train_test_split=0, seed=123, chunks=None, test_only=False, bad_visits_file=None, mag_mode="psf_mag",
                           psf_template="image"):
    butler = Butler(repo, collections=coll)
    h5train_path = os.path.join(save_path, "train.h5")
    h5test_path = os.path.join(save_path, "test.h5")

    refs = select_good_refs_random_check(
        repo=repo,
        collections=coll,
        where=where,
        instrument="LSSTCam",
        k=int(random_subset) if int(random_subset) > 0 else 200,
        seed=seed,
        pool_size=5000,  # increase if acceptance rate is low
        max_checks=200000,  # safety cap
    )
    print("Selected datasets:", len(refs))

    global total_tasks
    total_tasks = len(refs)  # Needed for progress display
    rng_split = np.random.default_rng(seed + 1)
    test_index = rng_split.choice(np.arange(len(refs)), int((1 - train_test_split) * len(refs)),
                                  replace=False) if 0 < train_test_split < 1 else []
    if test_only:
        total_tasks = len(test_index)
    manager = Manager()
    lock = manager.Lock()
    count_train = 0
    count_test = 0
    tasks = []
    for i, ref in enumerate(refs):
        dims = butler.get("preliminary_visit_image.dimensions", dataId=refs[i].dataId)
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
    ap.add_argument("--repo", type=str, default="dp2_prep")
    ap.add_argument("--collections", type=str, default="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2")
    ap.add_argument("--save-path", default="../DATA/")
    ap.add_argument("--where", default="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') ")
    ap.add_argument("--bad-visits-file", type=str, default=None,)
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