from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import logging
import os
import signal
import time
import traceback
from multiprocessing import Manager

import numpy as np

from ADCNN.data.dataset_creation.common import ensure_dir
from ADCNN.data.dataset_creation.simulate_inject_fill_deterministic import (
    ATTEMPT_DIAGNOSTICS,
    TASK_TIMEOUT_SECONDS,
    _alarm_timeout_handler,
    compute_split_targets,
    compute_target_total,
    format_attempt_diagnostics,
    one_detector_injection,
    select_candidate_refs_deterministic,
)


def worker(args):
    rank, image_id, split_name, data_id, repo, coll, number, trail_length, magnitude, beta, global_seed, mag_mode, psf_template, stack_detection_threshold, csvpath, lock = args
    seed = (int(global_seed) * 1_000_003 + int(data_id["visit"]) * 1_003 + int(data_id["detector"])) & 0xFFFFFFFF
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
                    None,
                    "preliminary_visit_image",
                    data_id,
                    seed,
                    mag_mode=mag_mode,
                    psf_template=psf_template,
                    detection_threshold=stack_detection_threshold,
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

        _, _, _, _, catalog, mem_log = res
        t2 = time.perf_counter()
        with lock:
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
            "dataId": data_id,
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
            "dataId": data_id,
            "error": repr(e),
            "traceback": tb,
        }


def build_output_slots(target_total, test_ordinals, test_only, train_csv_path, test_csv_path):
    slots = []
    train_image_id = 0
    test_image_id = 0
    if test_only:
        for ordinal in range(int(target_total)):
            slots.append(
                {
                    "ordinal": ordinal,
                    "split": "test",
                    "image_id": test_image_id,
                    "csvpath": test_csv_path,
                }
            )
            test_image_id += 1
    else:
        for ordinal in range(int(target_total)):
            if ordinal in test_ordinals:
                slots.append(
                    {
                        "ordinal": ordinal,
                        "split": "test",
                        "image_id": test_image_id,
                        "csvpath": test_csv_path,
                    }
                )
                test_image_id += 1
            else:
                slots.append(
                    {
                        "ordinal": ordinal,
                        "split": "train",
                        "image_id": train_image_id,
                        "csvpath": train_csv_path,
                    }
                )
                train_image_id += 1
    return slots


def run_parallel_injection(
    repo,
    coll,
    save_path,
    number,
    trail_length,
    magnitude,
    beta,
    where,
    parallel=4,
    random_subset=0,
    train_test_split=0,
    seed=123,
    test_only=False,
    mag_mode="psf_mag",
    psf_template="image",
    stack_detection_threshold=5.0,
):
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
    train_target, test_target, test_ordinals = compute_split_targets(
        target_total, train_test_split, test_only, seed
    )
    final_target_total = test_target if test_only else target_total
    print(
        f"Target successful catalogs: total={final_target_total} train={train_target} test={test_target}",
        flush=True,
    )

    train_csv_path = os.path.join(save_path, "train.csv")
    test_csv_path = os.path.join(save_path, "test.csv")
    for path in (train_csv_path, test_csv_path):
        if os.path.exists(path):
            os.remove(path)

    slots = build_output_slots(final_target_total, test_ordinals, test_only, train_csv_path, test_csv_path)
    success_count = 0
    attempts = 0
    timing = {"compute_s": 0.0, "stage_s": 0.0, "promote_s": 0.0, "successes": 0, "failures": 0, "rss_max_mb": 0.0}
    manager = Manager()
    lock = manager.Lock()
    next_candidate_for_slot = [slot_idx for slot_idx in range(len(slots))]
    next_slot_to_activate = 0

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
            number,
            trail_length,
            magnitude,
            beta,
            seed,
            mag_mode,
            psf_template,
            stack_detection_threshold,
            slot["csvpath"],
            lock,
        ]

    def log_timing(tag="timing"):
        success_denom = max(1, timing["successes"])
        attempt_denom = max(1, attempts)
        print(
            f"[{tag}] attempts={attempts} successes={timing['successes']} failures={timing['failures']} "
            f"avg_compute={timing['compute_s']/attempt_denom:.2f}s "
            f"avg_stage={timing['stage_s']/success_denom:.2f}s "
            f"avg_promote={timing['promote_s']/success_denom:.2f}s "
            f"peak_rss={timing['rss_max_mb']:.1f}MB",
            flush=True,
        )

    if parallel > 1:
        max_workers = max(1, int(parallel))
        future_to_slot = {}

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            max_tasks_per_child=1,
        ) as ex:
            while next_slot_to_activate < len(slots) and len(future_to_slot) < max_workers:
                task = make_task(next_slot_to_activate)
                if task is None:
                    next_slot_to_activate += 1
                    continue
                future = ex.submit(worker, task)
                future_to_slot[future] = next_slot_to_activate
                next_slot_to_activate += 1

            while future_to_slot:
                done, _ = concurrent.futures.wait(
                    future_to_slot.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    slot_idx = future_to_slot.pop(future)
                    result = future.result()
                    attempts += 1
                    timing["compute_s"] += float(result.get("compute_s", 0.0))
                    if ATTEMPT_DIAGNOSTICS:
                        print(f"[diag] {format_attempt_diagnostics(result)}", flush=True)

                    if result["status"] == "ok":
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
                            retry_future = ex.submit(worker, retry_task)
                            future_to_slot[retry_future] = slot_idx

                if attempts % 25 == 0:
                    log_timing()

                if success_count >= final_target_total:
                    for future in future_to_slot:
                        future.cancel()
                    break

                while next_slot_to_activate < len(slots) and len(future_to_slot) < max_workers:
                    task = make_task(next_slot_to_activate)
                    if task is None:
                        next_slot_to_activate += 1
                        continue
                    future = ex.submit(worker, task)
                    future_to_slot[future] = next_slot_to_activate
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
        log_timing(tag="timing-final")


def main():
    ap = argparse.ArgumentParser("Build a deterministic point-source injection catalog")
    ap.add_argument("--repo", type=str, default="dp2_prep")
    ap.add_argument("--collections", type=str, default="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2")
    ap.add_argument("--save-path", default="../DATA/")
    ap.add_argument("--where", default="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') ")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--random-subset", type=int, default=10)
    ap.add_argument("--train-test-split", type=float, default=0.1)
    ap.add_argument("--trail-length-min", type=float, default=0.0)
    ap.add_argument("--trail-length-max", type=float, default=0.0)
    ap.add_argument("--mag-min", type=float, default=19)
    ap.add_argument("--mag-max", type=float, default=24)
    ap.add_argument("--mag-mode", choices=["psf_mag", "snr", "surface_brightness", "integrated_mag"], default="psf_mag")
    ap.add_argument("--psf-template", choices=["image", "kernel"], default="kernel")
    ap.add_argument("--beta-min", type=float, default=0)
    ap.add_argument("--beta-max", type=float, default=180)
    ap.add_argument("--number", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13379)
    ap.add_argument("--test-only", action="store_true", default=False)
    ap.add_argument("--stack-detection-threshold", type=float, default=5.0, help="SNR threshold used for stack detection")
    args = ap.parse_args()

    ensure_dir(args.save_path)
    logging.getLogger("lsst").setLevel(logging.ERROR)

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
        test_only=args.test_only,
        seed=args.seed,
        psf_template=args.psf_template,
        stack_detection_threshold=args.stack_detection_threshold,
    )


if __name__ == "__main__":
    main()
