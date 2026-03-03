from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List

import pandas as pd
from lsst.daf.butler import Butler


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Check preliminary_visit_image metadata and write bad_visits.csv")
    ap.add_argument("--repo", type=str, default="/repo/main")
    ap.add_argument("--collections", type=str, required=True, help="Butler collections string")
    ap.add_argument("--where", type=str, required=True, help="Butler registry where clause")
    ap.add_argument("--dataset", type=str, default="preliminary_visit_image")
    ap.add_argument("--instrument", type=str, default="LSSTComCam")
    ap.add_argument("--max-workers", type=int, default=0,
                    help="0 = auto (SLURM_CPUS_PER_TASK-1, else os.cpu_count()-1).")
    ap.add_argument("--out", type=str, default="bad_visits.csv")
    ap.add_argument("--show-progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="show_progress", action="store_false")
    return ap.parse_args()


def resolve_max_workers(requested: int) -> int:
    if requested and requested > 0:
        return requested

    # Prefer SLURM allocation if available
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            return max(1, n - 1)
        except ValueError:
            pass

    ncpu = os.cpu_count() or 1
    return max(1, ncpu - 1)


def list_pairs(repo: str, collections: str, dataset: str, instrument: str, where: str) -> List[Tuple[int, int]]:
    butler = Butler(repo, collections=collections)
    refs = list(set(butler.registry.queryDatasets(dataset, where=where, instrument=instrument, findFirst=True)))
    pairs = sorted({(int(r.dataId["visit"]), int(r.dataId["detector"])) for r in refs})
    return pairs


def check_pair(args: Tuple[str, str, str, int, int]) -> Optional[Tuple[int, int, str]]:
    repo, collections, instrument, visit, detector = args
    try:
        b = Butler(repo, collections=collections)
        calexp = b.get(
            "preliminary_visit_image",
            dataId={"instrument": instrument, "visit": int(visit), "detector": int(detector)},
        )
        wcs = calexp.wcs
        photocalib = calexp.getPhotoCalib()

        if wcs is None:
            return (int(visit), int(detector), "wcs_none")
        if photocalib is None:
            return (int(visit), int(detector), "photocalib_none")

        try:
            _ = wcs.getPixelScale()
            return None
        except Exception as e:
            return (int(visit), int(detector), f"getPixelScale_fail:{type(e).__name__}")

    except Exception as e:
        return (int(visit), int(detector), f"butler_get_fail:{type(e).__name__}")


def main() -> None:
    args = parse_args()
    max_workers = resolve_max_workers(args.max_workers)

    pairs = list_pairs(args.repo, args.collections, args.dataset, args.instrument, args.where)
    print(f"Pairs: {len(pairs)}  Using max_workers={max_workers}", flush=True)

    tasks = [(args.repo, args.collections, args.instrument, v, d) for (v, d) in pairs]

    bad: List[Tuple[int, int, str]] = []
    done = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(check_pair, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res is not None:
                bad.append(res)
            if args.show_progress:
                print(f"\rProcessed {done}/{total}  bad:{len(bad)}", end="", flush=True)

    if args.show_progress:
        print()

    bad_df = pd.DataFrame(bad, columns=["visit", "detector", "reason"])
    bad_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(bad_df)} rows", flush=True)


if __name__ == "__main__":
    main()
