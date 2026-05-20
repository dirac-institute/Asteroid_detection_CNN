#!/usr/bin/env python
"""Leakage-vetted real-empty-background dataset builder (EXPERIMENTAL).

Lives under experiments/ — imports the tracked, validated stack machinery
(ADCNN.data.dataset_creation.simulate_inject_diffim) and does NOT modify it.
Faithful adaptation of `run_parallel_injection` that:

  * picks refs via the tracked `select_good_refs_random_check`,
  * HARD-FILTERS every (visit,detector) against a leakage exclude-set,
  * runs the tracked `worker` (→ one_detector_injection) verbatim,
  * writes one flat {out}/data.h5 (gzip) + data.csv + panels.csv,
  * mode=trail → faint streaks (snr bulk-faint, wide length),
    mode=empty → number=0 zero-injection real-residual panels.

Outputs MUST go to the 80G scratch area (the 932G repo quota is full).
Run on the LSST stack env via slurm_build_realneg.sh.
"""
from __future__ import annotations

import argparse
import sys
import time
import concurrent.futures
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# simulate_inject_diffim uses cwd-relative `from common/pipetasks import ...`
# (sibling modules); make its dir importable so the package import resolves.
_DSC = f"{_REPO}/ADCNN/data/dataset_creation"
if _DSC not in sys.path:
    sys.path.insert(0, _DSC)

from ADCNN.data.dataset_creation.simulate_inject_diffim import (  # noqa: E402
    select_good_refs_random_check,
    worker,
)
from lsst.daf.butler import Butler  # noqa: E402

STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP = "lsst_cells_v2"
REPO = "dp2_prep"


def _pairs(csv_path: str) -> set[tuple[int, int]]:
    p = Path(csv_path)
    if not p.exists():
        print(f"[exclude] MISSING (skipped): {csv_path}", flush=True)
        return set()
    df = pd.read_csv(p, usecols=lambda c: c in ("visit", "detector"))
    if not {"visit", "detector"} <= set(df.columns):
        print(f"[exclude] no visit/detector cols in {csv_path}", flush=True)
        return set()
    s = {(int(v), int(d)) for v, d in zip(df["visit"], df["detector"])}
    print(f"[exclude] {len(s):>6} pairs from {csv_path}", flush=True)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["trail", "empty"])
    ap.add_argument("--out", required=True, help="output dir (scratch)")
    ap.add_argument("--where", required=True)
    ap.add_argument("--n-panels", type=int, required=True)
    ap.add_argument("--parallel", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260519)
    ap.add_argument("--oversample", type=float, default=3.0)
    ap.add_argument("--exclude-csv", nargs="*", default=[
        f"{_REPO}/DATA_DIFFIM/test_real/panels.csv",
        f"{_REPO}/DATA_DIFFIM/train.csv",
        f"{_REPO}/DATA_DIFFIM/test.csv",
    ])
    ap.add_argument("--also-exclude", nargs="*", default=[])
    ap.add_argument("--number", type=int, default=20)
    ap.add_argument("--snr-min", type=float, default=3.0)
    ap.add_argument("--snr-max", type=float, default=10.0)
    ap.add_argument("--len-min", type=float, default=4.0)
    ap.add_argument("--len-max", type=float, default=200.0)
    ap.add_argument("--stack-threshold", type=float, default=5.0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    coll = [STAGE3, STAGE2]
    where = f"instrument='LSSTCam' AND {a.where} AND band in ('u','g','r','i','z','y')"

    exclude: set[tuple[int, int]] = set()
    for c in a.exclude_csv:
        exclude |= _pairs(c)
    for c in a.also_exclude:
        exclude |= _pairs(c)
    print(f"[exclude] TOTAL distinct excluded (visit,detector): {len(exclude)}",
          flush=True)

    pool_k = int(a.n_panels * a.oversample)
    print(f"[refs] select_good_refs_random_check k={pool_k} where={where}",
          flush=True)
    refs = select_good_refs_random_check(
        repo=REPO, collections=coll, where=where, skymap=SKYMAP,
        stage3_collection=STAGE3, instrument="LSSTCam",
        k=pool_k, seed=a.seed, pool_size=8000, max_checks=400000,
        check_refs=True, verbose=False,
    )
    print(f"[refs] pool returned {len(refs)}", flush=True)

    kept, seen, leaked = [], set(), 0
    for r in refs:
        v, d = int(r.dataId["visit"]), int(r.dataId["detector"])
        if (v, d) in exclude:
            leaked += 1
            continue
        if (v, d) in seen:
            continue
        seen.add((v, d))
        kept.append(r)
        if len(kept) >= a.n_panels:
            break
    print(f"[refs] vetted: kept={len(kept)} dropped_leak={leaked} "
          f"(want {a.n_panels})", flush=True)
    if not kept:
        sys.exit("[refs] FATAL: 0 refs after leakage vetting")

    chosen = pd.DataFrame(
        [(int(r.dataId["visit"]), int(r.dataId["detector"])) for r in kept],
        columns=["visit", "detector"])
    assert set(map(tuple, chosen.values)).isdisjoint(exclude), \
        "LEAKAGE: vetted refs intersect the exclude set"
    chosen.to_csv(out / "chosen_pairs.csv", index=False)
    print(f"[refs] wrote {out}/chosen_pairs.csv ({len(chosen)} pairs); "
          f"disjointness asserted OK", flush=True)

    butler = Butler(REPO, collections=coll)
    dims = butler.get("preliminary_visit_image.dimensions",
                      dataId=kept[0].dataId)
    h5path = str(out / "data.h5")
    csvpath = str(out / "data.csv")
    chunks = (1, min(128, dims.y), min(128, dims.x))
    # gzip+shuffle (tracked builder's test path) — uncompressed 4k float32
    # panels are ~35GB/400; compression cuts ~4-5x (scratch is only 80G).
    _z = dict(compression="gzip", compression_opts=4, shuffle=True)
    with h5py.File(h5path, "w") as f:
        f.create_dataset("images", shape=(len(kept), dims.y, dims.x),
                         dtype="float32", chunks=chunks, **_z)
        f.create_dataset("masks", shape=(len(kept), dims.y, dims.x),
                         dtype="bool", chunks=chunks, **_z)
        f.create_dataset("real_labels", shape=(len(kept), dims.y, dims.x),
                         dtype="uint16", chunks=chunks, **_z)

    number = 0 if a.mode == "empty" else a.number
    trail_length = [a.len_min, a.len_max]
    magnitude = [a.snr_min, a.snr_max]   # snr mode reads mag=(snr_min,snr_max)
    beta = [0.0, 180.0]

    mgr = Manager()
    lock = mgr.Lock()
    tasks = []
    for idx, r in enumerate(kept):
        tasks.append([idx, r.dataId, REPO, coll, dims, lock, h5path, csvpath,
                      number, trail_length, magnitude, beta,
                      "preliminary_visit_image", a.seed, "snr", "kernel",
                      a.stack_threshold, False, SKYMAP, STAGE3])

    print(f"[build] mode={a.mode} number={number} panels={len(tasks)} "
          f"snr={magnitude} len={trail_length} par={a.parallel}", flush=True)
    ok = err = 0
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                res = fut.result()
            except BaseException as e:  # noqa: BLE001
                err += 1
                print(f"[{n}/{len(tasks)}] CRASH {type(e).__name__}: {e}",
                      flush=True)
                continue
            if res[0] == "ok":
                ok += 1
            else:
                err += 1
            if n % 20 == 0 or n == len(tasks):
                print(f"[{n}/{len(tasks)}] ok={ok} err={err} "
                      f"{time.time()-t0:.0f}s", flush=True)

    rows = []
    for idx, r in enumerate(kept):
        di = r.dataId
        rows.append({"image_id": idx, "visit": int(di["visit"]),
                     "detector": int(di["detector"]),
                     "band": di.get("band", ""),
                     "role": "empty" if a.mode == "empty" else "trail"})
    pd.DataFrame(rows).to_csv(out / "panels.csv", index=False)
    print(f"[done] mode={a.mode} ok={ok} err={err} -> {h5path} "
          f"(+ data.csv, panels.csv, chosen_pairs.csv)", flush=True)


if __name__ == "__main__":
    main()
