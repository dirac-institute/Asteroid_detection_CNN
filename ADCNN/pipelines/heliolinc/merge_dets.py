#!/usr/bin/env python3
"""Merge detection catalogues from several passes over one night into a single dets CSV.

Needed because a GPU shard that dies (uncorrectable ECC killed one on five separate ada nodes in
one campaign) leaves a night partially detected. The fix is to re-detect only the missing panels
into a side directory and merge -- but the pieces are not trivially concatenable:

  * SCHEMA. Per-shard files (`_shard_adcnn_dets_N.csv`) lack the `detid` column that detect_night
    adds when it assembles shards into `adcnn_dets.csv`, so a naive `cat` yields rows with 22 and
    23 fields and pandas fails at the first assembled part.
  * DUPLICATES. Passes can overlap (a shard may have written rows for panels a later pass also
    covered), and a duplicated detection would be linkable against itself.

So: align on the common columns, drop any pre-existing detid, de-duplicate on
(visit, detector, x, y), and re-issue detid over the merged set.

Usage:
  python -m ADCNN.pipelines.heliolinc.merge_dets --out run/adcnn_dets.csv \
      run/_shard_adcnn_dets_*.csv run_fill/adcnn_dets.csv
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

_REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

KEY = ["visit", "detector", "x", "y"]


def merge(parts, out_path):
    frames = []
    for p in parts:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        d = pd.read_csv(p, low_memory=False)
        if "detid" in d.columns:
            d = d.drop(columns=["detid"])
        frames.append(d)
        print(f"  {p}: {len(d):,} rows, {d.shape[1]} cols", flush=True)
    if not frames:
        raise SystemExit("merge_dets: no non-empty inputs")
    cols = frames[0].columns.tolist()
    bad = [i for i, f in enumerate(frames) if sorted(f.columns) != sorted(cols)]
    if bad:
        raise SystemExit(f"merge_dets: column mismatch beyond detid in input(s) {bad}")
    d = pd.concat([f[cols] for f in frames], ignore_index=True)
    n0 = len(d)
    d = d.drop_duplicates(subset=KEY, keep="first")
    d.insert(0, "detid", range(len(d)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_path, index=False)
    print(f"merged {n0:,} -> {len(d):,} unique detections over "
          f"{d.groupby(['visit','detector']).ngroups:,} panels -> {out_path}", flush=True)
    return len(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args(argv)
    merge(a.parts, a.out)


if __name__ == "__main__":
    sys.exit(main())
