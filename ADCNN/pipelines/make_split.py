"""Create ONE leakage-safe panel split — train / train2 / test / val — all at once.

The three ADCNN datasets must never share a (visit, detector) panel: v7 trains on TRAIN, the stage-2
focal cutout CNN trains on TRAIN2, and everything is evaluated on TEST. The old per-dataset flow
enforced this only partially — each set was built with ``--exclude-pairs-csv <test>``, so train and
train2 were kept off the test panels but **not off each other** (their mutual disjointness rested only
on using different injection seeds, which does not guarantee disjoint panels).

This module fixes that by partitioning the panel universe **once**, up front, into mutually-disjoint
sets and writing a single ``split.json``. Each dataset is then generated from its own slice, so the
three are disjoint by construction.

Flow:
  1. Get the candidate (visit, detector) universe for the field as a CSV with ``visit,detector``
     columns (e.g. from ``experiments/heliolinc/butler_manifest.py`` or any ref dump for the region
     you will inject into).
  2. Partition it::

       python -m ADCNN.pipelines.make_split --refs refs.csv --out DATA_DIFFIM/split.json \\
           --train -1 --train2 500 --test 300 --val 64 --seed 0      # train=-1 -> the remainder

  3. Build each dataset from its slice (the disjointness is enforced by excluding the other sets'
     panels, see ``ADCNN.data.dataset_creation.simulate`` ``--split-json``/``--split-key``)::

       python -m ADCNN.pipelines.make_sim_data --split-json DATA_DIFFIM/split.json --split-key train  ...
       python -m ADCNN.pipelines.make_sim_data --split-json DATA_DIFFIM/split.json --split-key train2 ...
       python -m ADCNN.pipelines.make_sim_data --split-json DATA_DIFFIM/split.json --split-key test --test-only ...

``split.json`` records, per set, the list of ``[visit, detector]`` pairs plus the seed and sizes, so
the partition is reproducible and auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Scarce / held-out sets are allocated first; train takes whatever remains.
_ALLOC_ORDER = ("test", "val", "train2", "train")


def make_split(refs_csv: str, sizes: dict[str, int], seed: int = 0) -> dict[str, list[tuple[int, int]]]:
    """Partition the unique (visit, detector) pairs in ``refs_csv`` into disjoint named sets.

    ``sizes`` maps set name -> panel count; a size of -1 (or None) means "the remainder" and is
    allowed for exactly one set (typically ``train``). Raises if the requested counts exceed the
    available panels. Returns ``{name: [(visit, detector), ...]}`` with no panel in two sets.
    """
    df = pd.read_csv(refs_csv)
    if not {"visit", "detector"} <= set(df.columns):
        raise SystemExit(f"{refs_csv} must have 'visit' and 'detector' columns")
    pairs = [(int(v), int(d)) for v, d in df[["visit", "detector"]].drop_duplicates().to_numpy()]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)

    remainder_keys = [k for k in sizes if sizes.get(k) in (-1, None)]
    if len(remainder_keys) > 1:
        raise SystemExit(f"at most one set may be the remainder (-1); got {remainder_keys}")
    fixed_total = sum(int(n) for n in sizes.values() if n not in (-1, None))
    if fixed_total > len(pairs):
        raise SystemExit(f"requested {fixed_total} panels > {len(pairs)} available in {refs_csv}")

    out: dict[str, list[tuple[int, int]]] = {}
    i = 0
    for key in list(_ALLOC_ORDER) + [k for k in sizes if k not in _ALLOC_ORDER]:
        if key not in sizes:
            continue
        n = sizes[key]
        if n in (-1, None):
            out[key] = pairs[i:]; i = len(pairs)
        else:
            out[key] = pairs[i:i + int(n)]; i += int(n)
    # disjointness is guaranteed by construction (contiguous slices of a shuffled list); assert it.
    allp = [p for v in out.values() for p in v]
    assert len(allp) == len(set(allp)), "BUG: split sets overlap"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", required=True, help="CSV of the candidate panel universe (visit,detector cols)")
    ap.add_argument("--out", required=True, help="output split.json")
    ap.add_argument("--train", type=int, default=-1, help="train panel count (-1 = the remainder)")
    ap.add_argument("--train2", type=int, default=500, help="stage-2 CNN train panel count")
    ap.add_argument("--test", type=int, default=300, help="test panel count")
    ap.add_argument("--val", type=int, default=64, help="held-out val panel count (v7 selection + stage-2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sizes = {"train": a.train, "train2": a.train2, "test": a.test, "val": a.val}
    split = make_split(a.refs, sizes, a.seed)
    meta = {"seed": a.seed, "refs": str(Path(a.refs).resolve()),
            "sizes": {k: len(v) for k, v in split.items()},
            **{k: [list(p) for p in v] for k, v in split.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(meta, indent=2))
    print("split -> " + a.out + ": " + ", ".join(f"{k}={len(v)}" for k, v in split.items())
          + f" (total {sum(len(v) for v in split.values())} panels, seed {a.seed})")


if __name__ == "__main__":
    main()
