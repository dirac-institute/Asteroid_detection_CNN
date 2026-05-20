#!/usr/bin/env python
"""Merge the trail + empty realneg builds into one trainer-ready dataset.

{trail}/data.h5 + {empty}/data.h5 -> {out}/train.h5 (+ train.csv panels.csv)

* image_id is re-offset so empty panels follow trail panels contiguously.
* Variable PVI dims are zero-padded to the max, origin-aligned (same
  convention as build_test_real).
* train.csv = trail injection catalog with re-offset image_id (empty panels
  contribute no rows — they are negative-only panels by construction).
* panels.csv = image_id, visit, detector, band, role  (role∈{trail,empty}).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


def _load_panels(d: Path) -> pd.DataFrame:
    return pd.read_csv(d / "panels.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trail", required=True)
    ap.add_argument("--empty", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trail, empty, out = Path(a.trail), Path(a.empty), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pt, pe = _load_panels(trail), _load_panels(empty)
    with h5py.File(trail / "data.h5", "r") as ft, \
         h5py.File(empty / "data.h5", "r") as fe:
        nt, ht, wt = ft["images"].shape
        ne, he, we = fe["images"].shape
        N, H, W = nt + ne, max(ht, he), max(wt, we)
        print(f"[merge] trail={nt}x{ht}x{wt} empty={ne}x{he}x{we} "
              f"-> {N}x{H}x{W}", flush=True)
        # gzip+shuffle: uncompressed train.h5 would be ~60GB on 600x4004x4096
        # float32 + masks + real_labels — won't fit the 100G scratch quota.
        _z = dict(compression="gzip", compression_opts=4, shuffle=True)
        with h5py.File(out / "train.h5", "w") as fo:
            fo.create_dataset("images", (N, H, W), "float32",
                              chunks=(1, min(128, H), min(128, W)), **_z)
            fo.create_dataset("masks", (N, H, W), "bool",
                              chunks=(1, min(128, H), min(128, W)), **_z)
            fo.create_dataset("real_labels", (N, H, W), "uint16",
                              chunks=(1, min(128, H), min(128, W)), **_z)
            for src, n, off in ((ft, nt, 0), (fe, ne, nt)):
                for i in range(n):
                    j = off + i
                    img = src["images"][i]
                    h, w = img.shape
                    fo["images"][j, :h, :w] = img
                    fo["masks"][j, :h, :w] = src["masks"][i]
                    fo["real_labels"][j, :h, :w] = src["real_labels"][i]
                    if (i + 1) % 50 == 0:
                        print(f"  copied {off+i+1}/{N}", flush=True)

    # train.csv: trail injection catalog, image_id unchanged (trail panels
    # are 0..nt-1); empty panels (nt..N-1) intentionally absent.
    cat = trail / "data.csv"
    if cat.exists() and cat.stat().st_size > 0:
        df = pd.read_csv(cat)
        df.to_csv(out / "train.csv", index=False)
        print(f"[merge] train.csv rows={len(df)} "
              f"(img_id range {df.image_id.min()}..{df.image_id.max()})",
              flush=True)
    else:
        pd.DataFrame(columns=["image_id"]).to_csv(out / "train.csv",
                                                  index=False)
        print("[merge] WARNING trail data.csv empty/missing", flush=True)

    pe = pe.copy()
    pe["image_id"] = pe["image_id"] + nt
    pan = pd.concat([pt, pe], ignore_index=True)
    pan.to_csv(out / "panels.csv", index=False)
    print(f"[merge] panels.csv N={len(pan)} "
          f"(trail={int((pan.role=='trail').sum())} "
          f"empty={int((pan.role=='empty').sum())}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
