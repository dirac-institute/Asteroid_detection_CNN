#!/usr/bin/env python3
"""Build the FULL-detection multinight catalogue: every ADCNN + stack detection, ten nights.

User directive (2026-08-19): all detections enter the linker, not just alerts. This assembles
work/dets_merged.csv from all ten nights into one heliolinx-format detection table with
per-visit dedup (a stack row within 1" of an ADCNN row in the SAME visit is the same physical
source; keeping both would let one source appear twice in a tracklet). Slow-bright movers the
stack alone sees have no ADCNN counterpart and therefore survive dedup by construction.

Columns: ID,MJD,RA,Dec,mag,band,obscode,score,src,len_db -- heliolinx reads the first seven via
colformat; score/src/len_db ride along for arm filtering and post-hoc scoring.

    python -m ADCNN.analysis.multinight.build_full_catalog
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "outputs/runs/10k_cadence"
W = REPO / "outputs/runs/multinight/full/work"

USE = ["detid", "mjd", "ra", "dec", "mag", "visit", "score", "len_db", "src", "art_frac"]


def main():
    W.mkdir(parents=True, exist_ok=True)
    out = open(W / "dets_all.csv", "w")
    out.write("#ID,MJD,RA,Dec,mag,band,obscode,score,src,len_db\n")
    n_tot = n_dup = 0
    for f in sorted(glob.glob(str(RUNS / "run_night_*/work/dets_merged.csv"))):
        night = f.split("run_night_")[1][:8]
        t = pd.read_csv(f, usecols=USE, low_memory=False)
        t = t[t.art_frac.fillna(0) < 0.5].reset_index(drop=True)   # labels == positions below
        is_stk = t.src.astype(str).eq("stack")
        drop = np.zeros(len(t), bool)
        # per-visit dedup: stack row within 1" of an ADCNN row in the same visit = same source
        for v, g in t.groupby("visit"):
            gs = g[is_stk.loc[g.index]]
            ga = g[~is_stk.loc[g.index]]
            if not len(gs) or not len(ga):
                continue
            cd = np.cos(np.radians(g.dec.mean()))
            tree = cKDTree(np.c_[ga.ra * cd, ga.dec])
            d, _ = tree.query(np.c_[gs.ra * cd, gs.dec], distance_upper_bound=1.0 / 3600)
            drop[gs.index[np.isfinite(d)]] = True
        keep = t[~drop]
        n_dup += int(drop.sum()); n_tot += len(keep)
        mag = keep.mag.fillna(22.5)
        ldb = keep.len_db.fillna(-1)
        for row in zip(keep.detid.astype(str), keep.mjd, keep.ra, keep.dec, mag,
                       keep.score.fillna(-1), keep.src.astype(str), ldb):
            out.write(f"{night}_{row[0]},{row[1]:.7f},{row[2]:.7f},{row[3]:.7f},"
                      f"{row[4]:.2f},r,X05,{row[5]:.4f},{row[6]},{row[7]:.1f}\n")
        print(f"[full] {night}: kept {len(keep):,} ({int(drop.sum()):,} stack dups dropped)",
              flush=True)
    out.close()
    (W / "colformat.txt").write_text(
        "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
    print(f"[full] TOTAL {n_tot:,} detections ({n_dup:,} dups removed) -> {W}/dets_all.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
