#!/usr/bin/env python3
"""Cull a heliolinc pass and run link_purify SHARDED -- the serial-fitter fix.

The slow arm measured the problem: stock link_purify is single-core and spent ~13 h on 262k
clusters. This shards the culled cluster set into N disjoint subsets and runs N link_purify
processes in parallel (~N x speedup). Sharding is safe because purify's cross-cluster logic is
dedup/overlap resolution: clusters of the SAME object land in the same shard by sorting on the
cluster position state (posX bin) before splitting, and the residual cross-shard overlap is
resolved at scoring by member-set intersection (as the miners already do).

Also repairs the heliolinc_omp writer bug (missing orbit_incl column) and renumbers clusters
sequentially per shard (read_clustersum_file indexes by value).

    python -m ADCNN.analysis.multinight.cull_and_purify <tag>     # e.g. fastA
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HLX = REPO / "external/heliolinx/bin"
W = REPO / "outputs/runs/multinight/full/work"
CULL_CAP = 250_000
NSHARD = 16


def main(tag):
    sumf = W / f"hl_{tag}_neo_n2_sum.csv"
    c2df = W / f"hl_{tag}_neo_n2_c2d.csv"
    s = pd.read_csv(sumf)
    s.columns = [c.lstrip("#") for c in s.columns]
    if "orbit_incl" not in s.columns:          # heliolinc_omp writer bug
        s.insert(list(s.columns).index("orbit_e") + 1, "orbit_incl", 0.0)
    n0 = len(s)
    s = s[(s.uniquepoints >= 5) & (s.obsnights >= 2) & (s.posRMS > 0)]
    # RANK BY COMPACTNESS, not by heliolinc's metric. MEASURED on fastA: the metric's top-250k
    # have posRMS median 62,528 km (loose megaclusters; ALL 250k failed the orbit fit) while the
    # smallest-posRMS 250k are 5-6-point clusters at 16-30k km -- the size and tightness REAL
    # objects produce. The metric rewards point count, which at fast-band density is a junk proxy.
    s = s.nsmallest(min(len(s), CULL_CAP), "posRMS")
    print(f"[cp:{tag}] cull {n0:,} -> {len(s):,}", flush=True)
    c = pd.read_csv(c2df)
    c.columns = [cc.lstrip("#") for cc in c.columns]
    keep = set(s.clusternum.astype(int))
    c = c[c.clusternum.astype(int).isin(keep)]
    # shard by position state so same-object clusters co-locate
    s = s.sort_values(["posX", "posY"]).reset_index(drop=True)
    s["shard"] = (np.arange(len(s)) * NSHARD) // max(len(s), 1)
    c2 = c.merge(s[["clusternum", "shard"]], on="clusternum", how="inner")
    procs = []
    for k in range(NSHARD):
        sk = s[s.shard == k].drop(columns=["shard"]).reset_index(drop=True)
        if not len(sk):
            continue
        remap = {int(o): n for n, o in enumerate(sk.clusternum.astype(int))}
        sk["clusternum"] = range(len(sk))
        ck = c2[c2.shard == k].drop(columns=["shard"]).copy()
        ck["clusternum"] = ck.clusternum.astype(int).map(remap)
        ck = ck.sort_values(["clusternum", "detnum"])
        base = W / f"shard_{tag}_{k:02d}"
        for df, path in ((sk, f"{base}_sum.csv"), (ck, f"{base}_c2d.csv")):
            cols = list(df.columns); cols[0] = "#" + cols[0]
            df.to_csv(path, index=False, header=cols)
        Path(f"{base}_lf.txt").write_text(f"{base}_sum.csv {base}_c2d.csv\n")
        p = subprocess.Popen(
            [str(HLX / "link_purify"), "-imgs", str(W / f"img_{tag}.csv"),
             "-pairdet", str(W / f"pd_{tag}.csv"), "-lflist", f"{base}_lf.txt",
             "-minobsnights", "2", "-minpointnum", "5", "-maxrms", "200000",
             "-max_astrom_rms", "1.0", "-rejfrac", "0.2", "-rejnum", "2",
             "-outsum", f"{base}_LPL.csv", "-clust2det", f"{base}_LPLc2d.csv"],
            stdout=open(f"{base}.log", "w"), stderr=subprocess.STDOUT)
        procs.append((k, p))
    print(f"[cp:{tag}] {len(procs)} purify shards launched", flush=True)
    bad = 0
    for k, p in procs:
        rc = p.wait()
        n = sum(1 for _ in open(W / f"shard_{tag}_{k:02d}_LPL.csv")) - 1 \
            if (W / f"shard_{tag}_{k:02d}_LPL.csv").exists() else -1
        print(f"[cp:{tag}] shard {k:02d}: rc={rc} linkages={n}", flush=True)
        bad += rc != 0
    # merge shards
    frames = []
    for k, _ in procs:
        f = W / f"shard_{tag}_{k:02d}_LPL.csv"
        if f.exists() and f.stat().st_size > 0:
            d = pd.read_csv(f)
            d.columns = [cc.lstrip("#") for cc in d.columns]
            if len(d):
                d["shard"] = k
                frames.append(d)
    m = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    m.to_csv(W / f"LPL_{tag}.csv", index=False)
    print(f"[cp:{tag}] merged {len(m):,} purified linkages -> LPL_{tag}.csv ({bad} shard failures)",
          flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "fastA"))
