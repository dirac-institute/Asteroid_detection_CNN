#!/usr/bin/env python3
"""Score the full-detection multinight campaign: slow + fastA + fastB purified linkages.

Per arm: night-span tiers, astrometric quality, orbit-element sanity, member composition
(stack vs adcnn via the detection IDs), a SkyBoT sample for catalogued fraction (sanity +
novelty triage only -- see known-matches-not-a-purity-proxy), and the MN-2026-01 recovery
control (its five detections, by position/time, searched in RAW and PURIFIED fastB clusters).

Writes outputs/runs/multinight/full/{linkages_<arm>.csv, summary.json} and prints the report.

    python -m ADCNN.analysis.multinight.score_campaign [--skybot N]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
W = REPO / "outputs/runs/multinight/full/work"
OUT = REPO / "outputs/runs/multinight/full"

# MN-2026-01 members: (mjd, ra, dec) of the five detections (dossier, candidates_3night/README)
MN01 = [(61228.3214, None, None)]  # filled from the dossier alert records at runtime


def load_mn01():
    d = json.load(open(REPO / "outputs/runs/multinight/candidates_3night/candidates/cand_001/members.json")) \
        if (REPO / "outputs/runs/multinight/candidates_3night/candidates/cand_001/members.json").exists() else None
    pts = []
    for night, aid in (("20260706", "2v_61227_007109"), ("20260709", "2v_61230_000009")):
        for a in map(json.loads, open(REPO / f"outputs/runs/10k_cadence/run_night_{night}/work/stream/alerts.jsonl")):
            if a["alertId"] == aid:
                pts += [(e["mjd"], e["ra"], e["dec"]) for e in a["epochs"]]
                break
    pts.append((61230.29406, 320.000615, -9.166057))      # the 0708 single
    return pts


def members(tag, which="LPL"):
    """-> DataFrame(clusternum, mjd, ra, dec, idstr) for purified (LPL, shard-merged) or raw."""
    pdet = pd.read_csv(W / f"pd_{tag}.csv", low_memory=False)
    pdet.columns = [c.lstrip("#") for c in pdet.columns]
    if which == "raw":
        c2d = pd.read_csv(W / f"hl_{tag}_neo_n2_c2d.csv")
        c2d.columns = [c.lstrip("#") for c in c2d.columns]
        c2d["key"] = c2d.clusternum.astype(int)
    else:
        frames = []
        for f in sorted(W.glob(f"shard_{tag}_*_LPLc2d.csv")):
            k = int(f.name.split("_")[-2])
            d = pd.read_csv(f); d.columns = [c.lstrip("#") for c in d.columns]
            d["key"] = d.clusternum.astype(int) * 1000 + k        # (shard, cluster) -> unique
            frames.append(d)
        c2d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["detnum", "key"])
    m = c2d.merge(pdet[["MJD", "RA", "Dec", "idstring"]], left_on=c2d.detnum.astype(int),
                  right_index=True, how="left")
    return m


def control_check(tag, which):
    pts = load_mn01()
    m = members(tag, which)
    hit_keys = set()
    per_pt = []
    for mj, ra, dec in pts:
        d = np.hypot((m.RA - ra) * np.cos(np.radians(dec)), m.Dec - dec) * 3600
        sel = m[(d < 2.0) & (abs(m.MJD - mj) < 1e-3)]
        per_pt.append(len(sel))
        hit_keys |= set(sel.key)
    # a cluster containing >=3 of the 5 control points = recovery
    counts = {}
    for mj, ra, dec in pts:
        d = np.hypot((m.RA - ra) * np.cos(np.radians(dec)), m.Dec - dec) * 3600
        for k in set(m[(d < 2.0) & (abs(m.MJD - mj) < 1e-3)].key):
            counts[k] = counts.get(k, 0) + 1
    best = max(counts.values()) if counts else 0
    return {"points_present": per_pt, "clusters_touching": len(hit_keys),
            "best_cluster_points": best, "recovered": best >= 3}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skybot", type=int, default=0, help="sample N linkages per arm for SkyBoT")
    ap.add_argument("--control-only", action="store_true")
    a = ap.parse_args(argv)
    rep = {}
    if (W / "hl_fastB_neo_n2_c2d.csv").exists():
        rep["control_fastB_raw"] = control_check("fastB", "raw")
        print("[score] MN-2026-01 in RAW fastB:", rep["control_fastB_raw"], flush=True)
    if list(W.glob("shard_fastB_*_LPLc2d.csv")):
        rep["control_fastB_purified"] = control_check("fastB", "LPL")
        print("[score] MN-2026-01 in PURIFIED fastB:", rep["control_fastB_purified"], flush=True)
    if a.control_only:
        (OUT / "summary.json").write_text(json.dumps(rep, indent=2)); return 0
    for tag in ("slow", "fastA", "fastB"):
        f = W / f"LPL_{tag}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f); s.columns = [c.lstrip("#") for c in s.columns]
        if not len(s):
            rep[tag] = {"n": 0}; continue
        r = {"n": int(len(s)),
             "obsnights": {int(k): int(v) for k, v in s.obsnights.value_counts().sort_index().items()},
             "astromRMS_med": round(float(s.astromRMS.median()), 3),
             "uniquepoints_med": float(s.uniquepoints.median()),
             "timespan_med_d": round(float(s.timespan.median()), 1),
             "orbit_a_p10_50_90": [round(float(x), 2) for x in s.orbit_a.quantile([.1, .5, .9])],
             "orbit_e_med": round(float(s.orbit_e.median()), 2)}
        try:
            m = members(tag, "LPL")
            m["src"] = np.where(m.idstring.astype(str).str.contains("stack"), "stack", "adcnn")
            comp = m.groupby("key").src.agg(lambda x: "stack" if (x == "stack").all() else
                                            ("adcnn" if (x == "adcnn").all() else "mixed"))
            r["composition"] = {k: int(v) for k, v in comp.value_counts().items()}
        except Exception as e:
            r["composition"] = f"n/a ({type(e).__name__})"
        if a.skybot:
            from ADCNN.pipelines.heliolinc.mpc_crossmatch import skybot_conesearch
            m = members(tag, "LPL")
            keys = list(dict.fromkeys(m.key))[: a.skybot]
            kn = new = err = 0
            for k in keys:
                row = m[m.key == k].iloc[0]
                try:
                    cat = skybot_conesearch(float(row.RA), float(row.Dec), float(row.MJD) + 2400000.5, 0.01, "X05")
                    hit = any(np.hypot((float(c["ra"]) - row.RA) * np.cos(np.radians(row.Dec)),
                                       float(c["dec"]) - row.Dec) * 3600 < 10 for c in cat)
                    kn += hit; new += (not hit)
                except Exception:
                    err += 1
                time.sleep(0.4)
            r["skybot_sample"] = {"known": kn, "uncatalogued": new, "err": err}
        rep[tag] = r
        print(f"[score] {tag}: {r}", flush=True)
        s.to_csv(OUT / f"linkages_{tag}.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
