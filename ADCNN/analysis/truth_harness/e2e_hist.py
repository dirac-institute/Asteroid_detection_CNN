#!/usr/bin/env python3
"""END-TO-END delivered-at-1k completeness for ADCNN, the stack, and the merged product.

This is the number the product actually delivers -- injected mover -> detected -> linked -> survives
the full 1k op -> lands in the top 1000 by priorityScore. It is NOT the detection ceiling, which runs
~49% here and is the subject of completeness_hist.py.

THREE ARMS, ONE VARIABLE. All three detection catalogues come from a SINGLE merge_dets run and are
split by `src`, so the ADCNN arm is ring-cleaned exactly as the merged arm's ADCNN half is; and all
three were linked with identical settings AND an identical visit-pair set. That last point mattered:
the co-pointed filter keys on each visit's MEDIAN detection position, so adding stack detections
shifts the centroids and silently changes which visit-pairs are linked at all (14 / 12 / 11 at the
2.0 deg default). All 10 visit-pairs the truth actually uses pass in every arm, so completeness was
never confounded -- but the extra pairs carry no injections and contribute only false positives,
which compete for the fixed 1000 slots. At --max-visit-sep-deg 1.0 all three arms link the same 10.

TRAIL-AWARE ALERT MATCHING. compare_variants_1k.alert_oids uses a FIXED 3" radius, which under-counts
exactly the fast movers this comparison is about: at 0.2"/px a 50 px trail is 10" long, so an alert
positioned anywhere but the trail's middle can sit 5" from the catalogued centre. Here an alert epoch
matches an object if it lands within `perp` of that object's true trail SEGMENT in that visit, and an
object counts as delivered only if BOTH epochs of one alert match the SAME object.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

V = "outputs/runs/pa_validate"
OP = "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json"
REFCAT = "outputs/runs/10k_cadence/run_night_20260706/bright_refcat.parquet"
BUDGET = 1000
PERP = 1.0
PIX = 0.2
L_EDGES = [0, 8, 12, 16, 24, 32, 44, 60]
S_EDGES = [0, 3, 4, 5, 6, 8, 10, 99]
ARMS = ["ADCNN", "stack", "merged"]
TAG = {"ADCNN": "adcnn", "stack": "stack", "merged": "merged"}


def deliver(tag):
    """Full 1k op, then the top-BUDGET by priorityScore. Returns the delivered alerts."""
    src, surv = f"{V}/e2e_alerts_{tag}.jsonl", f"{V}/e2e_surv_{tag}.jsonl"
    r = subprocess.run([sys.executable, "-m", "ADCNN.qa.filter_op", "--alerts", src,
                        "--dets", f"{V}/e2e_dets_{tag}.csv", "--op", OP, "--out", surv,
                        "--refcat", REFCAT, "--allow-unranked"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    if r.returncode != 0:
        raise SystemExit(f"filter_op failed for {tag}:\n{r.stdout}\n{r.stderr}")
    al = [json.loads(l) for l in open(surv)]
    n_link = sum(1 for _ in open(src))
    # chi2 tiebreak: priorityScore is heavily tied, so an unstable cut would be a hidden variable
    al.sort(key=lambda a: (-(a.get("priorityScore") or 0.0),
                           float(((a.get("orbit") or {}).get("chi2")) or 1e9)))
    return al[:BUDGET], n_link, len(al)


def delivered_oids(alerts, T, perp=PERP):
    """oids whose trail BOTH epochs of one alert land on. Segment distance, not a fixed radius."""
    half = T.L_px.to_numpy() * PIX / 2.0
    pa = np.radians(T.pa.to_numpy())
    idx, ctr = {}, {}
    for ep in ("A", "B"):
        for v, g in T.groupby(f"visit{ep}"):
            pos = T.index.get_indexer(g.index)
            cd = np.cos(np.radians(g[f"dec{ep}"].to_numpy()))
            ctr[(int(v), ep)] = (g[f"ra{ep}"].to_numpy() * cd * 3600.0,
                                 g[f"dec{ep}"].to_numpy() * 3600.0, pos, cd)
            idx[(int(v), ep)] = cKDTree(np.c_[g[f"ra{ep}"].to_numpy() * cd, g[f"dec{ep}"].to_numpy()])
    rmax = (half.max() + perp + 0.5) / 3600.0
    out = set()
    for a in alerts:
        eps = a.get("epochs") or []
        if len(eps) < 2:
            continue
        hits = []
        for e in eps:
            got = -1
            for ep in ("A", "B"):
                key = (int(e["visit"]), ep)
                if key not in idx:
                    continue
                cx, cy, pos, cd = ctr[key]
                cdq = np.cos(np.radians(e["dec"]))
                for k in idx[key].query_ball_point([e["ra"] * cdq, e["dec"]], rmax):
                    p = pos[k]
                    ux, uy = np.cos(pa[p]), np.sin(pa[p])
                    gx, gy = e["ra"] * cdq * 3600.0, e["dec"] * 3600.0
                    t = np.clip((gx - cx[k]) * ux + (gy - cy[k]) * uy, -half[p], half[p])
                    if np.hypot(gx - (cx[k] + t * ux), gy - (cy[k] + t * uy)) < perp:
                        got = int(T.oid.iloc[p]); break
                if got >= 0:
                    break
            hits.append(got)
        if len(set(hits)) == 1 and hits[0] >= 0:
            out.add(hits[0])
    return out


def _table(T, arms, col, edges, label):
    print(f"\n{label:>12} {'n':>6}" + "".join(f"{a:>12}" for a in arms))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((T[col] >= lo) & (T[col] < hi)).to_numpy()
        if m.sum() < 20:
            continue
        print(f"{lo:>5.0f}-{hi:<6.0f}{int(m.sum()):>6}"
              + "".join(f"{100 * arms[a][m].mean():>11.2f}%" for a in arms))
    print(f"{'ALL':>12}{len(T):>6}" + "".join(f"{100 * arms[a].mean():>11.2f}%" for a in arms))


def main():
    T = pd.read_csv(f"{V}/truth_v2.csv").reset_index(drop=True)
    got = {}
    for arm in ARMS:
        al, n_link, n_surv = deliver(TAG[arm])
        oids = delivered_oids(al, T)
        got[arm] = T.oid.isin(oids).to_numpy()
        print(f"[{arm:6s}] linked {n_link:>7,} -> op survivors {n_surv:>6,} -> delivered {len(al):>5,}"
              f"   recovered {len(oids):>4} of {len(T):,}")
    print(f"\nDELIVERED-AT-{BUDGET} completeness (full 1k op, top {BUDGET} by priorityScore, "
          f"segment-matched)")
    _table(T, got, "L_px", L_EDGES, "trail px")
    _table(T, got, "snr_t", S_EDGES, "SNR")
    FF = ((T.rate > 2.0) & (T.rate <= 8.0) & (T.snr_t < 6.0)).to_numpy()
    print(f"\nFLAGSHIP cell (rate 2-8 deg/day, SNR<6, n={int(FF.sum())}):"
          + "".join(f"  {a} {100*got[a][FF].mean():.2f}%" for a in ARMS))
    a, s = got["ADCNN"], got["stack"]
    print(f"merged vs ADCNN: gained {int((got['merged'] & ~a).sum())}, "
          f"lost {int((a & ~got['merged']).sum())}   |   stack-only movers: {int((s & ~a).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
