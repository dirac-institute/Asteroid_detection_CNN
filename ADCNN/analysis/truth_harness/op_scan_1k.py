#!/usr/bin/env python3
"""2D (score_min, chi2_2v_max) scan for DELIVERED-AT-1k completeness. Tune on one night, VALIDATE on
another that never influenced the choice.

WHY THIS AND NOT THE EXISTING SWEEP. outputs/runs/pa_validate/truth_op_sweep.csv already scans
score x chi2 x mfsnr x veto, but it scores completeness at the ALERT STAGE -- its best cell is 3.47%
at N=24,079 alerts. The product ships ~1,000 alerts, so an alert-stage optimum answers a question
nobody asks: at a fixed budget, admitting more can DISPLACE real movers, and two verdicts have already
flipped on exactly that. This scans the same axes at the delivered budget.

WHY score COSTS A LINK AND chi2 DOES NOT. score_min decides which DETECTIONS enter linking, so it
changes the seed set and the greedy claim competition -- it needs its own link per value. chi2 filters
AFTER the orbit solve, so the chi2<=X alert set is a strict SUBSET of a chi2<=30 link: every chi2 point
is free, evaluated post-hoc from one linked stream. Hence 4 links, not 4x8.

The 1k op is applied for everything EXCEPT the two axes being scanned, so the scan measures those two
axes and not a different op.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e2e_hist import delivered_oids                      # trail-aware segment matching

V = "outputs/runs/pa_validate"
OP1K = "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json"
REFCAT = "outputs/runs/10k_cadence/run_night_20260706/bright_refcat.parquet"
BUDGET = 1000
SCORES = [0.50, 0.60, 0.70, 0.80]
CHI2S = [3, 5, 8, 12, 18, 25, 30]
FF = lambda T: (T.rate > 2.0) & (T.rate <= 8.0) & (T.snr_t < 6.0)


def _G(a, *ks, d=None):
    x = a
    for k in ks:
        if not isinstance(x, dict):
            return d
        x = x.get(k)
    return d if x is None else x


def deliver(alerts, T, chi2_max, op):
    """Apply the 1k op with chi2 overridden, take the top BUDGET by the shipped order, score it."""
    from ADCNN.qa.select_clean import _confident_fp
    keep = []
    for a in alerts:
        c = _G(a, "orbit", "chi2", d=None)
        if c is not None and float(c) > chi2_max:
            continue
        if _confident_fp(a) is not None:
            continue
        if _G(a, "vetting", "mfsnr_min", d=0) < op["mfsnr_min_2v"]:
            continue
        if np.mean(_G(a, "vetting", "trail_len_px", d=[0]) or [0]) < op["len_db_min"]:
            continue
        r = _G(a, "motion", "rate_degday", d=0)
        if not (op["rate_lo_2v"] <= r <= op["rate_hi_2v"]):
            continue
        keep.append(a)
    keep.sort(key=lambda a: (-(a.get("priorityScore") or 0.0),
                             float(((a.get("orbit") or {}).get("chi2")) or 1e9)))
    return delivered_oids(keep[:BUDGET], T), len(keep)


def scan(truth_csv, arms, label):
    T = pd.read_csv(truth_csv).reset_index(drop=True)
    op = json.load(open(OP1K))
    ff = FF(T).to_numpy()
    rows = []
    for s, path in arms:
        if not os.path.exists(path):
            print(f"  [{label}] score={s}: MISSING {path}"); continue
        alerts = [json.loads(l) for l in open(path)]
        for c in CHI2S:
            oids, n_surv = deliver(alerts, T, c, op)
            hit = T.oid.isin(oids).to_numpy()
            rows.append(dict(score=s, chi2=c, linked=len(alerts), surv=n_surv,
                             delivered=min(n_surv, BUDGET), movers=int(hit.sum()),
                             ALL=100 * hit.mean(), FLAG=100 * hit[ff].mean()))
            print(f"  [{label}] score={s:.2f} chi2<={c:<3} surv={n_surv:>6,} "
                  f"movers={int(hit.sum()):>4}  ALL={100*hit.mean():5.2f}%  "
                  f"FLAGSHIP={100*hit[ff].mean():5.2f}%", flush=True)
    return pd.DataFrame(rows)


L_EDGES = [0, 8, 12, 16, 24, 32, 44, 60]
S_EDGES = [0, 3, 4, 5, 6, 8, 10, 99]


def grid_table(T, hit, label):
    """Delivered-at-1k completeness as SNR x TRAIL LENGTH -- the shape of what the op actually ships."""
    print(f"\n{label}: delivered-at-{BUDGET} completeness (%), rows=SNR, cols=trail px")
    hdr = f"{'SNR':>9}" + "".join(f"{f'{lo}-{hi}':>9}" for lo, hi in zip(L_EDGES[:-1], L_EDGES[1:])) + f"{'ALL':>9}"
    print(hdr)
    for slo, shi in zip(S_EDGES[:-1], S_EDGES[1:]):
        sm = ((T.snr_t >= slo) & (T.snr_t < shi)).to_numpy()
        if sm.sum() < 20:
            continue
        row = f"{f'{slo}-{shi}':>9}"
        for llo, lhi in zip(L_EDGES[:-1], L_EDGES[1:]):
            m = sm & ((T.L_px >= llo) & (T.L_px < lhi)).to_numpy()
            row += f"{(100*hit[m].mean() if m.sum() >= 10 else float('nan')):>8.1f}%" if m.sum() >= 10 else f"{'-':>9}"
        row += f"{100*hit[sm].mean():>8.1f}%"
        print(row)
    row = f"{'ALL':>9}"
    for llo, lhi in zip(L_EDGES[:-1], L_EDGES[1:]):
        m = ((T.L_px >= llo) & (T.L_px < lhi)).to_numpy()
        row += f"{100*hit[m].mean():>8.1f}%" if m.sum() >= 10 else f"{'-':>9}"
    print(row + f"{100*hit.mean():>8.1f}%")


def main():
    arms = [(s, f"{V}/sweep_a_s{str(s).replace('.','')[:3]}.jsonl") for s in SCORES]
    T = pd.read_csv(f"{V}/truth_v3.csv").reset_index(drop=True)
    d = scan(f"{V}/truth_v3.csv", arms, "TUNE 0706")
    if not len(d):
        raise SystemExit("no arms found")
    d.to_csv(f"{V}/op_scan_1k_tune.csv", index=False)
    print("\n" + "=" * 72)
    print("DELIVERED-AT-1k completeness, ALL (%), rows=score_min, cols=chi2_2v_max")
    print(d.pivot(index="score", columns="chi2", values="ALL").round(2).to_string())
    print("\nFLAGSHIP cell (rate 2-8 deg/day, SNR<6) (%)")
    print(d.pivot(index="score", columns="chi2", values="FLAG").round(2).to_string())
    b = d.loc[d.ALL.idxmax()]
    bf = d.loc[d.FLAG.idxmax()]
    print(f"\nBEST by ALL      : score_min={b.score:.2f} chi2<={int(b.chi2)} -> {b.ALL:.2f}% "
          f"({int(b.movers)} movers, {int(b.surv):,} survivors)")
    print(f"BEST by FLAGSHIP : score_min={bf.score:.2f} chi2<={int(bf.chi2)} -> {bf.FLAG:.2f}%")
    print("\nThe chosen point must be CONFIRMED on 20260713, which took no part in this choice.")
    # the SNR x trail-length shape at the argmax cells
    op = json.load(open(OP1K))
    for tag, cell in (("BEST-ALL", b), ("BEST-FLAGSHIP", bf)):
        path = f"{V}/sweep_a_s{str(cell.score).replace('.','')[:3]}.jsonl"
        if not os.path.exists(path):
            continue
        oids, _ = deliver([json.loads(l) for l in open(path)], T, int(cell.chi2), op)
        grid_table(T, T.oid.isin(oids).to_numpy(),
                   f"{tag} (score_min={cell.score:.2f}, chi2<={int(cell.chi2)})")


if __name__ == "__main__":
    sys.exit(main())
