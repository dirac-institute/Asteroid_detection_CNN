#!/usr/bin/env python3
"""Score the asymmetric-threshold experiment: does going deep on the SHALLOW epoch buy faint-fast
completeness, and what does it cost in volume?

Both sides must be reported. A completeness gain bought with a 10x alert explosion is not a gain --
the product has a fixed ~1k budget, so extra false positives displace real movers one-for-one.

The B-epoch catalogue was detected once at the deepest threshold; higher thresholds are strict
SUBSETS, so the whole threshold curve is recovered by filtering on score -- no extra GPU passes.

Usage:  python analyze_asym.py <alerts.jsonl> <truth.csv> [baseline_alerts.jsonl baseline_truth.csv]
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "outputs/runs/pa_validate")
from build_rank_table import radec_to_unit


def match(alerts_path, truth_path, tol_arcsec=3.0):
    A = [json.loads(l) for l in open(alerts_path)]
    T = pd.read_csv(truth_path)
    tol = 2 * np.sin(np.radians(tol_arcsec / 3600.0) / 2)
    trees = {}
    for s in "AB":
        for v, g in T.groupby(f"visit{s}"):
            trees[(int(v), s)] = (cKDTree(radec_to_unit(g[f"ra{s}"], g[f"dec{s}"])), g["oid"].to_numpy())
    rows = []
    for a in A:
        e = a["epochs"]
        if len(e) < 2:
            continue
        oids, smin_e = [], []
        for ep in e:
            h = -1
            for s in "AB":
                tr = trees.get((int(ep["visit"]), s))
                if tr is None:
                    continue
                d, i = tr[0].query(radec_to_unit([ep["ra"]], [ep["dec"]]), k=1)
                if d[0] < tol:
                    h = int(tr[1][i[0]]); break
            oids.append(h)
            smin_e.append(float(ep.get("score") or 0.0))
        v = a.get("vetting") or {}; o = a.get("orbit") or {}; m = a.get("motion") or {}
        c, sm = o.get("chi2"), v.get("score_min")
        if c is None or sm is None:
            continue
        rows.append(dict(oid=oids[0] if len(set(oids)) == 1 else -1, chi2=float(c),
                         smin=float(sm), smin_epoch=min(smin_e),
                         mfsnr=float(v.get("mfsnr_min") or 0), rate=float(m.get("rate_degday") or 0),
                         pscore=2.0 + 0.95 * float(sm)))
    return pd.DataFrame(rows), T


def report(D, T, label):
    T = T.copy()
    T["detA_ok"] = T.detA_ok.fillna(False); T["detB_ok"] = T.detB_ok.fillna(False)
    FF = (T.rate > 4.0) & (T.snr_t < 6.0)
    alert = set(D[D.oid >= 0].oid)
    g = T[FF]
    print(f"\n=== {label} ===")
    print(f"  alerts {len(D):,} | injected {len(T):,} | faint-fast {int(FF.sum())}")
    print(f"  FAINT-FAST cascade: detA {100*g.detA_ok.mean():.1f}%  -> BOTH {100*(g.detA_ok&g.detB_ok).mean():.1f}%"
          f"  (shallow epoch keeps {100*g[g.detA_ok].detB_ok.mean():.1f}% of what deep found)"
          f"  -> alert {100*g.oid.isin(alert).mean():.2f}%")
    print(f"  ALL: detA {100*T.detA_ok.mean():.1f}%  BOTH {100*(T.detA_ok&T.detB_ok).mean():.1f}%"
          f"  alert {100*T.oid.isin(alert).mean():.2f}%")
    # threshold curve: the B catalogue is a superset, so higher thresholds are recovered by filtering
    print(f"\n  {'B-epoch score floor':>20}{'alerts':>10}{'FF completeness@1k':>21}{'ALL@1k':>10}{'FF alert ceiling':>19}")
    for thr in (0.10, 0.15, 0.20, 0.30, 0.50):
        d = D[D.smin_epoch >= thr]
        sel = d.sort_values("pscore", ascending=False).head(1000)
        o = set(sel[sel.oid >= 0].oid)
        ceil = set(d[d.oid >= 0].oid)
        print(f"{thr:>20.2f}{len(d):>10,}{100*g.oid.isin(o).mean():>20.2f}%{100*T.oid.isin(o).mean():>9.2f}%"
              f"{100*g.oid.isin(ceil).mean():>18.2f}%")


if __name__ == "__main__":
    D, T = match(sys.argv[1], sys.argv[2])
    report(D, T, f"ASYMMETRIC {sys.argv[1]}")
    if len(sys.argv) > 4:
        D0, T0 = match(sys.argv[3], sys.argv[4])
        report(D0, T0, f"BASELINE {sys.argv[3]}")
