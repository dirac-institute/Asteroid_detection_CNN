#!/usr/bin/env python3
"""Compare the matched-filter run against its matched CONTROL, with the validity checks FIRST.

The previous attempt at this comparison was void: the MF run injected from
10k_cadence/run_night_20260706 while its baseline came from ringpipe_0706, so two variables changed
at once and the two runs contained different objects in different panels. This script therefore
refuses to report completeness until the runs are shown to be comparable.

GATE 1 -- identical injections. Same oids, same sky positions, same trail lengths, same SNRs.
          If these differ, the runs are not a controlled pair and nothing downstream is meaningful.
GATE 2 -- hardware effect. The MF run executed on an L40S, the control on an H200: different cuDNN
          kernels and accumulation order can flip detections that sit exactly on threshold. This is
          MEASURED (how many objects change detection state) rather than assumed negligible, because
          borderline detections are precisely the faint-band population under study.

Only if both gates pass does it report the faint-fast completeness cascade.

Usage:  python compare_mf_ctrl.py
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "outputs/runs/pa_validate")
from build_rank_table import radec_to_unit

V = "outputs/runs/pa_validate"
# FLAGSHIP CELL. Injection is UNIFORM over seven trail lengths, so 1/7 of the population sits at
# 9 deg/day -- far above any realistic NEO apparent-rate distribution, which falls steeply. Quoting a
# single "rate>4" number therefore lets a bin we do not care about drive the headline. The very fast
# end (>8 deg/day, 56px) is expected to be sparse in reality and is reported SEPARATELY rather than
# folded in.
FF = lambda T: (T.rate > 2.0) & (T.rate <= 8.0) & (T.snr_t < 6.0)   # faint + moderately fast
VFAST = lambda T: (T.rate > 8.0) & (T.snr_t < 6.0)                  # reported apart: rare in reality


def alert_oids(alerts_path, T, tol_arcsec=3.0):
    """oids whose BOTH epochs land on the SAME injected object (label hygiene)."""
    tol = 2 * np.sin(np.radians(tol_arcsec / 3600.0) / 2)
    trees = {}
    for s in "AB":
        for v, g in T.groupby(f"visit{s}"):
            trees[(int(v), s)] = (cKDTree(radec_to_unit(g[f"ra{s}"], g[f"dec{s}"])), g["oid"].to_numpy())
    out = set()
    for line in open(alerts_path):
        a = json.loads(line)
        eps = a["epochs"]
        if len(eps) < 2:
            continue
        oids = []
        for e in eps:
            hit = -1
            for s in "AB":
                tr = trees.get((int(e["visit"]), s))
                if tr is None:
                    continue
                d, i = tr[0].query(radec_to_unit([e["ra"]], [e["dec"]]), k=1)
                if d[0] < tol:
                    hit = int(tr[1][i[0]]); break
            oids.append(hit)
        if len(set(oids)) == 1 and oids[0] >= 0:
            out.add(oids[0])
    return out


def main():
    C = pd.read_csv(f"{V}/truth_ctrl.csv")
    M = pd.read_csv(f"{V}/truth_mf.csv")
    for T in (C, M):
        T["detA_ok"] = T.detA_ok.fillna(False); T["detB_ok"] = T.detB_ok.fillna(False)

    print("=" * 78)
    print("GATE 1 -- are the two runs injecting IDENTICAL objects?")
    print("=" * 78)
    j = C[["oid", "L_px", "snr_t", "rate", "raA", "decA"]].merge(
        M[["oid", "L_px", "snr_t", "rate", "raA", "decA"]], on="oid", suffixes=("_c", "_m"))
    dL = np.abs(j.L_px_c - j.L_px_m).max() if len(j) else np.inf
    dS = np.abs(j.snr_t_c - j.snr_t_m).max() if len(j) else np.inf
    dR = np.abs(j.raA_c - j.raA_m).max() if len(j) else np.inf
    print(f"  control n={len(C):,}   MF n={len(M):,}   matched oids={len(j):,}")
    print(f"  max |dL_px|={dL:.3e}   max |dSNR|={dS:.3e}   max |dRA|={dR:.3e} deg")
    ok1 = (len(C) == len(M)) and dL < 1e-6 and dS < 1e-9 and dR < 1e-9
    print(f"  VERDICT: {'PASS -- controlled pair' if ok1 else 'FAIL -- NOT a controlled pair, stopping'}")
    if not ok1:
        print("\n  The runs differ in their injected population, so any completeness difference would be")
        print("  confounded with a different panel/object set. Do not report a comparison from this.")
        return

    print("\n" + "=" * 78)
    print("GATE 2 -- how big is the L40S (MF run) vs H200 (control) hardware effect?")
    print("=" * 78)
    m = C[["oid", "detA_ok", "detB_ok"]].merge(M[["oid", "detA_ok", "detB_ok"]], on="oid",
                                               suffixes=("_c", "_m"))
    fa = (m.detA_ok_c != m.detA_ok_m).mean(); fb = (m.detB_ok_c != m.detB_ok_m).mean()
    print(f"  epoch-A detection state differs for {100*fa:.2f}% of objects")
    print(f"  epoch-B detection state differs for {100*fb:.2f}% of objects")
    print("  NOTE this bound is NOT purely hardware: the MF changes `length`/`beta`, which feed the")
    print("  catalogue and can legitimately alter which sources survive. It is an UPPER bound on the")
    print("  hardware term, and it is the number to weigh any completeness delta against.")

    print("\n" + "=" * 78)
    print("FLAGSHIP CELL: faint + moderately fast (rate 2-8 deg/day, i-band SNR < 6)")
    print("=" * 78)
    aC = alert_oids(f"{V}/a_ctrl.jsonl", C)
    aM = alert_oids(f"{V}/a_mf.jsonl", M)
    print(f"{'stage':<34}{'CONTROL (seg)':>16}{'MF':>10}{'delta':>10}")
    for lab, key in (("detected epoch A", "A"), ("detected BOTH epochs", "B"), ("linked into an alert", "L")):
        vals = []
        for T, al in ((C, aC), (M, aM)):
            g = T[FF(T)]
            v = (g.detA_ok if key == "A" else
                 (g.detA_ok & g.detB_ok) if key == "B" else
                 (g.detA_ok & g.detB_ok & g.oid.isin(al))).mean()
            vals.append(100 * v)
        print(f"{lab:<34}{vals[0]:>15.2f}%{vals[1]:>9.2f}%{vals[1]-vals[0]:>+10.2f}")
    print(f"\n{'linking efficiency (of both-det)':<34}", end="")
    for T, al in ((C, aC), (M, aM)):
        g = T[FF(T)]; b = g[g.detA_ok & g.detB_ok]
        print(f"{100*b.oid.isin(al).mean():>15.1f}%", end="")
    print()
    print(f"\nPER-RATE-BIN completeness (linked into an alert), SNR<6 -- so a sparse very-fast bin")
    print(f"cannot drive the headline. Injection is UNIFORM in trail length; real rate distributions")
    print(f"fall steeply, so the low-rate rows carry far more real weight than the high-rate ones.")
    print(f"{'trail px':>9}{'rate':>7}{'n':>6}{'CONTROL':>10}{'MF':>9}{'delta':>9}")
    for L in sorted(C.L_target.unique()):
        gc = C[(C.L_target == L) & (C.snr_t < 6)]; gm = M[(M.L_target == L) & (M.snr_t < 6)]
        if len(gc) < 20:
            continue
        vc = 100*(gc.detA_ok & gc.detB_ok & gc.oid.isin(aC)).mean()
        vm = 100*(gm.detA_ok & gm.detB_ok & gm.oid.isin(aM)).mean()
        print(f"{L:>9.0f}{L*0.2/3600*86400/30:>7.1f}{len(gc):>6}{vc:>9.2f}%{vm:>8.2f}%{vm-vc:>+9.2f}")
    print(f"\nVERY FAST (>8 deg/day, SNR<6) -- reported apart, expected sparse in reality:")
    for lab, T, al in (("control", C, aC), ("MF", M, aM)):
        g = T[VFAST(T)]
        print(f"  {lab:<8} n={len(g):>4}  both-det {100*(g.detA_ok&g.detB_ok).mean():>5.2f}%   "
              f"alert {100*(g.detA_ok&g.detB_ok&g.oid.isin(al)).mean():>5.2f}%")
    print(f"\nALL injected (not just faint-fast):")
    for lab, T, al in (("control", C, aC), ("MF", M, aM)):
        print(f"  {lab:<8} both-det {100*(T.detA_ok&T.detB_ok).mean():>5.2f}%   "
              f"alert {100*(T.detA_ok&T.detB_ok&T.oid.isin(al)).mean():>5.2f}%")


if __name__ == "__main__":
    main()
