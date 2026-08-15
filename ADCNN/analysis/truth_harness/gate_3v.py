#!/usr/bin/env python3
"""Measure the 3+visit gate against injected TRUTH triplets — true vs chance separations.

Consumes a 3-epoch injection run (inject_night with INJ_EPOCHS=3 -> truth_{TAG}.csv +
inj_dets_{TAG}.csv) AFTER the catalogue has been linked. For every 3+visit alert in the linked
stream it decides TRUE (every member within MATCH_ARCSEC of the same injected mover's predicted
per-epoch position) or CHANCE, then reports the distribution of the published geometry statistics
(linRmsArcsec / trailMotionDpaMaxDeg / speedRatio) for each class, plus per-truth-triplet recovery.

That is the measurement the gate has never had: physical_check's thresholds (1" linear RMS, 20 deg
PA, 50% speed) were set by construction, and the 23 real campaign tracks only show that ACCEPTED
tracks sit far inside them (gate-censored, so uninformative about where the cut should be). The
chance-triplet side of the separation is what licenses any tightening -- and if the two
distributions overlap, that is a finding too: the gate cannot be tightened without recall cost,
judged as always at the delivered budget.

Usage:
    python -m ADCNN.analysis.truth_harness.gate_3v \
        --alerts outputs/runs/pa_validate/tri_a_n20260706.jsonl \
        --truth  outputs/runs/pa_validate/truth_tri_n20260706.csv
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np
import pandas as pd

MATCH_ARCSEC = 1.5


def truth_positions(T):
    """oid -> list of (mjd-free) per-epoch sky positions the mover was injected at.

    Epoch A is (raA, decA); follow-ups are (raB, decB), (raC, decC)... as written by inject_night.
    Only epochs whose det{X}_ok column says the detector recovered the mover are REQUIRED matches
    for recovery accounting, but for alert-membership matching every injected position counts —
    an alert member can match an epoch the cascade marked undetected only if the detector actually
    emitted something there, which is exactly what we want to credit.
    """
    out = {}
    for _, r in T.iterrows():
        eps = [("A", r.get("raA"), r.get("decA"))]
        for tg in ("B", "C", "D"):
            if f"ra{tg}" in T.columns and np.isfinite(r.get(f"ra{tg}", np.nan)):
                eps.append((tg, r[f"ra{tg}"], r[f"dec{tg}"]))
        out[int(r.oid)] = [(t, float(ra), float(dec)) for t, ra, dec in eps
                           if np.isfinite(ra) and np.isfinite(dec)]
    return out


def classify(alerts, T, match_arcsec=MATCH_ARCSEC):
    """-> (true_alerts, chance_alerts, matched_oids). An alert is TRUE iff every member epoch lies
    within match_arcsec of the SAME injected mover's predicted position for some epoch."""
    tp = truth_positions(T)
    oids = np.array(list(tp.keys()))
    # flatten for a coarse prefilter
    flat = [(o, ra, dec) for o, eps in tp.items() for _, ra, dec in eps]
    fra = np.array([f[1] for f in flat]); fdec = np.array([f[2] for f in flat])
    foid = np.array([f[0] for f in flat])
    tol = match_arcsec / 3600.0
    true_a, chance_a, matched = [], [], set()
    for a in alerts:
        if int(a.get("nEpochs", 2)) < 3:
            continue
        cands = None
        ok = True
        for e in a["epochs"]:
            d = np.hypot((fra - e["ra"]) * np.cos(np.radians(e["dec"])), fdec - e["dec"])
            near = set(foid[d < tol].tolist())
            cands = near if cands is None else (cands & near)
            if not cands:
                ok = False
                break
        if ok and cands:
            true_a.append(a); matched.update(cands)
        else:
            chance_a.append(a)
    return true_a, chance_a, matched


def geom_stats(alerts, label):
    rows = []
    for a in alerts:
        g = a.get("geometry") or {}
        rows.append(dict(alert=a.get("alertId"),
                         rms=g.get("linRmsArcsec"),
                         dpa=g.get("trailMotionDpaMaxDeg"),
                         spmax=g.get("speedRatioMax"),
                         spmin=g.get("speedRatioMin"),
                         arc=a.get("arcMin"),
                         rate=(a.get("motion") or {}).get("rate_degday")))
    D = pd.DataFrame(rows)
    print(f"\n=== {label}: {len(D)} alert(s)")
    if not len(D):
        return D
    for c, unit in (("rms", '"'), ("dpa", "deg"), ("spmax", "x"), ("arc", "min")):
        v = pd.to_numeric(D[c], errors="coerce").dropna()
        if len(v):
            print(f"  {c:>6}: median {v.median():7.3f}{unit}  p90 {v.quantile(.9):7.3f}  "
                  f"max {v.max():7.3f}  (n={len(v)})")
    return D


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True, help="linked stream alerts.jsonl of the injection run")
    ap.add_argument("--truth", required=True, help="truth_{TAG}.csv from inject_night INJ_EPOCHS=3")
    ap.add_argument("--match-arcsec", type=float, default=MATCH_ARCSEC)
    a = ap.parse_args(argv)

    alerts = [json.loads(l) for l in open(a.alerts)]
    T = pd.read_csv(a.truth)
    okc = [c for c in ("detA_ok", "detB_ok", "detC_ok") if c in T.columns]
    all_ok = np.logical_and.reduce([T[c].fillna(False).to_numpy(bool) for c in okc]) \
        if okc else np.zeros(len(T), bool)
    print(f"truth: {len(T):,} injected movers; detected in ALL {len(okc)} epochs: {int(all_ok.sum()):,} "
          f"({100 * all_ok.mean():.1f}%)")
    n3 = sum(1 for x in alerts if int(x.get("nEpochs", 2)) >= 3)
    print(f"alerts: {len(alerts):,} total, {n3} are 3+visit")

    true_a, chance_a, matched = classify(alerts, T, a.match_arcsec)
    # RECOVERY: of movers the detector found in every epoch, how many became a 3+visit alert?
    T_ok = T[all_ok]
    rec = T_ok.oid.isin(matched)
    print(f"\nRECOVERY: {int(rec.sum())} of {len(T_ok)} all-epoch-detected movers became a TRUE "
          f"3+visit alert ({100 * rec.mean():.1f}%)"
          if len(T_ok) else "\nRECOVERY: no all-epoch-detected movers")

    Dt = geom_stats(true_a, "TRUE 3+visit (all members match one injected mover)")
    Dc = geom_stats(chance_a, "CHANCE 3+visit (at least one member unmatched)")

    # The question the gate needs answered: does any published statistic separate the classes?
    if len(Dt) and len(Dc):
        print("\nSEPARATION (TRUE p90 vs CHANCE p10 — a gap means a cut exists):")
        for c in ("rms", "dpa", "spmax"):
            t = pd.to_numeric(Dt[c], errors="coerce").dropna()
            ch = pd.to_numeric(Dc[c], errors="coerce").dropna()
            if len(t) and len(ch):
                print(f"  {c:>6}: TRUE p90 {t.quantile(.9):7.3f}  CHANCE p10 {ch.quantile(.1):7.3f}  "
                      f"{'SEPARATED' if t.quantile(.9) < ch.quantile(.1) else 'OVERLAP'}")
    elif len(Dt) and not len(Dc):
        print("\nNo chance triplets in this stream — the gate's FP side is unconstrained here; "
              "tightening remains unjustified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
