#!/usr/bin/env python3
"""Audit EVERY cut in the shipped op-point against injected truth: what does each one actually cost?

A cut is only worth its purity if it sits OUTSIDE the true-mover distribution. `chi2_2v_max: 8.0`
turned out to sit at the MEDIAN of it (true-alert chi2 median 7.4), discarding ~46% of true movers
before ranking -- the same failure mode as `rate_hi_2v: 8.0`. This checks the rest the same way.

For each cut, in isolation (all other cuts off), on the ungated alert stream:
    kill_true  -- fraction of TRUE alerts it removes      (the cost)
    kill_fp    -- fraction of FP alerts it removes        (the benefit)
    ratio      -- kill_fp / kill_true                     (>1 = the cut earns its place)

Usage:  python audit_op.py <alerts.jsonl> <truth.csv> <op.json> [refcat.parquet]
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "outputs/runs/pa_validate")
from build_rank_table import radec_to_unit


def load(alerts_path, truth_path, tol_arcsec=3.0):
    A = [json.loads(l) for l in open(alerts_path)]
    T = pd.read_csv(truth_path)
    tol = 2 * np.sin(np.radians(tol_arcsec / 3600.0) / 2)
    trees = {}
    for side in "AB":
        for vis, g in T.groupby(f"visit{side}"):
            trees[(int(vis), side)] = (cKDTree(radec_to_unit(g[f"ra{side}"], g[f"dec{side}"])),
                                       g["oid"].to_numpy())
    rows = []
    for a in A:
        eps = a["epochs"]
        if len(eps) < 2:
            continue
        oids = []
        for e in eps:
            hit = -1
            for side in "AB":
                tr = trees.get((int(e["visit"]), side))
                if tr is None:
                    continue
                d, i = tr[0].query(radec_to_unit([e["ra"]], [e["dec"]]), k=1)
                if d[0] < tol:
                    hit = int(tr[1][i[0]]); break
            oids.append(hit)
        v = a.get("vetting") or {}; o = a.get("orbit") or {}; m = a.get("motion") or {}
        st = a.get("stationarity") or {}
        tl = [x for x in (v.get("trail_len_px") or []) if x is not None]
        def f(x, d=np.nan):
            return d if x is None else float(x)
        ncp = sum(1 for k in ("e1", "e2") if (st.get(k) or {}).get("counterpart"))
        rows.append(dict(
            y=(len(set(oids)) == 1 and oids[0] >= 0), oid=oids[0],
            chi2=f(o.get("chi2")), mfsnr=f(v.get("mfsnr_min")), smin=f(v.get("score_min")),
            rate=f(m.get("rate_degday")), tlen=float(np.mean(tl)) if tl else np.nan,
            tlen_min=float(np.min(tl)) if tl else np.nan,
            rms=f(a.get("rmsArcsec"), 0.0), arc=f(a.get("arcMin")),
            n_cp=ncp, static=a.get("staticVeto") is not None, train=a.get("trainVeto") is not None,
            ra=float(eps[0]["ra"]), dec=float(eps[0]["dec"]),
        ))
    return pd.DataFrame(rows)


def main(alerts, truth, op_path, refcat=None):
    D = load(alerts, truth)
    op = json.load(open(op_path))
    T, F = D[D.y], D[~D.y]
    print(f"stream {len(D):,} alerts | TRUE {len(T):,} ({T.oid.nunique():,} distinct injected objects, "
          f"so {len(T)/max(T.oid.nunique(),1):.2f} alerts per recovered object) | FP {len(F):,}\n")

    prox = np.zeros(len(D), bool)
    if refcat:
        try:
            R = pd.read_parquet(refcat)
            mcol = next((c for c in ("mag", "phot_g_mean_mag", "g") if c in R.columns), None)
            R = R[R[mcol] < op.get("bright_star_mag_max", 21.0)] if mcol else R
            rt = cKDTree(radec_to_unit(R.ra, R.dec))
            rad = 2 * np.sin(np.radians(op.get("bright_star_radius_arcsec", 2.5) / 3600.0) / 2)
            prox = rt.query(radec_to_unit(D.ra, D.dec), k=1)[0] < rad
        except Exception as e:
            print(f"  (refcat proximity skipped: {type(e).__name__}: {e})")

    CUTS = [
        (f"chi2 > {op['chi2_2v_max']}",            D.chi2 > op["chi2_2v_max"]),
        (f"mfsnr < {op['mfsnr_min_2v']}",          D.mfsnr < op["mfsnr_min_2v"]),
        (f"score_min < {op['score_min']}",         D.smin < op["score_min"]),
        (f"rate < {op['rate_lo_2v']} deg/day",     D.rate < op["rate_lo_2v"]),
        (f"rate > {op['rate_hi_2v']} deg/day",     D.rate > op["rate_hi_2v"]),
        (f"len_db < {op['len_db_min']} px",        D.tlen_min < op["len_db_min"]),
        (f"rms > {op['max_rms']}\"",               D.rms > op["max_rms"]),
        ("staticVeto",                             D.static),
        ("trainVeto",                              D.train),
        ("stationary, >=1 counterpart",            D.n_cp >= 1),
        ("stationary, BOTH counterparts (dropped)", D.n_cp >= 2),
        (f"bright-star prox <{op.get('bright_star_radius_arcsec')}\" mag<{op.get('bright_star_mag_max')}", prox),
    ]
    print(f"{'cut (in isolation)':<44}{'kills TRUE':>12}{'kills FP':>10}{'ratio':>8}   verdict")
    for name, mask in CUTS:
        mask = np.asarray(mask, bool)
        kt = mask[D.y.to_numpy()].mean() if len(T) else 0.0
        kf = mask[~D.y.to_numpy()].mean() if len(F) else 0.0
        ratio = (kf / kt) if kt > 1e-9 else np.inf
        verdict = ("FREE" if kt < 0.01 else "earns it" if ratio >= 3 else
                   "MARGINAL" if ratio >= 1 else "COSTS MORE THAN IT BUYS")
        print(f"{name:<44}{100*kt:>11.1f}%{100*kf:>9.1f}%{ratio:>8.2f}   {verdict}")

    print(f"\nTRUE-alert distribution vs each threshold (is the cut inside the signal?):")
    for col, thr, side in (("chi2", op["chi2_2v_max"], "max"), ("mfsnr", op["mfsnr_min_2v"], "min"),
                           ("rate", op["rate_hi_2v"], "max"), ("rate", op["rate_lo_2v"], "min"),
                           ("tlen_min", op["len_db_min"], "min")):
        q = T[col].dropna()
        if not len(q):
            continue
        pct = 100 * ((q > thr).mean() if side == "max" else (q < thr).mean())
        print(f"  {col:<9} cut {side}={thr:<6} | TRUE p10={q.quantile(.1):>7.2f} med={q.median():>7.2f} "
              f"p90={q.quantile(.9):>7.2f} | {pct:>5.1f}% of TRUE on the cut side")


if __name__ == "__main__":
    main(*sys.argv[1:])
