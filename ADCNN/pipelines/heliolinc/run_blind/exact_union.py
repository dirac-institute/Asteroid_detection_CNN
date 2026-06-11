#!/usr/bin/env python3
"""#249: EXACT deduplicated stack/ADCNN union baseline tables (paper version of report §3.1).

Inputs (per blind field k): stack_full_s{5,4}_{k}_peaks.csv (full 5σ/4σ peak catalogs on injected
panels), adcnn_dets_masked_{k}.csv, inject_{k}.csv + truth_{k}.csv (snr_target).

Definitions (metric taxonomy of BLIND_TEST_REPORT.md):
  completeness (T1)  per injected sighting: hit iff a config detection lies within TOL px on its panel;
  TP_det             unique config detections within TOL px of any injected sighting (det-side);
  FP                 total unique detections - TP_det;  purity (T2) = TP_det/total  [injection-set];
  union dedup        stack peaks + ADCNN dets NOT within TOL px of a stack peak on that panel
                     (each physical detection counted once);
  incremental vs 5σ  ΔTP_sightings and ΔFP relative to the stack-5σ row.

NO thresholds are changed anywhere; this is measurement only.
"""
import json, os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29]
BINS = [(2, 5), (5, 10), (10, 31)]
TOL = 10.0

CFGS = ["s5", "s4", "a50", "a80", "s5+a50", "s5+a80", "s4+a80"]


def main():
    # accumulators: per config -> [hit per sighting bin counts], tp_det, total_dets
    sight_hits = {c: {b: [0, 0] for b in BINS} for c in CFGS}
    sight_tot = {c: [0, 0] for c in CFGS}              # [hits, n] overall
    tp_det = {c: 0 for c in CFGS}
    n_det = {c: 0 for c in CFGS}
    n_panels = 0
    for k in KS:
        inj = pd.read_csv(f"{HERE}/inject_{k}.csv")
        t = pd.read_csv(f"{HERE}/truth_{k}.csv").set_index("objID")
        inj["snr_t"] = inj.objID.map(t.snr_target)
        p5 = pd.read_csv(f"{HERE}/stack_full_s5_{k}_peaks.csv")
        p4 = pd.read_csv(f"{HERE}/stack_full_s4_{k}_peaks.csv")
        ad = pd.read_csv(f"{HERE}/adcnn_dets_masked_{k}.csv", usecols=["visit", "detector", "x", "y", "score"])
        g5 = dict(tuple(p5.groupby(["visit", "detector"])))
        g4 = dict(tuple(p4.groupby(["visit", "detector"])))
        ga = dict(tuple(ad.groupby(["visit", "detector"])))
        for (v, det), gi in inj.groupby(["visit", "detector"]):
            n_panels += 1
            s5xy = g5.get((v, det), pd.DataFrame(columns=["x", "y"]))[["x", "y"]].to_numpy()
            s4xy = g4.get((v, det), pd.DataFrame(columns=["x", "y"]))[["x", "y"]].to_numpy()
            adp = ga.get((v, det), pd.DataFrame(columns=["x", "y", "score"]))
            a50 = adp[["x", "y"]].to_numpy()
            a80 = adp.loc[adp.score >= 0.80, ["x", "y"]].to_numpy()

            def union(stack_xy, ad_xy):
                """stack peaks + adcnn dets >TOL px from every stack peak (dedup; each det once)."""
                if not len(stack_xy):
                    return ad_xy
                if not len(ad_xy):
                    return stack_xy
                d, _ = cKDTree(stack_xy).query(ad_xy, distance_upper_bound=TOL)
                return np.vstack([stack_xy, ad_xy[~np.isfinite(d)]])

            sets = {"s5": s5xy, "s4": s4xy, "a50": a50, "a80": a80,
                    "s5+a50": union(s5xy, a50), "s5+a80": union(s5xy, a80), "s4+a80": union(s4xy, a80)}
            ixy = gi[["x", "y"]].to_numpy()
            snr = gi.snr_t.to_numpy()
            for c, xy in sets.items():
                n_det[c] += len(xy)
                if len(xy):
                    tree = cKDTree(xy)
                    dd, _ = tree.query(ixy, distance_upper_bound=TOL)
                    hit = np.isfinite(dd)
                    # det-side TP: unique detections within TOL of any injected sighting
                    dets_near = tree.query_ball_point(ixy, r=TOL)
                    tp_det[c] += len({i for sub in dets_near for i in sub})
                else:
                    hit = np.zeros(len(gi), bool)
                sight_tot[c][0] += int(hit.sum()); sight_tot[c][1] += len(gi)
                for b in BINS:
                    m = (snr >= b[0]) & (snr < b[1])
                    sight_hits[c][b][0] += int(hit[m].sum()); sight_hits[c][b][1] += int(m.sum())
    rows = []
    base_tp = sight_tot["s5"][0]; base_fp = n_det["s5"] - tp_det["s5"]
    for c in CFGS:
        fp = n_det[c] - tp_det[c]
        rows.append(dict(
            config=c,
            **{f"C_snr{b[0]}_{b[1]}": round(100 * sight_hits[c][b][0] / sight_hits[c][b][1], 1) for b in BINS},
            C_all=round(100 * sight_tot[c][0] / sight_tot[c][1], 1),
            tp_sightings=sight_tot[c][0], tp_dets=tp_det[c], fp_dets=fp,
            purity_T2=round(100 * tp_det[c] / n_det[c], 2),
            dets_per_panel=round(n_det[c] / n_panels, 1),
            d_tp_vs_s5=sight_tot[c][0] - base_tp, d_fp_vs_s5=fp - base_fp))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(f"{HERE}/exact_union_table.csv", index=False)
    json.dump(rows, open(f"{HERE}/exact_union_table.json", "w"), indent=1)
    print("EXACT_UNION_DONE")


if __name__ == "__main__":
    main()
