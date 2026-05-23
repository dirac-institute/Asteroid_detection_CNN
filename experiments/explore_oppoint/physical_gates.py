"""explore_oppoint deliverable 3: cheap physical FP gates.

A genuine fast-mover residual is a coherent elongated line. Tune cuts on the
72-feature files so they keep ~ALL synthetic true-trail positives
(syn5_ft.pkl label_v2==1, posR>=0.99) while removing real-residual FP
(empft_0.csv).

IMPORTANT CAVEAT: the real on_truth asteroid recoveries in cand_*.csv have
NO feature rows anywhere (cand_*.csv carries only score_rf/on_truth/ObjID).
So a gate's effect on the *real* science frontier cannot be measured
directly. We use the synthetic-positive recall (posR) as the standard proxy
the project already uses (see fp_fix.txt): a gate with posR>=0.99 on
syn5_ft is assumed not to drop real trails. We then project the FP/CCD
reduction onto the flat-threshold science frontier by scaling the empty
candidate count by the empft FP-survival fraction at each score threshold.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/diffim_runs/test_real/results")
OUT = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/explore_oppoint")

pos = pd.read_pickle(RES / "syn5_ft.pkl")
pos = pos[pos.label_v2 == 1].copy()
fp = pd.read_csv(RES / "parts/empft_0.csv")
N_POS = len(pos)
N_FP = len(fp)
N_EMP_IMG = fp.image_id.nunique()


def clean(s):
    return s.replace([np.inf, -np.inf], np.nan)


def posR(mask_pos):
    return mask_pos.sum() / N_POS


def fpK(mask_fp):
    return mask_fp.sum() / N_FP


# ---- candidate single-feature lower-bound gates: largest cut with posR>=0.99
SINGLE = ["or_agg_mean_loose", "or_agg_mean_tight", "or_agg_max", "or_r",
          "mf_length", "area", "mf_snr", "mf_n_line", "or_n_pix",
          "lpca_L_w48_t10", "elongation"]

lines = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


w("=" * 78)
w("PHYSICAL FP GATES  (keep syn5 positives posR>=0.99, cut empft FP)")
w(f"N_pos(label_v2==1)={N_POS}  N_FP(empft)={N_FP}  empty CCDs={N_EMP_IMG}")
w(f"baseline FP/CCD (all empft) = {N_FP / N_EMP_IMG:.1f}")
w("=" * 78)
w("\n-- single-feature LOWER-bound gates (x >= cut), tuned to posR>=0.99 --")
w(f"{'feature':22s} {'cut':>10s} {'posR':>6s} {'fpKeep':>7s} {'FP/CCD':>8s}")
single_results = {}
for f in SINGLE:
    if f not in pos or f not in fp:
        continue
    p = clean(pos[f])
    e = clean(fp[f])
    # cut = the 1st percentile of positives (keeps 99%)
    cut = np.nanpercentile(p, 1.0)
    mp = (p >= cut) | p.isna()
    me = (e >= cut) & e.notna()  # NaN FP -> dropped (conservative: counts as removed)
    pr, fk = posR(mp), fpK(me)
    single_results[f] = (cut, pr, fk)
    w(f"{f:22s} {cut:10.4f} {pr:6.3f} {fk:7.3f} {fk * N_FP / N_EMP_IMG:8.1f}")

# ---- combined coherent-streak gate ----
# A real trail: coherent oriented line (high or_agg), elongated (mf_length),
# real flux (mf_snr not deeply negative), enough pixels (area/or_n_pix).
# Tune each sub-cut to ~p1 of positives so the AND keeps most positives.
def gate(df, cuts):
    m = pd.Series(True, index=df.index)
    for f, c in cuts.items():
        col = clean(df[f])
        # NaN in positives -> keep (don't penalize); NaN in FP -> drop
        if df is pos:
            m &= (col >= c) | col.isna()
        else:
            m &= (col >= c) & col.notna()
    return m


# build cuts at positive p1 (per-feature 99% retention); AND will lose a bit
cand_feats = ["or_agg_mean_loose", "mf_length", "area", "mf_snr"]
cuts = {f: float(np.nanpercentile(clean(pos[f]), 1.0)) for f in cand_feats}
# relax a touch to recover joint posR
for relax in [1.0, 0.5, 0.25, 0.1]:
    cuts_r = {f: float(np.nanpercentile(clean(pos[f]), relax)) for f in cand_feats}
    mp = gate(pos, cuts_r)
    me = gate(fp, cuts_r)
    w(f"\n-- COMBINED AND gate @ pos p{relax} : {cuts_r}")
    w(f"   posR={posR(mp):.4f}  fpKeep={fpK(me):.4f}  FP/CCD={fpK(me) * N_FP / N_EMP_IMG:.1f}"
      f"  (baseline {N_FP / N_EMP_IMG:.1f})")

# pick the best single + a 2-feature combo that holds posR>=0.99
w("\n-- best practical gates (posR>=0.99) --")
best = sorted([(v[2], k, v) for k, v in single_results.items() if v[1] >= 0.99])
for fk, k, v in best[:5]:
    w(f"  {k:22s} cut>={v[0]:.4f}  posR={v[1]:.4f}  FP/CCD {fk * N_FP / N_EMP_IMG:.1f} "
      f"(={(1 - fk) * 100:.0f}% FP cut)")

# 2-feature combos among the top discriminators
TOP = ["or_agg_mean_loose", "or_agg_mean_tight", "mf_length", "mf_snr", "area", "or_r"]
w("\n-- 2-feature AND combos tuned to joint posR>=0.99 --")
combo_rows = []
for i in range(len(TOP)):
    for j in range(i + 1, len(TOP)):
        f1, f2 = TOP[i], TOP[j]
        # search relax level so joint posR>=0.99
        chosen = None
        for relax in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
            c = {f1: float(np.nanpercentile(clean(pos[f1]), relax)),
                 f2: float(np.nanpercentile(clean(pos[f2]), relax))}
            mp = gate(pos, c)
            if posR(mp) >= 0.99:
                me = gate(fp, c)
                chosen = (relax, c, posR(mp), fpK(me))
                break
        if chosen:
            relax, c, pr, fk = chosen
            combo_rows.append((f1, f2, pr, fk, fk * N_FP / N_EMP_IMG))
            w(f"  {f1:20s}+{f2:20s} posR={pr:.4f} FP/CCD={fk * N_FP / N_EMP_IMG:6.1f} "
              f"({(1 - fk) * 100:.0f}% cut)  cuts={ {k: round(v, 3) for k, v in c.items()} }")

# ---- project best gate onto the flat science frontier ----
# FP-survival fraction is score-dependent; compute fpKeep(t) on empft for the
# chosen gate, then scale the cand empty count at each thr.
w("\n" + "=" * 78)
w("PROJECTION onto flat science frontier (best single gate)")
w("=" * 78)
if best:
    gf, (gcut, gpr, _) = best[0][1], (best[0][2][0], best[0][2][1], 0)
    w(f"chosen gate: {gf} >= {gcut:.4f}  (posR={gpr:.4f})")
    cand = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(str(RES / "parts/cand_*.csv")))],
                     ignore_index=True)
    emp = cand[cand.role == "empty"]
    ast = cand[cand.role == "asteroid"]
    ps = pd.read_csv(RES / "per_sighting_snr.csv")
    never = set(ps.groupby("ObjID").stack_detected.any().pipe(lambda s: s[~s].index))
    n_emp_img = emp.image_id.nunique()
    gcol = clean(fp[gf])
    w(f"\n{'thr':>5s} {'FP/CCD_flat':>11s} {'FP/CCD_gated':>12s} {'gateKeep@thr':>12s} "
      f"{'new_obj':>7s} {'new_sight':>9s}")
    for t in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        # gate FP-survival among empft rows with score_rf>=t (gate is posR-safe
        # so positives/asteroid recoveries assumed retained -> obj/sight unchanged)
        sub = fp[fp.score_rf >= t]
        if len(sub):
            keep_frac = ((clean(sub[gf]) >= gcut) & clean(sub[gf]).notna()).mean()
        else:
            keep_frac = 0.0
        flat_fp = (emp.score_rf >= t).sum() / n_emp_img
        gated_fp = flat_fp * keep_frac
        hit = ast[(ast.score_rf >= t) & (ast.on_truth == 1)]
        no = hit[hit.ObjID.isin(never)].ObjID.nunique()
        ns = hit.image_id.nunique()
        w(f"{t:5.2f} {flat_fp:11.1f} {gated_fp:12.1f} {keep_frac:12.3f} {no:7d} {ns:9d}")

(OUT / "_gates_report.txt").write_text("\n".join(lines) + "\n")
print("\nwrote", OUT / "_gates_report.txt")
