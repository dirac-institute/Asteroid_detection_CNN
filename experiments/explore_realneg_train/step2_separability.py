"""Step 2: feature-space separability of REAL empty-CCD false positives
vs SYNTHETIC true positives (the discrimination on-real-bg training targets).

Inputs (read-only, on disk):
  - experiments/diffim_runs/test_real/results/parts/emp_*.csv     (seg_model-raw real-empty cand, 72 feat + meta)
  - experiments/diffim_runs/test_real/results/parts/empft_*.csv   (seg_ft real-empty cand)
  - experiments/diffim_runs/test_real/results/syn5_ft.pkl         (synthetic cand; label_v2==1 = true trail)
  - experiments/diffim_runs/test_real/results/per_panel_fp.csv    (FP/panel)

Output: stdout summary + experiments/explore_realneg_train/step2_separability.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
PARTS = os.path.join(REPO, "experiments/diffim_runs/test_real/results/parts")
RES = os.path.join(REPO, "experiments/diffim_runs/test_real/results")
OUT = os.path.join(REPO, "experiments/explore_realneg_train/step2_separability.txt")

FEATS = ["mf_snr", "mf_length", "elongation", "loc_dipole", "aspect",
         "or_agg_max", "or_snr_L50", "loc_std_z", "loc_skew",
         "lpca_elong_w48_t10", "max_p", "mf_flux"]

lines = []
def p(s=""):
    print(s)
    lines.append(str(s))


def load_concat(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


emp = load_concat(os.path.join(PARTS, "emp_*.csv"))
empft = load_concat(os.path.join(PARTS, "empft_*.csv"))
syn = pd.read_pickle(os.path.join(RES, "syn5_ft.pkl"))
ppf = pd.read_csv(os.path.join(RES, "per_panel_fp.csv"))

syn_pos = syn[syn["label_v2"] == 1].copy()
syn_neg = syn[syn["label_v2"] == 0].copy()

p("=" * 78)
p("STEP 2  —  REAL empty-CCD FP  vs  SYNTHETIC true positives")
p("=" * 78)
p(f"real empty cand (emp_*, seg_model raw)   : {len(emp):>7d} rows  "
  f"panels={emp['image_id'].nunique() if emp is not None else 0}")
p(f"real empty cand (empft_*, seg_ft)  : {len(empft):>7d} rows  "
  f"panels={empft['image_id'].nunique() if empft is not None else 0}")
p(f"synthetic cand total              : {len(syn):>7d} rows")
p(f"  synthetic TRUE  (label_v2==1)   : {len(syn_pos):>7d}")
p(f"  synthetic FALSE (label_v2==0)   : {len(syn_neg):>7d}")
p("")
p("Per-panel FP (per_panel_fp.csv), role=empty:")
emp_pan = ppf[ppf["role"] == "empty"]
p(f"  empty panels: {len(emp_pan)}  nn_fp mean={emp_pan['nn_fp'].mean():.1f} "
  f"median={emp_pan['nn_fp'].median():.0f} max={emp_pan['nn_fp'].max()} "
  f"total={emp_pan['nn_fp'].sum()}")
p(f"  empty panels with 0 NN FP: {(emp_pan['nn_fp']==0).sum()}/{len(emp_pan)}")
p("")

# ---------------------------------------------------------------------------
# Distributional comparison: real-empty FP  vs  synthetic true trails
# ---------------------------------------------------------------------------
def qstats(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return None
    return dict(n=len(s), p05=s.quantile(.05), p25=s.quantile(.25),
                med=s.median(), p75=s.quantile(.75), p95=s.quantile(.95),
                mean=s.mean())


p("-" * 78)
p("Feature distributions  [p05 | p25 | median | p75 | p95]   (mean)")
p("-" * 78)
hdr = f"{'feature':<22}{'group':<14}{'p05':>9}{'p25':>9}{'med':>9}{'p75':>9}{'p95':>9}{'mean':>10}"
for feat in FEATS:
    p("")
    p(f"{feat}")
    p(hdr)
    for name, df in [("real-FP (raw)", emp),
                     ("real-FP (ft)", empft),
                     ("syn TRUE", syn_pos),
                     ("syn FALSE", syn_neg)]:
        if df is None or feat not in df.columns:
            continue
        st = qstats(df[feat])
        if st is None:
            continue
        p(f"{'':<22}{name:<14}{st['p05']:>9.2f}{st['p25']:>9.2f}"
          f"{st['med']:>9.2f}{st['p75']:>9.2f}{st['p95']:>9.2f}{st['mean']:>10.2f}")

# ---------------------------------------------------------------------------
# Overlap / separability metric: for each feature, how much of the real-FP
# mass lies inside the central 90% of the synthetic-TRUE distribution?
# A high fraction => the feature cannot separate them => the network needs
# to see real FP directly (the whole point of on-real-bg training).
# ---------------------------------------------------------------------------
p("")
p("-" * 78)
p("Real-FP mass falling inside the synthetic-TRUE [p05,p95] band")
p("(high => feature does NOT separate => need real-bg training signal)")
p("-" * 78)
p(f"{'feature':<22}{'syn_lo':>10}{'syn_hi':>10}{'%FPraw_in':>12}{'%FPft_in':>12}")
for feat in FEATS:
    if feat not in syn_pos.columns:
        continue
    sp = pd.to_numeric(syn_pos[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(sp) < 10:
        continue
    lo, hi = sp.quantile(.05), sp.quantile(.95)
    def frac_in(df):
        if df is None or feat not in df.columns:
            return float("nan")
        v = pd.to_numeric(df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) == 0:
            return float("nan")
        return 100.0 * ((v >= lo) & (v <= hi)).mean()
    p(f"{feat:<22}{lo:>10.2f}{hi:>10.2f}{frac_in(emp):>11.1f}%{frac_in(empft):>11.1f}%")

# ---------------------------------------------------------------------------
# Linear-separability proxy: a single logistic-regression AUC, real-FP(ft) vs
# syn TRUE, on the 12 features. Low AUC for any single feature, high for the
# joint set, tells us the discrimination IS learnable from these inputs (the
# net sees a strictly richer representation than these 72 features).
# ---------------------------------------------------------------------------
p("")
p("-" * 78)
p("Discriminability real-FP(ft) vs syn-TRUE  (held-out AUC)")
p("-" * 78)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    cols = [c for c in FEATS if c in empft.columns and c in syn_pos.columns]
    A = empft[cols].apply(pd.to_numeric, errors="coerce")
    B = syn_pos[cols].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([A, B], ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=4, random_state=0)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    p(f"joint 12-feature RF AUC (real-FP vs syn-TRUE): {auc:.3f}")
    p("per-feature single-split AUC:")
    for c in cols:
        v = X[c].values
        a = roc_auc_score(y, v)
        a = max(a, 1 - a)
        p(f"  {c:<24} AUC~{a:.3f}")
    imp = sorted(zip(cols, clf.feature_importances_), key=lambda t: -t[1])
    p("top RF feature importances:")
    for c, i in imp[:6]:
        p(f"  {c:<24} {i:.3f}")
except Exception as e:
    p(f"[skipped sklearn block: {e}]")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
p("")
p(f"[written] {OUT}")
