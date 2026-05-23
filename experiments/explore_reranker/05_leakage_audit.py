"""Step 5 — leakage / artifact audit for the GBM Pareto win.

The metric defines a genuine FP as (score>=t) & (frac_real_label_overlap <
EPS_GENUINE=0.05). `frac_real_label_overlap` is ALSO RF_FEATURES_V2[9] — a
training feature. On empty CCDs almost every candidate has fro==0 (it only
fires on real stack residuals). The training pool's negatives, however, are a
mix: synthetic FPs (fro often 0) + empty-CCD hard negs (fro≈0) but the
positives are synthetic injections whose fro is also ~0. So fro should NOT be
a strong discriminator — but if a high-capacity GBM keys on it (or on any
feature that is trivially separable between the syn-pool and empty CCDs
because of how they were generated), the held-out-empty FP would be
artificially crushed and the "win" is an artifact, not a real second-stage
improvement.

Checks:
  (1) Permutation-style feature importance for HGB_mono_hnm: drop fro (zero it
      at train+eval) and re-measure FP@posR=1.0. If FP barely changes, the win
      does not hinge on fro leakage.
  (2) Distribution shift probe: train a domain classifier (syn-pool rows vs
      empty-CCD rows, ignore the real label) on the 72 feats. Its AUC tells us
      how trivially separable the two domains are. If ~1.0, the GBM can cheat
      by domain, not by genuine-trail discrimination -> the metric itself is
      degenerate for ALL models (incl. the RF baseline) and the comparison is
      only valid in so far as both models face the same degeneracy.
  (3) The decisive test: FP at MATCHED synthetic recall using the model's own
      score sweep (calibration-free), AND a hard-negatives-only sanity: on the
      held-out empties, what fraction of *all* empty candidates (regardless of
      fro) does each model keep at the cut that holds posR=1.0? This is the
      true added-FP rate a deployed 2nd stage pays, robust to the fro
      definition.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from harness import score_predict_proba  # noqa: E402
from ADCNN.evaluation.fp_analysis import FEATS, EPS_GENUINE  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import importlib
zoo = importlib.import_module("02_model_zoo")

FRO_IDX = FEATS.index("frac_real_label_overlap")


def cut_for_recall(s_syn, ys, target_r=1.0):
    sp = np.sort(s_syn[ys == 1])
    npos = len(sp)
    k = min(max(int(np.ceil(target_r * npos)), 1), npos)
    return sp[npos - k]


def fp_all_and_gen(s_emp, fro, n_ccd, cut):
    keep = s_emp >= cut
    fp_all = int(keep.sum()) / n_ccd
    fp_gen = int((keep & (fro < EPS_GENUINE)).sum()) / n_ccd
    return fp_all, fp_gen


def main():
    A = np.load(HERE / "_arrays.npz")
    X, y, grp = A["X"], A["y"].astype(np.int8), A["grp"]
    Xs, ys = A["Xs"], A["ys"].astype(np.int8)
    Xe, fro = A["Xe"], A["fro"]
    n_ccd = int(A["n_ccd"])
    base_s_emp, base_s_syn = A["base_s_emp"], A["base_s_syn"]
    mono = zoo.mono_vector()
    w = zoo.balanced_w(y)
    rep = {}

    # fro stats across the three populations
    syn_pos_fro = Xs[ys == 1, FRO_IDX]
    syn_neg_fro = Xs[ys == 0, FRO_IDX]
    print(f"[fro] syn_pos: mean={syn_pos_fro.mean():.4f} "
          f">=EPS frac={np.mean(syn_pos_fro >= EPS_GENUINE):.4f}")
    print(f"[fro] syn_neg: mean={syn_neg_fro.mean():.4f} "
          f">=EPS frac={np.mean(syn_neg_fro >= EPS_GENUINE):.4f}")
    print(f"[fro] emp    : mean={fro.mean():.4f} "
          f">=EPS frac={np.mean(fro >= EPS_GENUINE):.4f}")
    rep["fro_stats"] = {
        "syn_pos_ge_eps": float(np.mean(syn_pos_fro >= EPS_GENUINE)),
        "syn_neg_ge_eps": float(np.mean(syn_neg_fro >= EPS_GENUINE)),
        "emp_ge_eps": float(np.mean(fro >= EPS_GENUINE))}

    # (1) drop-fro retrain of the best GBM and the RF baseline
    def fit_hmh(Xtr):
        return zoo.hard_neg_mine(
            lambda: HistGradientBoostingClassifier(
                max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                min_samples_leaf=40, l2_regularization=0.3,
                early_stopping=False, monotonic_cst=mono, random_state=0),
            Xtr, y, n_rounds=2)

    for tag, zero_fro in [("with_fro", False), ("no_fro", True)]:
        Xt, Xst, Xet = X.copy(), Xs.copy(), Xe.copy()
        if zero_fro:
            Xt[:, FRO_IDX] = 0.0
            Xst[:, FRO_IDX] = 0.0
            Xet[:, FRO_IDX] = 0.0
        m = fit_hmh(Xt)
        ss = score_predict_proba(m, Xst)
        se = score_predict_proba(m, Xet)
        cut = cut_for_recall(ss, ys, 1.0)
        fa, fg = fp_all_and_gen(se, fro, n_ccd, cut)
        auc = roc_auc_score(ys, ss)
        print(f"\n[HGB_mono_hnm {tag}] posR=1.0 cut={cut:.4g} "
              f"AUC={auc:.5f} FP_all/CCD={fa:.2f} FP_gen/CCD={fg:.2f}")
        rep[f"hmh_{tag}"] = {"cut": float(cut), "auc": float(auc),
                             "fp_all": fa, "fp_gen": fg}

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5,
        class_weight="balanced", n_jobs=32, random_state=0).fit(X, y)
    rss = score_predict_proba(rf, Xs)
    rse = score_predict_proba(rf, Xe)
    rcut = cut_for_recall(rss, ys, 1.0)
    rfa, rfg = fp_all_and_gen(rse, fro, n_ccd, rcut)
    print(f"\n[RF_balanced] posR=1.0 cut={rcut:.4g} "
          f"AUC={roc_auc_score(ys, rss):.5f} "
          f"FP_all/CCD={rfa:.2f} FP_gen/CCD={rfg:.2f}")
    rep["rf_balanced"] = {"cut": float(rcut),
                          "auc": float(roc_auc_score(ys, rss)),
                          "fp_all": rfa, "fp_gen": rfg}
    # also recompute baseline (cached) at calibration-free posR=1.0 cut
    bcut = cut_for_recall(base_s_syn, ys, 1.0)
    bfa, bfg = fp_all_and_gen(base_s_emp, fro, n_ccd, bcut)
    print(f"[base_RF_FT ] posR=1.0 cut={bcut:.4g} "
          f"FP_all/CCD={bfa:.2f} FP_gen/CCD={bfg:.2f}")
    rep["base_RF_FT"] = {"cut": float(bcut), "fp_all": bfa, "fp_gen": bfg}

    # (2) domain classifier: syn-pool vs empty-CCD rows (label=domain)
    Xd = np.vstack([Xs, Xe])
    yd = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xe))]).astype(int)
    gd = np.concatenate([np.full(len(Xs), -1),  # syn = one big group split
                         (fro * 0).astype(int)])  # placeholder
    # group by syn panel vs empty CCD is overkill; use simple stratified KFold
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    oof = np.zeros(len(yd))
    for tr, te in skf.split(Xd, yd):
        dc = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=0)
        dc.fit(Xd[tr], yd[tr])
        oof[te] = dc.predict_proba(Xd[te])[:, 1]
    dom_auc = roc_auc_score(yd, oof)
    print(f"\n[domain probe] syn-pool vs empty-CCD AUC = {dom_auc:.4f}  "
          f"(1.0 => trivially separable; degenerate for ALL models)")
    rep["domain_auc"] = float(dom_auc)

    (HERE / "_leakage.json").write_text(json.dumps(rep, indent=1))
    print(f"\n[saved] _leakage.json")


if __name__ == "__main__":
    main()
