"""Step 6 — WHICH features make syn-pool vs empty-CCD perfectly separable?

The domain probe (step 5) hit AUC=1.0: the synthetic-injection candidate
population and the real-empty-CCD candidate population are perfectly separable
in feature space, irrespective of the asteroid label. That makes the
held-out-empty FP@posR proxy degenerate for high-capacity models (they win by
keying on the synthetic-vs-real domain, not on faint-trail sensitivity).

Here we localize the leak:
  (1) per-feature univariate domain AUC (syn-pool rows=0, empty rows=1)
  (2) greedy: how FEW features still give domain AUC≈1.0
  (3) restrict ALL models to the domain-robust feature subset (drop features
      with |domain_auc-0.5|>0.15) and rerun the calibration-free FP@posR=1.0
      curve — does the GBM advantage survive once it can no longer cheat on
      domain? This is the real apples-to-apples test of the IDEA.
"""
from __future__ import annotations

import json
import sys
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

import importlib
zoo = importlib.import_module("02_model_zoo")


def cut_for_recall(s_syn, ys, target_r=1.0):
    sp = np.sort(s_syn[ys == 1])
    npos = len(sp)
    k = min(max(int(np.ceil(target_r * npos)), 1), npos)
    return sp[npos - k]


def fp_gen(s_emp, fro, n_ccd, cut):
    return int(((s_emp >= cut) & (fro < EPS_GENUINE)).sum()) / n_ccd


def main():
    A = np.load(HERE / "_arrays.npz")
    X, y = A["X"], A["y"].astype(np.int8)
    Xs, ys = A["Xs"], A["ys"].astype(np.int8)
    Xe, fro = A["Xe"], A["fro"]
    n_ccd = int(A["n_ccd"])

    # domain target: syn-pool rows (0) vs empty-CCD rows (1)
    Xd = np.vstack([Xs, Xe])
    yd = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xe))]).astype(int)

    # (1) univariate domain AUC per feature
    uni = []
    for j, f in enumerate(FEATS):
        col = Xd[:, j]
        if np.allclose(col, col[0]):
            a = 0.5
        else:
            a = roc_auc_score(yd, col)
            a = max(a, 1.0 - a)  # direction-agnostic separability
        uni.append((f, float(a)))
    uni.sort(key=lambda z: -z[1])
    print("[univariate domain AUC] top 20 (1.0 = perfectly tells "
          "synthetic vs real-empty apart):")
    for f, a in uni[:20]:
        print(f"  {a:.4f}  {f}")

    near05 = [f for f, a in uni if a <= 0.65]
    print(f"\n[domain-robust feats |auc-0.5|<=0.15]: {len(near05)} / "
          f"{len(FEATS)}")
    print("  ", near05)

    # (2) does even ONE feature separate domains?
    best_f, best_a = uni[0]
    print(f"\n[single best domain feature] {best_f} AUC={best_a:.4f}")

    # (3) restrict to domain-robust subset; rerun FP@posR=1.0 curve
    rob_idx = [j for j, f in enumerate(FEATS) if f in set(near05)]
    if len(rob_idx) < 4:
        print("\n[restricted] <4 domain-robust features — the proxy cannot "
              "support a fair high-capacity comparison at all.")
    Xr, Xsr, Xer = X[:, rob_idx], Xs[:, rob_idx], Xe[:, rob_idx]
    print(f"\n[restricted to {len(rob_idx)} domain-robust feats] "
          "calibration-free FP/CCD @ posR=1.0:")

    rep = {"univariate_top": uni[:25], "robust_feats": near05,
           "single_best": [best_f, best_a], "restricted": {}}

    # domain AUC on the restricted set (is it still separable?)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(4, shuffle=True, random_state=0)
    oof = np.zeros(len(yd))
    Xdr = Xd[:, rob_idx]
    for tr, te in skf.split(Xdr, yd):
        dc = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=0)
        dc.fit(Xdr[tr], yd[tr])
        oof[te] = dc.predict_proba(Xdr[te])[:, 1]
    dom_auc_r = roc_auc_score(yd, oof)
    print(f"  [domain probe on restricted set] AUC = {dom_auc_r:.4f}")
    rep["restricted"]["domain_auc"] = float(dom_auc_r)

    w = zoo.balanced_w(y)
    for name, mk, sw in [
        ("RF_balanced",
         lambda: RandomForestClassifier(
             n_estimators=500, max_depth=14, min_samples_leaf=5,
             class_weight="balanced", n_jobs=32, random_state=0), False),
        ("HGB",
         lambda: HistGradientBoostingClassifier(
             max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
             min_samples_leaf=40, l2_regularization=0.3,
             early_stopping=False, random_state=0), True),
    ]:
        m = mk()
        if sw:
            m.fit(Xr, y, sample_weight=w)
        else:
            m.fit(Xr, y)
        ss = score_predict_proba(m, Xsr)
        se = score_predict_proba(m, Xer)
        auc = roc_auc_score(ys, ss)
        cut = cut_for_recall(ss, ys, 1.0)
        fp = fp_gen(se, fro, n_ccd, cut)
        print(f"  {name:<12} AUC={auc:.5f}  FP_gen/CCD@posR1.0={fp:.2f}  "
              f"(cut={cut:.4g})")
        rep["restricted"][name] = {"auc": float(auc), "fp": float(fp),
                                   "cut": float(cut)}

    (HERE / "_domain.json").write_text(json.dumps(rep, indent=1))
    print(f"\n[saved] _domain.json")


if __name__ == "__main__":
    main()
