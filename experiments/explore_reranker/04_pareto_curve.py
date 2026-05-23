"""Step 4 — the rigorous, calibration-free comparison.

The fixed-THRS table in step 2 is confounded: RF(class_weight=balanced) and a
sample-weighted GBM put their decision boundary at very different numeric
scores, so reading both at thr=0.10 is NOT apples-to-apples. The honest
question is threshold-free:

    at the SAME synthetic true-trail recall (posR), what genuine FP/CCD does
    each model produce on the held-out empties?

For each model we sweep its OWN score over a dense grid, and for every target
posR in {1.000, 0.999, 0.99, 0.98, 0.95, 0.90} we find the score cut that
*just* achieves that recall on the synthetic pool, then read genuine FP/CCD on
the held-out empties at that cut. This removes the calibration confound and is
the true Pareto comparison. We also report ROC-style AUC and the empty-CCD FP
at the cut that holds posR==1.000 exactly (the project's headline operating
regime).

Skeptic guards:
  * recall pool == _pos_recall's pool (pos | hard-neg<0.5)  [from harness]
  * FP == _fp_gen's genuine definition (fro<EPS) per held-out CCD
  * cut chosen on synthetic recall ONLY; FP read on disjoint empties
  * also recompute everything for the cached promoted baseline so the
    baseline row uses the identical sweep machinery (no formula asymmetry).
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

import importlib
zoo = importlib.import_module("02_model_zoo")

TARGET_R = [1.0, 0.999, 0.99, 0.98, 0.95, 0.90]


def fp_at_recall(s_syn, ys, s_emp, fro, n_ccd, target_r):
    """Lowest-FP score cut whose synthetic recall >= target_r.

    We scan candidate cuts = the score values of synthetic POSITIVES sorted
    descending; recall at cut c = mean(s_pos >= c). The minimal c achieving
    recall>=target is the (1-target) quantile of the positive scores. At that
    c, genuine FP/CCD on held-out empties = #(s_emp>=c & fro<EPS)/n_ccd.
    Returns (cut, achieved_recall, fp_per_ccd, neg_kept_frac).
    """
    sp = np.sort(s_syn[ys == 1])          # ascending
    npos = len(sp)
    # smallest cut s.t. fraction of positives >= cut is >= target_r
    # => keep ceil(target_r*npos) highest positives; cut = the
    #    (npos - k)-th order stat (0-idx) where k = #kept.
    k = int(np.ceil(target_r * npos))
    k = min(max(k, 1), npos)
    cut = sp[npos - k]                     # threshold (>= cut keeps k of them)
    ach = float((s_syn[ys == 1] >= cut).mean())
    gen = (fro < EPS_GENUINE)
    fp = int(((s_emp >= cut) & gen).sum()) / n_ccd
    negk = float((s_syn[ys == 0] >= cut).mean())
    return float(cut), ach, float(fp), negk


def curve(name, s_syn, ys, s_emp, fro, n_ccd):
    auc_syn = roc_auc_score(ys, s_syn)
    rows = []
    for tr in TARGET_R:
        cut, ach, fp, negk = fp_at_recall(s_syn, ys, s_emp, fro, n_ccd, tr)
        rows.append({"target_r": tr, "cut": cut, "ach_r": ach,
                     "fp_ccd": fp, "negK": negk})
    print(f"\n=== {name}   syn-pool AUC={auc_syn:.5f} ===", flush=True)
    print("  tgtR  achR |  FP/CCD  negK   (cut)", flush=True)
    for r in rows:
        print(f" {r['target_r']:>5.3f} {r['ach_r']:>5.3f} | "
              f"{r['fp_ccd']:>7.2f} {r['negK']:>6.4f}  ({r['cut']:.4g})",
              flush=True)
    return {"auc": float(auc_syn), "rows": rows}


def main():
    t0 = time.time()
    A = np.load(HERE / "_arrays.npz")
    X, y, grp = A["X"], A["y"].astype(np.int8), A["grp"]
    Xs, ys = A["Xs"], A["ys"].astype(np.int8)
    Xe, fro = A["Xe"], A["fro"]
    n_ccd = int(A["n_ccd"])
    base_s_emp, base_s_syn = A["base_s_emp"], A["base_s_syn"]
    mono = zoo.mono_vector()
    w = zoo.balanced_w(y)

    out = {}

    # promoted baseline (cached scores -> identical sweep machinery)
    out["base_RF_FT"] = curve("base_RF_FT", base_s_syn, ys,
                              base_s_emp, fro, n_ccd)

    # RF balanced retrained (sanity: should match base curve closely)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5,
        class_weight="balanced", n_jobs=32, random_state=0).fit(X, y)
    out["RF_balanced"] = curve(
        "RF_balanced",
        score_predict_proba(rf, Xs), ys,
        score_predict_proba(rf, Xe), fro, n_ccd)

    # HGB balanced
    hgb = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=0.3, early_stopping=False,
        random_state=0).fit(X, y, sample_weight=w)
    out["HGB"] = curve(
        "HGB", score_predict_proba(hgb, Xs), ys,
        score_predict_proba(hgb, Xe), fro, n_ccd)

    # HGB mono
    hgm = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=0.3, early_stopping=False,
        monotonic_cst=mono, random_state=0).fit(X, y, sample_weight=w)
    out["HGB_mono"] = curve(
        "HGB_mono", score_predict_proba(hgm, Xs), ys,
        score_predict_proba(hgm, Xe), fro, n_ccd)

    # HGB mono + hard-neg mining (best in step 2)
    hmh = zoo.hard_neg_mine(
        lambda: HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=0.3,
            early_stopping=False, monotonic_cst=mono, random_state=0),
        X, y, n_rounds=2)
    out["HGB_mono_hnm"] = curve(
        "HGB_mono_hnm", score_predict_proba(hmh, Xs), ys,
        score_predict_proba(hmh, Xe), fro, n_ccd)

    (HERE / "_pareto.json").write_text(json.dumps(out, indent=1))
    print(f"\n[saved] _pareto.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
