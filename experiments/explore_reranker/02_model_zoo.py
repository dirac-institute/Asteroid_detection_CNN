"""Step 2 — train a zoo of rerankers on the EXACT FT pool/split and evaluate
on the SAME held-out-empty FP@posR metric as fp_fix2.txt.

All models consume the cached arrays from step 1 (X,y,grp on the FT training
pool; Xs,ys synthetic recall pool; Xe,fro,n_ccd held-out empties). The metric
is recomputed with the established _fp_gen / _pos_recall formulas, so every
row is directly comparable to the promoted FT RandomForest.

Convention note: the baseline's posR (_pos_recall) is *in-sample* — the FT RF
is trained on the full syn pool and recall is read back on that same pool.
To stay apples-to-apples we report posR the same way; step 3 adds an honest
group-disjoint OOB cross-check so we don't fool ourselves on recall.

Models:
  base_RF_FT            promoted baseline (cached from step 1)
  RF_balanced           RF retrained here (identity check vs base)
  RF_w5 / RF_deep       RF hyperparam variants
  HGB                   HistGradientBoosting, class_weight=balanced
  HGB_mono              + monotone>=0 on SNR/flux/prob features
  HGB_hnm               HGB + 2 rounds iterative hard-negative mining
  HGB_mono_hnm          mono + hard-neg mining
  GB_sk                 sklearn GradientBoosting (subsample, deviance)
  HGB_cal_iso           HGB_mono_hnm + isotonic calibration (group-disjoint)

Saves per-model THRS rows to _zoo_rows.json and feature importances.
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

from harness import eval_model_scores, score_predict_proba  # noqa: E402
from ADCNN.evaluation.fp_analysis import THRS, FEATS  # noqa: E402

from sklearn.ensemble import (  # noqa: E402
    RandomForestClassifier, HistGradientBoostingClassifier,
    GradientBoostingClassifier)
# (isotonic calibration done by hand below — rank-preserving, see note)

RNG = 0

# Monotone-positive: more signal -> more likely a real trail. Conservative:
# only true SNR / flux / NN-probability / aggregated-response features.
_MONO_SUBSTR = ("snr", "flux", "max_p", "mean_p", "top5_mean_p",
                "integrated_logit", "or_agg", "masnr", "maflux",
                "lpca_snr", "lpca_flux", "loc_sum_z", "loc_max_z")
_MONO_EXCLUDE = {"mf_snr_norm"}  # panel-relative, sign not monotone


def mono_vector():
    v = np.zeros(len(FEATS), dtype=int)
    for i, f in enumerate(FEATS):
        if f in _MONO_EXCLUDE:
            continue
        if any(s in f for s in _MONO_SUBSTR):
            v[i] = 1
    return v


def balanced_w(y):
    n = len(y)
    n1 = max(int(y.sum()), 1)
    n0 = max(n - n1, 1)
    w = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
    return w.astype(np.float64)


def hard_neg_mine(make, X, y, n_rounds=2, top_frac=0.25, boost=3.0):
    """Iterative hard-negative mining: train, find the highest-scored TRUE
    negatives (label 0), multiply their sample weight, retrain. Positives
    keep the class-balance weight throughout (recall must not drop)."""
    w = balanced_w(y)
    clf = None
    for r in range(n_rounds + 1):
        clf = make()
        clf.fit(X, y, sample_weight=w)
        if r == n_rounds:
            break
        s = clf.predict_proba(X)[:, 1]
        neg = np.where(y == 0)[0]
        cut = np.quantile(s[neg], 1.0 - top_frac)
        hard = neg[s[neg] >= cut]
        w[hard] *= boost
    return clf


def main():
    t0 = time.time()
    A = np.load(HERE / "_arrays.npz")
    X, y, grp = A["X"], A["y"].astype(np.int8), A["grp"]
    Xs, ys = A["Xs"], A["ys"].astype(np.int8)
    Xe, fro = A["Xe"], A["fro"]
    n_ccd = int(A["n_ccd"])
    base_s_emp, base_s_syn = A["base_s_emp"], A["base_s_syn"]
    print(f"[arrays] X={X.shape} pos={int(y.sum())} neg={int((y==0).sum())} "
          f"| Xs={Xs.shape} Xe={Xe.shape} n_ccd={n_ccd}", flush=True)

    mono = mono_vector()
    print(f"[mono] {int(mono.sum())} monotone-positive features: "
          f"{[FEATS[i] for i in np.where(mono)[0]][:8]}...", flush=True)

    results = {}
    fimp = {}

    def record(name, model, t_train):
        s_emp = score_predict_proba(model, Xe)
        s_syn = score_predict_proba(model, Xs)
        rows = eval_model_scores(s_emp, fro, n_ccd, s_syn, ys, THRS)
        results[name] = rows
        print(f"\n=== {name}  (train {t_train:.0f}s) ===", flush=True)
        print("  thr |  FPgen/CCD |  posR |  negK", flush=True)
        for tt, fp, pr, nk in rows:
            print(f" {tt:>4.2f} | {fp:>10.1f} | {pr:>5.3f} | {nk:>5.3f}",
                  flush=True)
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            fimp[name] = sorted(
                zip(FEATS, [float(v) for v in fi]),
                key=lambda z: -z[1])[:15]

    # 0) cached promoted baseline (from step 1, recomputed identically)
    rows = eval_model_scores(base_s_emp, fro, n_ccd, base_s_syn, ys, THRS)
    results["base_RF_FT"] = rows
    print("\n=== base_RF_FT (promoted, cached) ===")
    for tt, fp, pr, nk in rows:
        print(f" {tt:>4.2f} | {fp:>10.1f} | {pr:>5.3f} | {nk:>5.3f}")

    w_bal = balanced_w(y)

    # 1) RF balanced identity check
    t = time.time()
    m = RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5,
        class_weight="balanced", n_jobs=32, random_state=RNG)
    m.fit(X, y)
    record("RF_balanced", m, time.time() - t)

    # 2) RF milder positive weight
    t = time.time()
    m = RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5,
        class_weight={0: 1.0, 1: 8.0}, n_jobs=32, random_state=RNG)
    m.fit(X, y)
    record("RF_w8", m, time.time() - t)

    # 3) RF deeper / more trees
    t = time.time()
    m = RandomForestClassifier(
        n_estimators=900, max_depth=24, min_samples_leaf=2,
        class_weight="balanced", n_jobs=32, random_state=RNG)
    m.fit(X, y)
    record("RF_deep", m, time.time() - t)

    # 4) HGB balanced
    def mk_hgb(monoc=None):
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=0.3, early_stopping=False,
            monotonic_cst=monoc, random_state=RNG)
    t = time.time()
    m = mk_hgb()
    m.fit(X, y, sample_weight=w_bal)
    record("HGB", m, time.time() - t)

    # 5) HGB + monotone constraints
    t = time.time()
    m = mk_hgb(monoc=mono)
    m.fit(X, y, sample_weight=w_bal)
    record("HGB_mono", m, time.time() - t)

    # 6) HGB + hard-negative mining
    t = time.time()
    m = hard_neg_mine(lambda: mk_hgb(), X, y, n_rounds=2)
    record("HGB_hnm", m, time.time() - t)

    # 7) HGB mono + hard-negative mining
    t = time.time()
    m_mh = hard_neg_mine(lambda: mk_hgb(monoc=mono), X, y, n_rounds=2)
    record("HGB_mono_hnm", m_mh, time.time() - t)

    # 8) sklearn GradientBoosting (slower; subsample for speed/variance)
    t = time.time()
    # n_estimators kept modest: sklearn GB is single-threaded & slow on
    # 117k rows; the 400-tree run (see _zoo_console.txt) already showed
    # shallow GB does NOT domain-overfit (worse than RF) — 150 trees
    # reproduces that conclusion at ~1/3 the wall time.
    m = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=3,
        subsample=0.6, min_samples_leaf=40, random_state=RNG)
    m.fit(X, y, sample_weight=w_bal)
    record("GB_sk", m, time.time() - t)

    # Persist the zoo table BEFORE the (slow, optional) calibration step so a
    # crash there never loses the head-to-head results.
    out = {k: [[float(c) for c in r] for r in v]
           for k, v in results.items()}
    (HERE / "_zoo_rows.json").write_text(json.dumps(out, indent=1))
    (HERE / "_zoo_fimp.json").write_text(json.dumps(fimp, indent=1))

    # 9) Probability calibration (isotonic, group-disjoint).
    #    NOTE: isotonic calibration is strictly rank-preserving, so it CANNOT
    #    change a rank-based FP@posR metric — included only for completeness.
    #    CalibratedClassifierCV(cv=GroupKFold) cannot route `groups` through
    #    sample_weight in this sklearn version, so we do an explicit
    #    panel-disjoint prefit + isotonic regressor by hand.
    from sklearn.isotonic import IsotonicRegression
    t = time.time()
    ids = np.array(sorted(np.unique(grp)))
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    fit_ids = set(ids[: int(len(ids) * 0.75)])
    cal_mask = ~np.array([g in fit_ids for g in grp])
    fit_mask = ~cal_mask

    class GroupCalibrated:
        def __init__(self, base, iso):
            self.base, self.iso = base, iso

        def predict_proba(self, Z):
            raw = self.base.predict_proba(Z)[:, 1]
            p = self.iso.predict(raw)
            return np.column_stack([1.0 - p, p])

    base = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=0.3,
        early_stopping=False, monotonic_cst=mono, random_state=RNG)
    base.fit(X[fit_mask], y[fit_mask], sample_weight=w_bal[fit_mask])
    raw_cal = base.predict_proba(X[cal_mask])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y[cal_mask], sample_weight=w_bal[cal_mask])
    record("HGB_mono_iso", GroupCalibrated(base, iso), time.time() - t)

    out = {k: [[float(c) for c in r] for r in v]
           for k, v in results.items()}
    (HERE / "_zoo_rows.json").write_text(json.dumps(out, indent=1))
    (HERE / "_zoo_fimp.json").write_text(json.dumps(fimp, indent=1))
    print(f"\n[saved] _zoo_rows.json _zoo_fimp.json  "
          f"(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
