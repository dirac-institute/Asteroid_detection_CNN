"""DIAGNOSTIC (not a deployable model): is real trail vs real residual separable in
the v2 candidate features at all, and does the synthetic-trained RF just fail to use
the discriminative features?

Two questions:
  (1) per-feature AUC(real TP vs real FP) — which features carry real discrimination,
      vs the synthetic-RF's feature_importances_ (a feature with high real-AUC but low
      synthetic importance = an underused, feature-engineering opportunity).
  (2) ORACLE separability ceiling: cross-validated classifier trained directly on
      real TP vs real FP (LEAKY by construction — diagnostic only, never deployed).
      High oracle AUC => info is present, a better-matched stage-2 can exploit it.
      Low => features insufficient -> need cutout-CNN or temporal linking.
test_real is only READ; the oracle is a separability probe, not a shipped model.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.inference.diffim_postproc_v2 import RF_FEATURES_V2, load_rf

CACHE = Path("/sdf/scratch/users/m/mrakovci/e2e_cache")
EMPTY = Path("/sdf/scratch/users/m/mrakovci/realistic/empty_fp/parts")
OUT = REPO / "experiments/explore_simreal_gap"
FEATS = list(RF_FEATURES_V2)


def clean(df):
    return df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def main():
    rf = load_rf(str(OUT / "rf_postproc_v2_realistic_neg5.pkl"))
    imp = pd.Series(rf.feature_importances_, index=FEATS)

    # prefer full-panel TP (same inference path as the empty FP -> no size artifact)
    tp_fp_parts = sorted(glob.glob("/sdf/scratch/users/m/mrakovci/realistic/real_tp_fp/parts/tp_*.parquet"))
    if tp_fp_parts:
        tp_all = clean(pd.concat([pd.read_parquet(f) for f in tp_fp_parts], ignore_index=True))
        print(f"[TP] full-panel path ({len(tp_all)} truth-cands) -- comparable to FP", flush=True)
    else:
        tp_all = clean(pd.read_parquet(CACHE / "real_feats_realistic.parquet"))
        print("[TP] WARNING windowed-path TP (size features confounded)", flush=True)
    fp_all_df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(EMPTY / "empty_*.parquet")))],
                          ignore_index=True)
    fp_all = clean(fp_all_df)
    # CONDITION on RF survival: the operationally relevant FP are the candidates the
    # RF keeps (the ~68/panel that pass thr), NOT the ~2000/panel raw noise blobs the
    # RF already rejects (those separate trivially on size = meaningless). Compare the
    # *surviving* TP and FP -- the hard, trail-like residuals.
    THR = 0.3
    stp = rf.predict_proba(tp_all)[:, 1]; sfp = rf.predict_proba(fp_all)[:, 1]
    tp = tp_all[stp >= THR]
    fp_surv = fp_all[sfp >= THR]
    rng = np.random.default_rng(0)
    fp = fp_surv[rng.choice(len(fp_surv), min(8000, len(fp_surv)), replace=False)]
    X = np.vstack([tp, fp]); y = np.r_[np.ones(len(tp)), np.zeros(len(fp))]
    print(f"surviving (RF>= {THR}): real TP={len(tp)}/{len(tp_all)}  "
          f"real FP={len(fp_surv)} (sampled {len(fp)})", flush=True)

    # (1) per-feature real AUC
    rows = []
    for j, f in enumerate(FEATS):
        a = roc_auc_score(y, X[:, j])
        rows.append((f, max(a, 1 - a), imp[f]))   # direction-agnostic
    t = pd.DataFrame(rows, columns=["feat", "real_auc", "synth_imp"]).sort_values("real_auc", ascending=False)
    print("\n=== top real-TP-vs-FP discriminative features (vs synthetic RF importance) ===", flush=True)
    print(t.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    print("\n underused (high real_auc, low synth_imp):", flush=True)
    t["gap"] = t.real_auc - 0.5 - 50 * t.synth_imp
    print(t.sort_values("gap", ascending=False).head(8)[["feat", "real_auc", "synth_imp"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    # (2) oracle separability ceiling (CV on real TP vs FP) -- diagnostic only
    clf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                 class_weight="balanced", n_jobs=32, random_state=0)
    cv = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                         scoring="roc_auc")
    print(f"\n=== ORACLE separability ceiling (5-fold CV on real TP vs FP) ===", flush=True)
    print(f"  AUC = {cv.mean():.3f} +/- {cv.std():.3f}", flush=True)
    print("  (interpretation: >>0.5 => discrimination EXISTS in features -> feature-eng / "
          "real-matched stage-2 viable; ~0.5 => features insufficient -> cutout-CNN / linking)", flush=True)
    t.to_csv(OUT / "real_separability.csv", index=False)
    print("SEPARABILITY DONE", flush=True)


if __name__ == "__main__":
    main()
