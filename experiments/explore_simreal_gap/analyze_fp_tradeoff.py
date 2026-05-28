"""Offline TP-vs-FP operating-curve analysis (no GPU). Goal: hold the real TPs while
cutting the 68 FP/panel. For each RF (negative-subsample ratio) and threshold, compute:
  real TP recall   = frac of real truth-cands (72) scored >= thr
  real FP / panel  = (# empty-panel candidates scored >= thr) / 150
Real TP features = real_feats_realistic.parquet; real FP features = empty_fp parquet.
Both come from the realistic seg_model. Synthetic-only RF training; test_real only READ.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_rf_leakage"))
import improve_rf as ir
from ADCNN.inference.diffim_postproc_v2 import RF_FEATURES_V2

CACHE = Path("/sdf/scratch/users/m/mrakovci/e2e_cache")
EMPTY = Path("/sdf/scratch/users/m/mrakovci/realistic/empty_fp/parts")
OUT = REPO / "experiments/explore_simreal_gap"
FEATS = list(RF_FEATURES_V2)
N_EMPTY = 150


def rf_for(X, y, neg_ratio, leaf=5, seed=0):
    if neg_ratio is None:
        Xs, ys = X, y
    else:
        rng = np.random.default_rng(seed)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        keep = np.concatenate([pos, rng.choice(neg, min(len(neg), neg_ratio * len(pos)), replace=False)])
        Xs, ys = X[keep], y[keep]
    return RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=leaf,
        max_features="sqrt", class_weight="balanced", n_jobs=32, random_state=0).fit(Xs, ys)


def main():
    vcand = pd.read_parquet(CACHE / "vcand.parquet"); vlab = np.load(CACHE / "vlab.npy")
    X, y, _ = ir.build_pool(vcand, vlab)
    tp = pd.read_parquet(OUT / "test_real_realistic" / "real_feats_realistic.parquet") \
        if (OUT / "test_real_realistic" / "real_feats_realistic.parquet").exists() \
        else pd.read_parquet(CACHE / "real_feats_realistic.parquet")
    Xtp = tp[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    emp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(EMPTY / "empty_*.parquet")))],
                    ignore_index=True)
    Xfp = emp[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    nTP = len(Xtp)
    print(f"real TP truth-cands={nTP}  empty-panel candidates={len(Xfp)} over {N_EMPTY} panels", flush=True)

    thrs = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    print(f"\n{'RF':>10} | " + "  ".join(f"thr{t:.2f}(TP/FPp)" for t in thrs), flush=True)
    rows = []
    for tag, ratio in [("neg5", 5), ("neg10", 10), ("neg20", 20), ("neg50", 50), ("all", None)]:
        rf = rf_for(X, y, ratio)
        stp = rf.predict_proba(Xtp)[:, 1]; sfp = rf.predict_proba(Xfp)[:, 1]
        cells = []
        for t in thrs:
            tp_n = int((stp >= t).sum()); fp_pp = (sfp >= t).sum() / N_EMPTY
            cells.append(f"{tp_n:2d}/{fp_pp:5.1f}")
            rows.append(dict(rf=tag, thr=t, TP=tp_n, TP_recall=tp_n / nTP, FP_per_panel=fp_pp))
        print(f"{tag:>10} | " + "  ".join(cells), flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "fp_tradeoff.csv", index=False)

    # operating point: hold TP within 1 of the neg5@0.5 baseline (16), min FP/panel
    base = df[(df.rf == "neg5") & (np.isclose(df.thr, 0.5))].iloc[0]
    print(f"\nbaseline neg5@0.5: TP={int(base.TP)} FP/panel={base.FP_per_panel:.1f}", flush=True)
    hold = df[df.TP >= base.TP - 1].sort_values("FP_per_panel")
    if len(hold):
        b = hold.iloc[0]
        print(f"min-FP point holding TP>={int(base.TP)-1}: rf={b.rf} thr={b.thr:.2f} "
              f"TP={int(b.TP)} FP/panel={b.FP_per_panel:.1f}  "
              f"(FP cut {100*(1-b.FP_per_panel/max(base.FP_per_panel,1e-9)):.0f}%)", flush=True)
    print("ANALYZE DONE", flush=True)


if __name__ == "__main__":
    main()
