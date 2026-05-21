"""Leakage-free RF remedy: make the stage-2 RF robust to real trail morphology by
augmenting SYNTHETIC positives in feature space toward lower trail-coherence
(partial / non-uniform trails), the regime real trails occupy. Train on synthetic
ONLY, preserve synthetic test performance, then read real ONCE.

Augmentation model: a real/partial trail's COHERENCE features (matched-filter
SNR/flux, oriented-aggregator MEANS, integrated logit, elongation, area, length)
lie partway between a clean synthetic trail and background. So for each positive we
make N_AUG copies with
    feat_aug = neg_median + f * (feat_pos - neg_median),   f ~ U(F_MIN, 1.0)
on the coherence features only; PEAK features (max_p, top5_mean_p, or_agg_max) and
local-pixel stats are left intact (a bright peak survives non-uniformity). f and the
feature set are fixed a priori from physical reasoning + SYNTHETIC negative medians
-- no real-data numbers used. Committed design, evaluated on real exactly once.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_rf_leakage"))
from ADCNN.inference.diffim_postproc_v2 import RF_FEATURES_V2, save_rf
import improve_rf as ir

SC = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
OUT = REPO / "experiments/explore_simreal_gap"
FEATS = list(RF_FEATURES_V2)
F_MIN, N_AUG, SEED = 0.35, 3, 0

# coherence / integrated-strength / shape features a partial trail degrades.
# (everything EXCEPT the peak-prob and local-pixel-stat features.)
KEEP_INTACT = {
    "max_p", "mean_p", "top5_mean_p", "or_agg_max", "or_r", "p_med_mf_snr",
    "frac_real_label_overlap", "meanang_30", "meanang_50", "meanang_80",
    "loc_med_z", "loc_std_z", "loc_max_z", "loc_min_z", "loc_skew", "loc_pos_frac",
    "loc_neg_frac", "loc_sum_z", "loc_dipole", "loc_npos", "loc_nneg",
}
COH = [f for f in FEATS if f not in KEEP_INTACT]


def augment(X, y, neg_med, rng):
    """Return augmented (X, y) = originals + N_AUG degraded copies of each positive."""
    coh_idx = np.array([FEATS.index(f) for f in COH])
    pos = X[y == 1]
    base = neg_med[coh_idx]
    aug_blocks = [X]
    aug_y = [y]
    for _ in range(N_AUG):
        cp = pos.copy()
        f = rng.uniform(F_MIN, 1.0, size=(len(pos), 1))
        jit = np.exp(rng.normal(0, 0.05, size=(len(pos), len(coh_idx))))  # mild
        cp[:, coh_idx] = base + (f * jit) * (cp[:, coh_idx] - base)
        aug_blocks.append(cp)
        aug_y.append(np.ones(len(pos), dtype=y.dtype))
    return np.vstack(aug_blocks), np.concatenate(aug_y)


def train_rf(X, y):
    return RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=32, random_state=0).fit(X, y)


def score_real(rf):
    real = pd.read_csv(OUT / "probe_real_features.csv")
    Xr = real[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    s = rf.predict_proba(Xr)[:, 1]
    return s, real


def main():
    rng = np.random.default_rng(SEED)
    vc = pd.read_parquet(SC / "val_cand.parquet"); vl = np.load(SC / "val_labels.npy")
    X, y, groups = ir.build_pool(vc, vl)
    neg_med = np.nanmedian(X[y == 0], axis=0)
    print(f"[pool] {len(y)} rows pos={int(y.sum())}; degrading {len(COH)} coherence "
          f"feats, keeping {len(KEEP_INTACT)} intact", flush=True)

    tcat = pd.read_csv(REPO / "DATA_DIFFIM/test_5sigma/test.csv")
    tcand = pd.read_parquet(SC / "test_cand.parquet")
    tprob = np.load(SC / "test_probs.npy"); treal = np.load(SC / "test_real.npy")

    results = {}
    for tag, (Xt, yt) in {
        "baseline": (X, y),
        "augmented": augment(X, y, neg_med, rng),
    }.items():
        rf = train_rf(Xt, yt)
        df, match = ir.eval_on_test(rf, tcand, tprob, treal, tcat)
        sreal, real = score_real(rf)
        nreal = len(real)
        kept05 = int((sreal >= 0.5).sum()); kept03 = int((sreal >= 0.3).sum())
        results[tag] = dict(rf=rf, synth=df, match=match,
                            real_med=float(np.median(sreal)),
                            real_k05=kept05, real_k03=kept03, n=nreal)
        print(f"\n===== {tag} (train rows={len(yt)}, pos={int(yt.sum())}) =====", flush=True)
        print("  SYNTHETIC test_5sigma:", flush=True)
        print(df.to_string(index=False), flush=True)
        print(f"  comb_TP @ NN_FP={ir.MATCH_FP} = {match:.0f} ({match/10:.1f}% recall)", flush=True)
        print(f"  REAL truth-cands (n={nreal}): RF score med={np.median(sreal):.3f}  "
              f"kept@0.5={kept05} ({100*kept05/nreal:.0f}%)  "
              f"kept@0.3={kept03} ({100*kept03/nreal:.0f}%)", flush=True)

    b, a = results["baseline"], results["augmented"]
    print("\n================ SUMMARY ================", flush=True)
    print(f"SYNTH comb_TP@matchFP : baseline {b['match']:.0f} -> augmented {a['match']:.0f} "
          f"(/1000 = {b['match']/10:.1f}% -> {a['match']/10:.1f}%)", flush=True)
    print(f"REAL truth-cand kept@0.5: baseline {b['real_k05']}/{b['n']} -> "
          f"augmented {a['real_k05']}/{a['n']}", flush=True)
    print(f"REAL truth-cand RF med  : baseline {b['real_med']:.3f} -> augmented {a['real_med']:.3f}", flush=True)
    save_rf(a["rf"], OUT / "rf_postproc_v2_cohaug.pkl")
    print(f"\nsaved augmented RF -> {OUT/'rf_postproc_v2_cohaug.pkl'}", flush=True)
    print("AUGMENT-RF DONE", flush=True)


if __name__ == "__main__":
    main()
