"""Train the stage-2 RF on the REALISTIC-trail val panels (synthetic only), then
evaluate on synthetic test_5sigma (no-regression guard) and on the cached real
truth-candidates (leakage-free: test_real never trained on).

Reuses improve_rf.{infer_features,build_pool,eval_on_test}. v7 inference (GPU) on
the 50 realistic-trail val panels -> candidate features + injection labels ->
RF -> eval. Compares against the baseline RF (trained on the stock uniform-trail
val candidates, cached val_cand.parquet).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestClassifier

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_rf_leakage"))
import improve_rf as ir
from ADCNN.inference.diffim_postproc_v2 import RF_FEATURES_V2, save_rf

SC = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
RESIM = Path("/sdf/scratch/users/m/mrakovci/resim_realistic_val")
OUT = REPO / "experiments/explore_simreal_gap"
FEATS = list(RF_FEATURES_V2)
MODEL = REPO / "experiments/diffim_runs/pilot_v7/ckpts/v7_scripted.pt"


def train_rf(X, y):
    return RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=32, random_state=0).fit(X, y)


def score_real(rf):
    real = pd.read_csv(OUT / "probe_real_features.csv")
    Xr = real[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    return rf.predict_proba(Xr)[:, 1], len(real)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={dev}", flush=True)
    # --- features from the realistic-trail val panels ---
    rcache = RESIM / "realistic_cand.parquet"
    if rcache.exists():
        rcand = pd.read_parquet(rcache); rlab = np.load(RESIM / "realistic_labels.npy")
        print(f"[cache] realistic feats {len(rcand)}", flush=True)
    else:
        model = torch.jit.load(str(MODEL), map_location=dev).eval()
        cat = pd.read_csv(RESIM / "train.csv")
        npan = int(cat.image_id.max()) + 1
        panel_idx = list(range(npan))
        t0 = time.time()
        rcand, rlab, _, _ = ir.infer_features(model, RESIM / "train.h5", panel_idx, cat)
        print(f"[realistic] {len(rcand)} cand pos={int(rlab.sum())} ({time.time()-t0:.0f}s)", flush=True)
        keep = list(dict.fromkeys(FEATS + ["panel_id", "candidate_id",
                    "effective_t_low", "frac_real_label_overlap"]))
        rcand[keep].to_parquet(rcache); np.save(RESIM / "realistic_labels.npy", rlab)
        del model; torch.cuda.empty_cache()

    # --- synthetic test eval inputs (cached) ---
    tcat = pd.read_csv(REPO / "DATA_DIFFIM/test_5sigma/test.csv")
    tcand = pd.read_parquet(SC / "test_cand.parquet")
    tprob = np.load(SC / "test_probs.npy"); treal = np.load(SC / "test_real.npy")

    # --- baseline (stock uniform-trail val candidates) ---
    bvc = pd.read_parquet(SC / "val_cand.parquet"); bvl = np.load(SC / "val_labels.npy")

    for tag, (cand, lab) in {"baseline_uniform": (bvc, bvl),
                             "realistic": (rcand, rlab)}.items():
        X, y, groups = ir.build_pool(cand, lab)
        rf = train_rf(X, y)
        df, match = ir.eval_on_test(rf, tcand, tprob, treal, tcat)
        sreal, nreal = score_real(rf)
        print(f"\n===== {tag}: pool={len(y)} pos={int(y.sum())} =====", flush=True)
        print(df.to_string(index=False), flush=True)
        print(f"  SYNTH comb_TP@NN_FP={ir.MATCH_FP} = {match:.0f} ({match/10:.1f}%)", flush=True)
        print(f"  REAL truth-cands n={nreal}: RF med={np.median(sreal):.3f} "
              f"kept@0.5={int((sreal>=0.5).sum())} ({100*(sreal>=0.5).mean():.0f}%) "
              f"kept@0.3={int((sreal>=0.3).sum())} ({100*(sreal>=0.3).mean():.0f}%)", flush=True)
        if tag == "realistic":
            save_rf(rf, OUT / "rf_postproc_v2_realistic.pkl")
            print(f"  saved -> {OUT/'rf_postproc_v2_realistic.pkl'}", flush=True)
    print("\nRETRAIN-REALISTIC DONE", flush=True)


if __name__ == "__main__":
    main()
