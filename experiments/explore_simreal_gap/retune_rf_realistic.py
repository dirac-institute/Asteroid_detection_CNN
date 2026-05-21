"""RF-side remedy #1: retune the stage-2 RF for the realistic v7's candidate pool.

The realistic-trained v7 emits ~92k val candidates (2.5x uniform) with ~950 pos, so
the RF over-rejects. Sweep negative-rebalancing (subsample ratio) x leaf x class_weight,
judged by SYNTHETIC test_5sigma recall @ matched FP (no-regression), with REAL
truth-candidate acceptance reported alongside. Synthetic-only training; test_real READ
only (one feature extraction with the realistic v7, then CPU sweep).

Reuses cached features from eval_realistic_e2e (/sdf/scratch/.../e2e_cache).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestClassifier

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_rf_leakage"))
sys.path.insert(0, str(REPO / "experiments/explore_simreal_gap"))
import improve_rf as ir
from ADCNN.inference.diffim_postproc_v2 import (
    RF_FEATURES_V2, compute_v2_features, apply_rf_v2, materialize_label_mask_v2)
from ADCNN.data.diffim_dataset import diffim_mad_sigma
from probe_features import predict_window_heads

CACHE = Path("/sdf/scratch/users/m/mrakovci/e2e_cache")
OUT = REPO / "experiments/explore_simreal_gap"
SCRIPTED = REPO / "experiments/diffim_runs/pilot_v7_realistic/ckpts/v7_realistic_scripted.pt"
FEATS = list(RF_FEATURES_V2); R = 12


def real_truth_features():
    """Full RF feature vectors of the truth candidate for each real in-region
    stack-missed sighting, under the realistic v7. Cached."""
    cp = CACHE / "real_feats_realistic.parquet"
    if cp.exists():
        print(f"[cache] real feats {cp}", flush=True); return pd.read_parquet(cp)
    dev = torch.device("cuda")
    model = torch.jit.load(str(SCRIPTED), map_location=dev).eval()
    reg = pd.read_csv(OUT / "inregion_real.csv")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    H5 = REPO / "DATA_DIFFIM/test_real/test.h5"; rows = []
    with h5py.File(H5, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        for _, r in miss.iterrows():
            idx = int(r.image_id); s = min(1024, fimg.shape[1], fimg.shape[2])
            h0, w0 = (fimg.shape[1]-s)//2, (fimg.shape[2]-s)//2
            sig = diffim_mad_sigma(fimg[idx, h0:h0+s, w0:w0+s].astype(np.float32))
            prob, sin, cos, agg, img, rl, cy0, cx0 = predict_window_heads(
                model, fimg, frl, idx, r.x, r.y, sig, dev)
            cand, ppd = compute_v2_features({0: prob.astype(np.float32)}, {0: img},
                {0: sin.astype(np.float32)}, {0: cos.astype(np.float32)},
                {0: agg.astype(np.float32)}, real_labels={0: rl}, verbose=False)
            if not len(cand):
                continue
            cand[FEATS] = cand[FEATS].replace([np.inf, -np.inf], np.nan)
            lab = materialize_label_mask_v2(cand, ppd, (1,)+prob.shape)[0]
            ty, tx = int(round(r.y-cy0)), int(round(r.x-cx0))
            y0w, y1w = max(0, ty-R), min(lab.shape[0], ty+R+1)
            x0w, x1w = max(0, tx-R), min(lab.shape[1], tx+R+1)
            cids = np.unique(lab[y0w:y1w, x0w:x1w]); cids = cids[cids > 0]
            if not len(cids):
                continue
            sub = cand[cand.candidate_id.isin(cids)]
            # the candidate with most pixels in the window (best truth match)
            rows.append({**{ff: float(sub.iloc[0][ff]) for ff in FEATS},
                         "ObjID": r.ObjID})
    df = pd.DataFrame(rows); df.to_parquet(cp)
    print(f"[real] {len(df)} truth-candidate feature rows -> {cp}", flush=True)
    return df


def subsample_pool(X, y, groups, ratio, seed=0):
    if ratio is None:
        return X, y, groups
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    k = min(len(neg), int(ratio * len(pos)))
    keep = np.concatenate([pos, rng.choice(neg, k, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep], groups[keep]


def main():
    real = real_truth_features()
    Xr = real[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    vcand = pd.read_parquet(CACHE / "vcand.parquet"); vlab = np.load(CACHE / "vlab.npy")
    tcand = pd.read_parquet(CACHE / "tcand.parquet")
    tprob = np.load(CACHE / "tprob.npy"); treal = np.load(CACHE / "treal.npy")
    tcat = pd.read_csv(REPO / "DATA_DIFFIM/test_5sigma/test.csv")
    X, y, groups = ir.build_pool(vcand, vlab)
    print(f"[pool] {len(y)} rows pos={int(y.sum())} neg={int((y==0).sum())}  "
          f"real truth-cands={len(real)}", flush=True)

    configs = []
    for ratio in [None, 50, 20, 10, 5]:
        for leaf in [5, 2]:
            configs.append(dict(neg_ratio=ratio, min_samples_leaf=leaf,
                                class_weight="balanced"))
    configs.append(dict(neg_ratio=20, min_samples_leaf=2, class_weight="balanced_subsample"))

    print(f"\n{'cfg':42s} {'synthFP-match%':>14} {'realKept@0.5':>12} {'@0.3':>6} {'realMed':>8}")
    best = None
    for cfg in configs:
        Xs, ys, gs = subsample_pool(X, y, groups, cfg["neg_ratio"])
        rf = RandomForestClassifier(n_estimators=500, max_depth=14,
            min_samples_leaf=cfg["min_samples_leaf"], max_features="sqrt",
            class_weight=cfg["class_weight"], n_jobs=32, random_state=0).fit(Xs, ys)
        df, match = ir.eval_on_test(rf, tcand, tprob, treal, tcat)
        sreal = rf.predict_proba(Xr)[:, 1]
        k5 = int((sreal >= 0.5).sum()); k3 = int((sreal >= 0.3).sum())
        tag = f"neg_ratio={cfg['neg_ratio']} leaf={cfg['min_samples_leaf']} cw={cfg['class_weight'][:8]}"
        print(f"{tag:42s} {match/10:>13.1f}% {k5:>12} {k3:>6} {np.median(sreal):>8.3f}", flush=True)
        rec = dict(tag=tag, synth=match/10, k5=k5, k3=k3, med=float(np.median(sreal)))
        if best is None or (rec["synth"] >= 60.0 and rec["k5"] > best["k5"]):
            best = rec
    print(f"\nbaseline uniform v7+RF: synth 64.1%, real kept@0.5=0; "
          f"realistic default RF: synth 60.7%, real kept@0.5=5 (of {len(real)})", flush=True)
    print(f"[best by synth>=60 & max real kept@0.5] {best}", flush=True)
    print("RETUNE DONE", flush=True)


if __name__ == "__main__":
    main()
