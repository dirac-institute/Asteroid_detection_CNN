"""End-to-end v7+RF eval for the NEW reg2 model (lambda_orient0+dropout+wd+intensity-aug).
Recomputes ALL feature sets with reg2 (old caches are old-v7), retrains the neg5 RF on
reg2's VAL candidates (shard_3 holdout, leakage-safe), and reports synth + real TP/FP
vs the deployed v7-realistic+neg5. Saves the reg2 RF. GPU. Read-only on test sets.
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
from ADCNN.inference.diffim_postproc_v2 import (RF_FEATURES_V2, compute_v2_features,
                                                apply_rf_v2, materialize_label_mask_v2, save_rf)
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from probe_features import predict_window_heads
from ADCNN.data.diffim_dataset import diffim_mad_sigma

REG2 = REPO / "experiments/diffim_runs/pilot_v7_reg2/ckpts/v7_reg2_best_scripted.pt"
SHARD3 = REPO / "DATA_DIFFIM_realistic/shard_3/train.h5"
VALCSV = REPO / "DATA_DIFFIM_realistic/shard_3_val.csv"
TEST5 = REPO / "DATA_DIFFIM/test_5sigma"
REAL = REPO / "DATA_DIFFIM/test_real/test.h5"
PANELS = REPO / "DATA_DIFFIM/test_real/panels.csv"
OUT = REPO / "experiments/explore_simreal_gap"
CACHE = Path("/sdf/scratch/users/m/mrakovci/reg2_cache"); CACHE.mkdir(parents=True, exist_ok=True)
FEATS = list(RF_FEATURES_V2); R = 12; N_EMPTY = 150


def rf_neg5(X, y, ratio=5, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, min(len(neg), ratio * len(pos)), replace=False)])
    return RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced", n_jobs=32, random_state=0).fit(X[keep], y[keep])


def real_truth_feats(model, dev):
    """Windowed reg2 -> features of the truth candidate for each in-region stack-missed."""
    reg = pd.read_csv(OUT / "inregion_real.csv")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    rows = []
    with h5py.File(REAL, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        for _, r in miss.iterrows():
            idx = int(r.image_id)
            s = min(1024, fimg.shape[1], fimg.shape[2]); h0, w0 = (fimg.shape[1]-s)//2, (fimg.shape[2]-s)//2
            sig = diffim_mad_sigma(fimg[idx, h0:h0+s, w0:w0+s].astype(np.float32))
            prob, sin, cos, agg, img, rl, cy0, cx0 = predict_window_heads(model, fimg, frl, idx, r.x, r.y, sig, dev)
            cand, ppd = compute_v2_features({0: prob.astype(np.float32)}, {0: img}, {0: sin.astype(np.float32)},
                {0: cos.astype(np.float32)}, {0: agg.astype(np.float32)}, real_labels={0: rl}, verbose=False)
            if not len(cand):
                continue
            lab = materialize_label_mask_v2(cand, ppd, (1,)+prob.shape)[0]
            ty, tx = int(round(r.y-cy0)), int(round(r.x-cx0))
            y0, y1 = max(0, ty-R), min(lab.shape[0], ty+R+1); x0, x1 = max(0, tx-R), min(lab.shape[1], tx+R+1)
            cids = np.unique(lab[y0:y1, x0:x1]); cids = cids[cids > 0]
            if len(cids):
                rows.append(cand[cand.candidate_id.isin(cids)].iloc[[0]][FEATS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FEATS), len(miss)


def empty_fp_feats(model, dev):
    """reg2 on the 150 empty real diffim panels -> all candidate features (FP pool)."""
    empty = sorted(int(i) for i in pd.read_csv(PANELS).query("role=='empty'").image_id.unique())
    rows = []
    with h5py.File(REAL, "r") as f:
        for n, idx in enumerate(empty):
            img = f["images"][idx][:]; rl = f["real_labels"][idx][:].astype(np.uint16)
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            cand, _ = compute_v2_features(prob[None], img[None].astype(np.float32), sin[None], cos[None],
                                          agg[None], real_labels=rl[None], verbose=False)
            if len(cand):
                rows.append(cand[FEATS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FEATS), len(empty)


def main():
    dev = torch.device("cuda")
    model = torch.jit.load(str(REG2), map_location=dev).eval()
    print(f"[reg2-e2e] model={REG2}", flush=True)

    # 1. RF train pool: reg2 on shard_3 val (leakage-safe; held out from reg2 training)
    vcat = pd.read_csv(VALCSV); val = sorted(vcat.image_id.unique())
    remap = {o: i for i, o in enumerate(val)}; vcat = vcat.copy(); vcat["image_id"] = vcat["image_id"].map(remap)
    if (CACHE / "vcand.parquet").exists():
        vcand = pd.read_parquet(CACHE / "vcand.parquet"); vlab = np.load(CACHE / "vlab.npy")
    else:
        t0 = time.time(); vcand, vlab, _, _ = ir.infer_features(model, SHARD3, val, vcat)
        vcand.to_parquet(CACHE / "vcand.parquet"); np.save(CACHE / "vlab.npy", vlab)
        print(f"[val] {len(vcand)} cand pos={int(vlab.sum())} ({time.time()-t0:.0f}s)", flush=True)

    # 2. test_5sigma features
    tcat = pd.read_csv(TEST5 / "test.csv")
    if (CACHE / "tcand.parquet").exists():
        tcand = pd.read_parquet(CACHE / "tcand.parquet"); tprob = np.load(CACHE / "tprob.npy"); treal = np.load(CACHE / "treal.npy")
    else:
        npan = h5py.File(TEST5 / "test.h5")["images"].shape[0]
        tcand, _, tprob, treal = ir.infer_features(model, TEST5 / "test.h5", list(range(npan)), tcat)
        tcand.to_parquet(CACHE / "tcand.parquet"); np.save(CACHE / "tprob.npy", tprob); np.save(CACHE / "treal.npy", treal)
        print(f"[test5] {len(tcand)} cand", flush=True)

    # 3. real TP truth-cand features ; 4. empty-panel FP features
    if (CACHE / "real_tp.parquet").exists():
        Xtp_df = pd.read_parquet(CACHE / "real_tp.parquet"); n_miss = int(np.load(CACHE / "n_miss.npy"))
    else:
        Xtp_df, n_miss = real_truth_feats(model, dev)
        Xtp_df.to_parquet(CACHE / "real_tp.parquet"); np.save(CACHE / "n_miss.npy", n_miss)
        print(f"[real TP] truth-cand found {len(Xtp_df)}/{n_miss}", flush=True)
    if (CACHE / "empty_fp.parquet").exists():
        Xfp_df = pd.read_parquet(CACHE / "empty_fp.parquet")
    else:
        Xfp_df, nempty = empty_fp_feats(model, dev)
        Xfp_df.to_parquet(CACHE / "empty_fp.parquet")
        print(f"[empty FP] {len(Xfp_df)} cand on {nempty} empty panels", flush=True)

    # 5. train neg5 RF on reg2 val candidates
    X, y, _ = ir.build_pool(vcand, vlab)
    rf = rf_neg5(X, y, 5)
    save_rf(rf, OUT / "rf_postproc_v2_reg2_neg5.pkl")

    # 6. synth eval (matched FP operating point)
    df, match = ir.eval_on_test(rf, tcand, tprob, treal, tcat)
    # 7. real TP / FP at thresholds
    Xtp = Xtp_df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    Xfp = Xfp_df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    stp = rf.predict_proba(Xtp)[:, 1]; sfp = rf.predict_proba(Xfp)[:, 1]

    print("\n================ reg2 v7+RF (neg5) END-TO-END ================", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"SYNTH comb_TP @ NN_FP={ir.MATCH_FP} = {match:.0f} ({match/10:.1f}% recall)", flush=True)
    print(f"\nREAL (in-region stack-missed truth-cands found {len(Xtp)}/{n_miss}):", flush=True)
    for t in [0.3, 0.5, 0.7]:
        print(f"  thr={t}: real TP kept={int((stp>=t).sum())}/{len(Xtp)}  "
              f"empty FP/panel={(sfp>=t).sum()/N_EMPTY:.1f}", flush=True)
    print("\nDEPLOYED baseline (v7-realistic+neg5, from notebooks): synth obj TP=682/1000 FP=2107; "
          "real per-sighting NN TP=705 FP=10386 (69.2/panel)", flush=True)
    print("REG2 E2E DONE", flush=True)


if __name__ == "__main__":
    main()
