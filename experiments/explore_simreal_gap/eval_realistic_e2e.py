"""End-to-end eval of the realistic-trained pipeline (NEW v7 + NEW RF).

Everything is recomputed with the realistic-trained v7 (the cached old-v7 features
can't be reused). Steps:
  1. features on the realistic-trained v7's VAL panels (DATA_DIFFIM_realistic) -> RF train pool
  2. features on uniform test_5sigma -> synthetic eval (no-regression guard)
  3. windowed features on the real in-region stack-missed sightings -> truth-candidate
     acceptance (does the realistic RF now keep real trails?)
Compares the realistic v7+RF against the published baseline. test_real only READ.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestClassifier

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_rf_leakage"))
sys.path.insert(0, str(REPO / "experiments/explore_simreal_gap"))
import improve_rf as ir
from ADCNN.inference.diffim_postproc_v2 import (
    RF_FEATURES_V2, compute_v2_features, apply_rf_v2, materialize_label_mask_v2, save_rf)
from probe_features import predict_window_heads   # windowed v7 heads
from ADCNN.data.diffim_dataset import diffim_mad_sigma

RUN = REPO / "experiments/diffim_runs/pilot_v7_realistic"
SCRIPTED = RUN / "ckpts/v7_realistic_scripted.pt"
RDATA = REPO / "DATA_DIFFIM_realistic"
TEST = REPO / "DATA_DIFFIM/test_5sigma"
OUT = REPO / "experiments/explore_simreal_gap"
FEATS = list(RF_FEATURES_V2)
R = 12


def train_rf(X, y):
    return RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced", n_jobs=32, random_state=0).fit(X, y)


def real_acceptance(model, rf, dev):
    """Windowed v7 -> compute_v2_features -> new RF on the 119 real in-region
    stack-missed sightings; return RF score of the truth candidate."""
    reg = pd.read_csv(OUT / "inregion_real.csv")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    H5 = REPO / "DATA_DIFFIM/test_real/test.h5"
    scores = []
    with h5py.File(H5, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        for _, r in miss.iterrows():
            idx = int(r.image_id)
            s = min(1024, fimg.shape[1], fimg.shape[2])
            h0, w0 = (fimg.shape[1]-s)//2, (fimg.shape[2]-s)//2
            sig = diffim_mad_sigma(fimg[idx, h0:h0+s, w0:w0+s].astype(np.float32))
            prob, sin, cos, agg, img, rl, cy0, cx0 = predict_window_heads(
                model, fimg, frl, idx, r.x, r.y, sig, dev)
            cand, ppd = compute_v2_features({0: prob.astype(np.float32)}, {0: img},
                {0: sin.astype(np.float32)}, {0: cos.astype(np.float32)},
                {0: agg.astype(np.float32)}, real_labels={0: rl}, verbose=False)
            if not len(cand):
                scores.append(np.nan); continue
            cand[FEATS] = cand[FEATS].replace([np.inf, -np.inf], np.nan)
            cand = apply_rf_v2(cand, rf)
            lab = materialize_label_mask_v2(cand, ppd, (1,)+prob.shape)[0]
            ty, tx = int(round(r.y-cy0)), int(round(r.x-cx0))
            y0w, y1w = max(0, ty-R), min(lab.shape[0], ty+R+1)
            x0w, x1w = max(0, tx-R), min(lab.shape[1], tx+R+1)
            cids = np.unique(lab[y0w:y1w, x0w:x1w]); cids = cids[cids > 0]
            scores.append(float(cand[cand.candidate_id.isin(cids)].score_rf.max())
                          if len(cids) else np.nan)
    return np.array(scores, float)


def main():
    dev = torch.device("cuda")
    model = torch.jit.load(str(SCRIPTED), map_location=dev).eval()
    print(f"[e2e] new v7 = {SCRIPTED}", flush=True)

    CACHE = Path("/sdf/scratch/users/m/mrakovci/e2e_cache"); CACHE.mkdir(parents=True, exist_ok=True)
    # 1. RF train pool from realistic-val panels (cached for preemption-safety)
    split = json.load(open(RUN / "split.json"))
    val = sorted(split["val_panels"])
    vcat = pd.read_csv(RDATA / "train.csv")
    remap = {o: i for i, o in enumerate(val)}
    vcat = vcat[vcat.image_id.isin(val)].copy(); vcat["image_id"] = vcat["image_id"].map(remap)
    if (CACHE / "vcand.parquet").exists():
        vcand = pd.read_parquet(CACHE / "vcand.parquet"); vlab = np.load(CACHE / "vlab.npy")
        print(f"[cache] train feats {len(vcand)}", flush=True)
    else:
        t0 = time.time()
        vcand, vlab, _, _ = ir.infer_features(model, RDATA / "train.h5", val, vcat)
        print(f"[train] {len(vcand)} cand pos={int(vlab.sum())} ({time.time()-t0:.0f}s)", flush=True)
        vcand.to_parquet(CACHE / "vcand.parquet"); np.save(CACHE / "vlab.npy", vlab)

    # 2. test_5sigma features (new v7), cached
    tcat = pd.read_csv(TEST / "test.csv")
    if (CACHE / "tcand.parquet").exists():
        tcand = pd.read_parquet(CACHE / "tcand.parquet")
        tprob = np.load(CACHE / "tprob.npy"); treal = np.load(CACHE / "treal.npy")
        print(f"[cache] test feats {len(tcand)}", flush=True)
    else:
        npan = h5py.File(TEST / "test.h5")["images"].shape[0]
        tcand, _, tprob, treal = ir.infer_features(model, TEST / "test.h5", list(range(npan)), tcat)
        tcand.to_parquet(CACHE / "tcand.parquet")
        np.save(CACHE / "tprob.npy", tprob); np.save(CACHE / "treal.npy", treal)

    # 3. train RF + eval
    X, y, _ = ir.build_pool(vcand, vlab)
    rf = train_rf(X, y)
    df, match = ir.eval_on_test(rf, tcand, tprob, treal, tcat)
    sreal = real_acceptance(model, rf, dev)
    n = np.isfinite(sreal).sum()
    print("\n================ REALISTIC v7+RF (end-to-end) ================", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"SYNTH comb_TP @ NN_FP={ir.MATCH_FP} = {match:.0f} ({match/10:.1f}% recall)", flush=True)
    print(f"REAL in-region stack-missed truth-cands: cand@truth={n}/119  "
          f"RF med={np.nanmedian(sreal):.3f}  "
          f"kept@0.5={int(np.nansum(sreal>=0.5))}  kept@0.3={int(np.nansum(sreal>=0.3))}", flush=True)
    print("\nBaseline (old uniform v7+RF): synth ~64.1%; real kept@0.5=0/46, @0.3=3.", flush=True)
    save_rf(rf, OUT / "rf_postproc_v2_realistic_e2e.pkl")
    pd.DataFrame({"score_rf": sreal}).to_csv(OUT / "real_acceptance_e2e.csv", index=False)
    print("E2E DONE", flush=True)


if __name__ == "__main__":
    main()
