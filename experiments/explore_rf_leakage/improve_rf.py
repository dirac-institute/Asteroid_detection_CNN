"""Leak-free attempt to improve the stage-2 RF via hyperparameters / training
choices. Model selection is done by panel-disjoint 5-fold CV on the VAL set
ONLY; the winner is evaluated on test_5sigma exactly once. Selecting a config by
its test recall would be test-set leakage, so we never do that.

Caches val/test candidate feature tables + test prob maps to scratch so further
CPU-only hyperparameter sweeps need no GPU.
"""
from __future__ import annotations
import json, sys, time, itertools
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))

from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    RF_FEATURES_V2, compute_v2_features, materialize_label_mask_v2,
    label_candidates_by_injection_overlap)
import ADCNN.evaluation.detection as evals

DATA  = REPO / "DATA_DIFFIM"
CK    = REPO / "experiments/diffim_runs/pilot_v7/ckpts"
MODEL = CK / "v7_scripted.pt"
SPLIT = REPO / "experiments/diffim_runs/pilot_v7/split.json"
OUT   = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dedupe: frac_real_label_overlap is already in RF_FEATURES_V2
KEEP = list(dict.fromkeys(list(RF_FEATURES_V2) +
            ["panel_id", "candidate_id", "effective_t_low", "frac_real_label_overlap"]))
MATCH_FP = 443   # leaky RF's NN_FP operating point (apples-to-apples target)


def infer_features(model, h5_path, panel_idx, catalog):
    probs, sins, coss, aggs, diffims, reals = [], [], [], [], [], []
    with h5py.File(h5_path, "r") as f:
        for orig in panel_idx:
            img = f["images"][orig][:]; rl = f["real_labels"][orig][:].astype(np.uint16)
            p, s, c, a = predict_panel_overlap_3ch_full(model, img, rl, device=DEVICE)
            probs.append(p.astype(np.float32)); sins.append(s); coss.append(c); aggs.append(a)
            diffims.append(img.astype(np.float32)); reals.append(rl)
    stk = lambda L: np.stack(L, 0)
    prob, real = stk(probs), stk(reals)
    cand, _ = compute_v2_features(prob, stk(diffims), stk(sins), stk(coss), stk(aggs),
                                  real_labels=real, verbose=False)
    labels = label_candidates_by_injection_overlap(cand, catalog, prob)
    return cand, labels, prob, real


def build_pool(cand, labels):
    fp = (labels == 0) & (cand["frac_real_label_overlap"].to_numpy() < 0.5)
    m = (labels == 1) | fp
    X = cand.loc[m, list(RF_FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
    return X, labels[m], cand.loc[m, "panel_id"].to_numpy()


def cv_auc(cfg, X, y, groups):
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=32,
                                     random_state=0, **cfg)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def eval_on_test(rf, tcand, tprob, treal, tcat):
    ppd = {i: tprob[i] for i in range(tprob.shape[0])}
    X = tcand[list(RF_FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
    score = rf.predict_proba(X)[:, 1]
    rows = []
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        keep = tcand[score >= thr]
        masks = materialize_label_mask_v2(keep, ppd, tprob.shape)
        ntp, nfp, nfn, _ = evals.objectwise_confusion(tcat, masks, 0.5, stack_fp=treal)
        (ctp, cfp, cfn), _ = evals.combined_confusion_separate(
            tcat, None, masks, 0.5, stack_mask=treal, verbose=False)
        rows.append(dict(thr=thr, nn_tp=ntp, nn_fp=nfp, comb_tp=ctp, comb_fp=cfp))
    df = pd.DataFrame(rows).sort_values("nn_fp")
    comb_at_match = float(np.interp(MATCH_FP, df.nn_fp, df.comb_tp))
    return df, comb_at_match


def main():
    print(f"DEVICE={DEVICE}", flush=True)
    tcat = pd.read_csv(DATA / "test_5sigma" / "test.csv")
    cache = {f: OUT / f for f in ["val_cand.parquet", "val_labels.npy",
             "test_cand.parquet", "test_probs.npy", "test_real.npy"]}

    if all(p.exists() for p in cache.values()):
        print("[cache] loading features from scratch (skip GPU inference)", flush=True)
        vcand = pd.read_parquet(cache["val_cand.parquet"])
        vlabels = np.load(cache["val_labels.npy"])
        tcand = pd.read_parquet(cache["test_cand.parquet"])
        tprob = np.load(cache["test_probs.npy"]); treal = np.load(cache["test_real.npy"])
    else:
        val_panels = sorted(json.load(open(SPLIT))["val_panels"])
        remap = {o: i for i, o in enumerate(val_panels)}
        model = torch.jit.load(str(MODEL), map_location=DEVICE); model.eval()
        vcat = pd.read_csv(DATA / "train.csv")
        vcat = vcat[vcat.image_id.isin(val_panels)].copy(); vcat["image_id"] = vcat["image_id"].map(remap)
        t0 = time.time()
        vcand, vlabels, _, _ = infer_features(model, DATA / "train.h5", val_panels, vcat)
        print(f"[val] {len(vcand)} cand pos={int(vlabels.sum())} ({time.time()-t0:.0f}s)", flush=True)
        vcand[KEEP].to_parquet(cache["val_cand.parquet"]); np.save(cache["val_labels.npy"], vlabels)
        tpanels = list(range(_h5n(DATA / "test_5sigma" / "test.h5")))
        tcand, _, tprob, treal = infer_features(model, DATA / "test_5sigma" / "test.h5", tpanels, tcat)
        del model; import gc; gc.collect(); torch.cuda.empty_cache()
        tcand[KEEP].to_parquet(cache["test_cand.parquet"])
        np.save(cache["test_probs.npy"], tprob); np.save(cache["test_real.npy"], treal)
    print(f"[data] val={len(vcand)} test={len(tcand)} cand", flush=True)

    X, y, groups = build_pool(vcand, vlabels)
    print(f"[pool] {len(y)} rows pos={int(y.sum())}", flush=True)

    # ---- config search (selection by VAL CV only) ----
    base = dict(n_estimators=500, max_depth=14, min_samples_leaf=5, max_features="sqrt")
    configs = {
        "baseline(prod)": base,
        "depth25":        {**base, "max_depth": 25},
        "depthNone":      {**base, "max_depth": None},
        "leaf2":          {**base, "min_samples_leaf": 2},
        "leaf1":          {**base, "min_samples_leaf": 1},
        "mf0.3":          {**base, "max_features": 0.3},
        "mf0.5":          {**base, "max_features": 0.5},
        "n1200":          {**base, "n_estimators": 1200},
        "deep_leaf1_mf03":{**base, "max_depth": 30, "min_samples_leaf": 1, "max_features": 0.3},
        "tuned_combo":    {"n_estimators": 1200, "max_depth": 25, "min_samples_leaf": 2,
                           "max_features": 0.3},
    }
    print("\n=== VAL panel-disjoint 5-fold CV (model selection) ===", flush=True)
    cv = {}
    for name, cfg in configs.items():
        m, s = cv_auc(cfg, X, y, groups)
        cv[name] = m
        print(f"  {name:18s} CV_AUC={m:.4f} +/- {s:.4f}  {cfg}", flush=True)
    best = max(cv, key=cv.get)
    print(f"\n[select] best by VAL CV = {best} (AUC={cv[best]:.4f})", flush=True)

    # ---- evaluate baseline + selected on test ONCE ----
    print("\n=== test_5sigma eval (matched NN_FP=%d) ===" % MATCH_FP, flush=True)
    for tag in dict.fromkeys(["baseline(prod)", best]):
        rf = RandomForestClassifier(class_weight="balanced", n_jobs=32, random_state=0,
                                    **configs[tag]).fit(X, y)
        df, match = eval_on_test(rf, tcand, tprob, treal, tcat)
        print(f"\n[{tag}] {configs[tag]}", flush=True)
        print(df.to_string(index=False), flush=True)
        print(f"  --> comb_TP @ NN_FP={MATCH_FP} = {match:.0f}  ({match/10:.1f}% recall)", flush=True)
        if tag == best:
            from ADCNN.inference.diffim_postproc_v2 import save_rf
            save_rf(rf, OUT / "rf_postproc_v2_valtrain_tuned.pkl")
            print(f"  saved tuned RF -> {OUT/'rf_postproc_v2_valtrain_tuned.pkl'}", flush=True)
    print("\nReference: leaky=85.8%, clean-baseline matched=64.1%", flush=True)
    print("DONE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)


def _h5n(p):
    with h5py.File(p, "r") as f:
        return int(f["images"].shape[0])


if __name__ == "__main__":
    main()
