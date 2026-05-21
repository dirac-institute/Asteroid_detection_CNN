"""Leak-free stage-2 RF: train on the 50 held-out val panels, eval on test_5sigma.

The shipped rf_postproc_v2.pkl was trained on test_5sigma — the SAME set it is
evaluated on (see postproc_iter/train_rf_v2.py). That is evaluation leakage.

This script removes it with a textbook train/val/test split:
  * v7 trained on the 750 train panels of train.h5 (split.json:train_panels)
  * stage-2 RF trained HERE on the 50 val panels (split.json:val_panels) that
    v7 only ever saw for early-stopping — same 5-sigma build, (visit,detector)-
    disjoint from test_5sigma, 1000 injections (matches test_5sigma).
  * evaluate on test_5sigma (untouched by either model).

Outputs (to scratch, NOT promoted over the shipped RF):
  - rf_postproc_v2_valtrain.pkl
  - prints: train-set relabel stats + panel-disjoint 5-fold CV AUC,
    and the test_5sigma headline (combined / NN objectwise TP/FP at thr 0.5)
    for BOTH the old leaky RF and the new clean RF, side by side.
"""
from __future__ import annotations
import json, sys, time
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
    RF_FEATURES_V2, DEFAULT_THR, compute_v2_features, apply_rf_v2,
    materialize_label_mask_v2, label_candidates_by_injection_overlap,
    train_rf_v2, build_rf_postproc_v2, load_rf, save_rf,
)
import ADCNN.evaluation.detection as evals

DATA   = REPO / "DATA_DIFFIM"
CK     = REPO / "experiments/diffim_runs/pilot_v7/ckpts"
MODEL  = CK / "v7_scripted.pt"
OLD_RF = CK / "rf_postproc_v2.pkl"                       # leaky (trained on test)
SPLIT  = REPO / "experiments/diffim_runs/pilot_v7/split.json"
OUT    = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
NEW_RF = OUT / "rf_postproc_v2_valtrain.pkl"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THR = DEFAULT_THR


def load_model():
    m = torch.jit.load(str(MODEL), map_location=DEVICE)
    m.eval()
    return m


def infer_panels(model, h5_path, panel_idx):
    """Run v7 full inference over the given panel indices of an h5.

    Returns (prob, sin, cos, agg, diffims, real_labels, gt) each (n,H,W),
    panels in the order given by `panel_idx`.
    """
    probs, sins, coss, aggs, diffims, reals, gts = [], [], [], [], [], [], []
    with h5py.File(h5_path, "r") as f:
        has_gt = "masks" in f
        for orig in panel_idx:
            img = f["images"][orig][:]
            rl  = f["real_labels"][orig][:].astype(np.uint16)
            p, s, c, a = predict_panel_overlap_3ch_full(model, img, rl, device=DEVICE)
            probs.append(p.astype(np.float32)); sins.append(s); coss.append(c); aggs.append(a)
            diffims.append(img.astype(np.float32)); reals.append(rl)
            if has_gt:
                gts.append(f["masks"][orig][:].astype(np.uint8))
    stk = lambda L: np.stack(L, axis=0)
    gt = stk(gts) if gts else None
    return (stk(probs), stk(sins), stk(coss), stk(aggs),
            stk(diffims), stk(reals), gt)


def objectwise(catalog, gt, masks, stack_fp, tag):
    """Print + return (NN tp/fp/fn, combined tp/fp/fn) exactly as the notebook."""
    print(f"\n----- test_5sigma objectwise @ thr={THR}  [{tag}] -----")
    (ntp, nfp, nfn), _, _ = evals.full_confusion(
        catalog=catalog, ground_truth=gt, predictions=masks,
        threshold=THR, stack_fp=stack_fp, verbose=True)
    (ctp, cfp, cfn), _ = evals.combined_confusion_separate(
        catalog=catalog, ground_truth=gt, predictions=masks,
        threshold=THR, stack_mask=stack_fp, verbose=True)
    return dict(nn_tp=ntp, nn_fp=nfp, nn_fn=nfn, comb_tp=ctp, comb_fp=cfp, comb_fn=cfn)


def main():
    print(f"DEVICE={DEVICE}  thr={THR}", flush=True)
    val_panels = sorted(json.load(open(SPLIT))["val_panels"])
    assert len(val_panels) == 50, val_panels
    remap = {orig: i for i, orig in enumerate(val_panels)}

    model = load_model()

    # ===================== PART 1: train clean RF on val panels =====================
    print(f"\n========== TRAIN RF on {len(val_panels)} val panels (leak-free) ==========",
          flush=True)
    t0 = time.time()
    vp, vs, vc, va, vdiff, vreal, _ = infer_panels(model, DATA / "train.h5", val_panels)
    print(f"[val] inference {vp.shape} in {time.time()-t0:.0f}s", flush=True)

    vcat = pd.read_csv(DATA / "train.csv")
    vcat = vcat[vcat.image_id.isin(val_panels)].copy()
    vcat["image_id"] = vcat["image_id"].map(remap)   # 0..49 to match array order
    print(f"[val] {len(vcat)} injections over {vcat.image_id.nunique()} panels", flush=True)

    # Features + objectwise-overlap labels (same matching as eval metric).
    vcand, _ = compute_v2_features(vp, vdiff, vs, vc, va, real_labels=vreal, verbose=True)
    vlabels = label_candidates_by_injection_overlap(vcand, vcat, vp)
    print(f"[val] candidates={len(vcand)}  pos={int(vlabels.sum())} "
          f"neg={int((vlabels==0).sum())}", flush=True)

    # Panel-disjoint 5-fold CV on the training pool (honest in-sample reference).
    fp_mask = ((vlabels == 0) & (vcand["frac_real_label_overlap"].to_numpy() < 0.5))
    pool_mask = (vlabels == 1) | fp_mask
    pool = vcand[pool_mask]
    y = vlabels[pool_mask]
    X = pool[list(RF_FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
    groups = pool["panel_id"].to_numpy()
    print(f"[val] training pool={len(pool)}  pos={int(y.sum())} neg={int((y==0).sum())}",
          flush=True)
    print("\n=== val-panel panel-disjoint 5-fold CV ===", flush=True)
    aucs = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        clf = RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=32, random_state=0)
        clf.fit(X[tr], y[tr])
        auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        aucs.append(auc)
        print(f"  fold {fold}: AUC={auc:.4f}  ({len(np.unique(groups[tr]))} train panels)",
              flush=True)
    print(f"  mean CV AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}", flush=True)

    # Final clean RF on the full val pool.
    clean_rf = train_rf_v2(vcand, labels=vlabels)
    save_rf(clean_rf, NEW_RF)
    print(f"[val] saved clean RF -> {NEW_RF}", flush=True)
    del vp, vs, vc, va, vdiff, vreal, vcand
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ===================== PART 2: eval test_5sigma, old vs new RF =====================
    print("\n========== EVAL on test_5sigma (old leaky RF vs new clean RF) ==========",
          flush=True)
    test_panels = list(range(h5_n(DATA / "test_5sigma" / "test.h5")))
    tp_, ts_, tc_, ta_, tdiff, treal, tgt = infer_panels(
        model, DATA / "test_5sigma" / "test.h5", test_panels)
    del model; gc.collect(); torch.cuda.empty_cache()
    tcat = pd.read_csv(DATA / "test_5sigma" / "test.csv")

    tcand, ppd = compute_v2_features(tp_, tdiff, ts_, tc_, ta_, real_labels=treal, verbose=True)
    print(f"[test] candidates={len(tcand)}", flush=True)

    results = {}
    for tag, rf in (("OLD leaky (trained on test_5sigma)", load_rf(OLD_RF)),
                    ("NEW clean (trained on val panels)", clean_rf)):
        scored = apply_rf_v2(tcand, rf)
        kept = scored[scored["score_rf"] >= THR]
        masks = materialize_label_mask_v2(kept, ppd, tp_.shape)
        print(f"\n[{tag}] kept {len(kept)} of {len(scored)} candidates @ thr={THR}", flush=True)
        results[tag] = objectwise(tcat, tgt, masks, treal, tag)

    print("\n================= SUMMARY (test_5sigma, thr=%.2f) =================" % THR)
    hdr = f"{'RF':<40}{'NN_TP':>7}{'NN_FP':>7}{'comb_TP':>9}{'comb_FP':>9}"
    print(hdr); print("-" * len(hdr))
    for tag, r in results.items():
        print(f"{tag:<40}{r['nn_tp']:>7}{r['nn_fp']:>7}{r['comb_tp']:>9}{r['comb_fp']:>9}")
    print("\nDONE", time.strftime("%Y-%m-%dT%H:%M:%S"))


def h5_n(p):
    with h5py.File(p, "r") as f:
        return int(f["images"].shape[0])


if __name__ == "__main__":
    main()
