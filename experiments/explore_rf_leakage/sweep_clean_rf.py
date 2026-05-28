"""Threshold sweep on test_5sigma for the leaky vs clean RF, so we can compare
recall at MATCHED false-positive rate (thr=0.50 was tuned for the leaky RF and
is not a fair operating point for the clean one).

For each RF and threshold: materialize the kept-candidate mask, then objectwise
NN confusion (with stack_fp ignored) + combined (stack OR NN) confusion.
Prints a per-RF sweep table and the clean-RF recall interpolated at the leaky
RF's NN_FP operating point. Also persists the scored candidate tables to scratch.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))

from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    compute_v2_features, apply_rf_v2, materialize_label_mask_v2, load_rf)
import ADCNN.evaluation.detection as evals

DATA   = REPO / "DATA_DIFFIM"
CK     = REPO / "experiments/diffim_runs/pilot_seg/ckpts"
MODEL  = CK / "segmentation_scripted.pt"
OLD_RF = CK / "rf_postproc_v2.pkl"
NEW_RF = Path("/sdf/scratch/users/m/mrakovci/rf_leakage/rf_postproc_v2_valtrain.pkl")
OUT    = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def main():
    model = torch.jit.load(str(MODEL), map_location=DEVICE); model.eval()
    with h5py.File(DATA / "test_5sigma" / "test.h5", "r") as f:
        N = int(f["images"].shape[0])
        probs, sins, coss, aggs, diffims, reals, gts = [], [], [], [], [], [], []
        for i in range(N):
            img = f["images"][i][:]; rl = f["real_labels"][i][:].astype(np.uint16)
            p, s, c, a = predict_panel_overlap_3ch_full(model, img, rl, device=DEVICE)
            probs.append(p.astype(np.float32)); sins.append(s); coss.append(c); aggs.append(a)
            diffims.append(img.astype(np.float32)); reals.append(rl)
            gts.append(f["masks"][i][:].astype(np.uint8))
    stk = lambda L: np.stack(L, 0)
    prob, sin, cos, agg = stk(probs), stk(sins), stk(coss), stk(aggs)
    diffim, real, gt = stk(diffims), stk(reals), stk(gts)
    del model; import gc; gc.collect(); torch.cuda.empty_cache()
    cat = pd.read_csv(DATA / "test_5sigma" / "test.csv")

    cand, ppd = compute_v2_features(prob, diffim, sin, cos, agg, real_labels=real, verbose=True)
    print(f"[test] candidates={len(cand)}", flush=True)

    def sweep(rf, name):
        scored = apply_rf_v2(cand, rf)
        scored[["panel_id", "candidate_id", "effective_t_low", "score_rf"]].to_parquet(
            OUT / f"scored_{name}.parquet")
        rows = []
        for thr in THRS:
            kept = scored[scored["score_rf"] >= thr]
            masks = materialize_label_mask_v2(kept, ppd, prob.shape)
            (ntp, nfp, nfn), _, _ = evals.full_confusion(
                catalog=cat, ground_truth=gt, predictions=masks,
                threshold=0.5, stack_fp=real, verbose=False)
            (ctp, cfp, cfn), _ = evals.combined_confusion_separate(
                catalog=cat, ground_truth=gt, predictions=masks,
                threshold=0.5, stack_mask=real, verbose=False)
            rows.append(dict(thr=thr, n_kept=len(kept), nn_tp=ntp, nn_fp=nfp,
                             comb_tp=ctp, comb_fp=cfp))
            print(f"  [{name}] thr={thr:.2f} kept={len(kept):5d} "
                  f"NN_TP={ntp:4d} NN_FP={nfp:4d} comb_TP={ctp:4d} comb_FP={cfp:5d}",
                  flush=True)
        return pd.DataFrame(rows)

    print("\n=== OLD leaky RF sweep ===", flush=True)
    old = sweep(load_rf(OLD_RF), "old")
    print("\n=== NEW clean RF sweep ===", flush=True)
    new = sweep(load_rf(NEW_RF), "new")
    old.to_csv(OUT / "sweep_old.csv", index=False)
    new.to_csv(OUT / "sweep_new.csv", index=False)

    # Clean recall at the leaky operating point (thr=0.50): match on NN_FP.
    leaky = old[old.thr == 0.50].iloc[0]
    target_fp = int(leaky.nn_fp)
    # interpolate clean comb_TP vs nn_fp (nn_fp increases as thr drops)
    ns = new.sort_values("nn_fp")
    clean_tp_at_match = float(np.interp(target_fp, ns.nn_fp, ns.comb_tp))
    clean_nntp_at_match = float(np.interp(target_fp, ns.nn_fp, ns.nn_tp))
    print("\n================= MATCHED-FP COMPARISON =================")
    print(f"Leaky @ thr0.50: NN_FP={target_fp}  comb_TP={int(leaky.comb_tp)} "
          f"({leaky.comb_tp/10:.1f}% recall)")
    print(f"Clean @ NN_FP={target_fp} (interp): comb_TP={clean_tp_at_match:.0f} "
          f"({clean_tp_at_match/10:.1f}% recall)  NN_TP={clean_nntp_at_match:.0f}")
    print("\nDONE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
