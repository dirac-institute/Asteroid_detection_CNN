"""Is val_pixel_auc the right early-stop metric? Compare the BEST checkpoint
(peak val_pixel_auc, ep9) vs the LAST checkpoint (ep24, lower pixel-AUC, higher
agg_alpha) of pilot_seg_big on the TASK metric (object detection), not pixel AUC:
  - SYNTH test_5sigma: seg_model candidate-at-truth objectwise recall (model only, no RF)
  - REAL in-region stack-missed (119): seg_model fires at truth (the +8-object regime)
If LAST >= BEST on these despite lower pixel-AUC, then pixel-AUC is the WRONG
early-stop metric and the post-ep10 "overfitting" is largely a metric artifact.
Read-only on test sets.
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_simreal_gap"))
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import compute_v2_features, materialize_label_mask_v2
from ADCNN.data.diffim_dataset import diffim_mad_sigma
from probe_features import predict_window_heads
import ADCNN.evaluation.detection as evals

CK = REPO / "experiments/diffim_runs/pilot_seg_big/ckpts"
TEST5 = REPO / "DATA_DIFFIM/test_5sigma"
REAL = REPO / "DATA_DIFFIM/test_real/test.h5"
R = 12


def real_fire(model, dev):
    reg = pd.read_csv(REPO / "experiments/explore_simreal_gap/inregion_real.csv")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    pmax = []
    with h5py.File(REAL, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        sig_cache = {}
        for _, r in miss.iterrows():
            idx = int(r.image_id)
            if idx not in sig_cache:
                s = min(1024, fimg.shape[1], fimg.shape[2]); h0 = (fimg.shape[1]-s)//2; w0 = (fimg.shape[2]-s)//2
                sig_cache[idx] = diffim_mad_sigma(fimg[idx, h0:h0+s, w0:w0+s].astype(np.float32))
            prob, *_ , cy0, cx0 = predict_window_heads(model, fimg, frl, idx, r.x, r.y, sig_cache[idx], dev)
            ty, tx = int(round(r.y-cy0)), int(round(r.x-cx0))
            y0, y1 = max(0, ty-R), min(prob.shape[0], ty+R+1); x0, x1 = max(0, tx-R), min(prob.shape[1], tx+R+1)
            pmax.append(float(prob[y0:y1, x0:x1].max()))
    pmax = np.array(pmax)
    return len(pmax), (pmax >= 0.5).sum(), (pmax >= 0.3).sum(), float(np.median(pmax))


def synth_recall(model, dev):
    cat = pd.read_csv(TEST5 / "test.csv")
    with h5py.File(TEST5 / "test.h5", "r") as f:
        n = f["images"].shape[0]; probs = []
        for i in range(n):
            img = f["images"][i][:].astype(np.float32); rl = f["real_labels"][i][:]
            p, *_ = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            probs.append(p.astype(np.float32))
    probs = np.stack(probs)
    # seg_model-only objectwise recall at a fixed pixel threshold (candidate at truth)
    tp, fp, fn, _ = evals.objectwise_confusion(cat, (probs >= 0.5).astype(np.uint8), 0.5,
                                               use_threads=True, max_workers=8)
    return tp, fn, tp / max(tp + fn, 1)


def main():
    dev = torch.device("cuda")
    for tag in ["best", "last"]:
        m = torch.jit.load(str(CK / f"seg_big_{tag}_scripted.pt"), map_location=dev).eval()
        n, f5, f3, med = real_fire(m, dev)
        stp, sfn, srec = synth_recall(m, dev)
        print(f"\n===== {tag} =====", flush=True)
        print(f"  SYNTH test_5sigma seg_model-only objectwise recall: {stp}/{stp+sfn} = {100*srec:.1f}%", flush=True)
        print(f"  REAL in-region stack-missed seg_model-fire@truth: n={n}  >=0.5={f5} ({100*f5/n:.0f}%)  "
              f">=0.3={f3} ({100*f3/n:.0f}%)  med_pmax={med:.3f}", flush=True)
    print("\n(best=ep9 val_pixel_auc 0.944; last=ep24 ~0.90. If last>=best here -> pixel-AUC "
          "is the wrong early-stop metric.)", flush=True)
    print("BEST-VS-LAST DONE", flush=True)


if __name__ == "__main__":
    main()
