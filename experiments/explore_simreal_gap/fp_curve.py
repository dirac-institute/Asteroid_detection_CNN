"""v7-level TP-vs-FP operating curve on test_5sigma: sweep prob threshold and report
objectwise TP-recall AND FP count for each model. The model whose curve dominates
(>= recall at < FP) is better for 'more TP, fewer FP'. Args: pairs of ckpt:label.
"""
import sys
from pathlib import Path
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/explore_simreal_gap"))
import h5py, numpy as np, pandas as pd, torch
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
import ADCNN.evaluation.detection as evals

TEST5 = REPO / "DATA_DIFFIM/test_5sigma"
THRS = [0.3, 0.5, 0.7, 0.9]

def probs_for(ckpt, dev):
    m = torch.jit.load(ckpt, map_location=dev).eval()
    cat = pd.read_csv(TEST5 / "test.csv")
    with h5py.File(TEST5 / "test.h5", "r") as f:
        n = f["images"].shape[0]; out = []
        for i in range(n):
            img = f["images"][i][:].astype(np.float32); rl = f["real_labels"][i][:]
            p, *_ = predict_panel_overlap_3ch_full(m, img, rl, device=dev)
            out.append(p.astype(np.float32))
    return cat, np.stack(out)

def main():
    dev = torch.device("cuda")
    for arg in sys.argv[1:]:
        ckpt, label = arg.split("::")
        cat, probs = probs_for(ckpt, dev)
        print(f"\n===== {label} : test_5sigma TP/FP operating curve =====", flush=True)
        print(f"  {'thr':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'recall%':>8}", flush=True)
        for thr in THRS:
            tp, fp, fn, _ = evals.objectwise_confusion(cat, (probs >= thr).astype(np.uint8), thr,
                                                       use_threads=True, max_workers=8)
            print(f"  {thr:>5} {tp:>5} {fp:>5} {fn:>5} {100*tp/max(tp+fn,1):>7.1f}", flush=True)
    print("FP CURVE DONE", flush=True)

if __name__ == "__main__":
    main()
