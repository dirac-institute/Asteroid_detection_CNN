"""Extract ALL candidate feature vectors on the real EMPTY diffim panels (the FP
source) under the realistic v7, so the TP-vs-FP operating curve can be built
offline for ANY RF / threshold without more GPU. Pairs with the real TP truth-cand
features (real_feats_realistic.parquet) to answer: can we hold the real TPs while
cutting the 68 FP/panel? Read-only on test_real. Sharded.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import compute_v2_features, RF_FEATURES_V2

MODEL = REPO / "experiments/diffim_runs/pilot_v7_realistic/ckpts/v7_realistic_scripted.pt"
OUT = REPO / "DATA_DIFFIM/test_real"
RES = Path("/sdf/scratch/users/m/mrakovci/realistic/empty_fp")
FEATS = list(RF_FEATURES_V2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "parts").mkdir(exist_ok=True)

    dev = torch.device("cuda")
    model = torch.jit.load(str(MODEL), map_location=dev).eval()
    panels = pd.read_csv(OUT / "panels.csv")
    empty = sorted(int(i) for i in panels[panels.role == "empty"].image_id.unique())
    mine = empty[args.shard::args.nshards]
    print(f"[empty-fp] shard {args.shard}/{args.nshards}: {len(mine)} of {len(empty)} empty panels", flush=True)

    rows = []
    with h5py.File(OUT / "test.h5", "r") as f:
        for n, idx in enumerate(mine):
            img = f["images"][idx][:].astype(np.float32)
            rl = f["real_labels"][idx][:]
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            cand, _ = compute_v2_features(
                {0: prob.astype(np.float32)}, {0: img}, {0: sin.astype(np.float32)},
                {0: cos.astype(np.float32)}, {0: agg.astype(np.float32)},
                real_labels={0: rl}, verbose=False)
            if len(cand):
                sub = cand[FEATS].replace([np.inf, -np.inf], np.nan)
                sub = sub.assign(image_id=idx)
                rows.append(sub)
            if n % 10 == 0 or n == len(mine) - 1:
                print(f"  [{n+1}/{len(mine)}] panel={idx} cands={len(cand)}", flush=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FEATS + ["image_id"])
    df.to_parquet(RES / "parts" / f"empty_{args.shard}.parquet")
    print(f"[done] shard {args.shard}: {len(df)} empty-panel candidates", flush=True)


if __name__ == "__main__":
    main()
