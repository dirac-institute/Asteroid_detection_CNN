"""Extract real-TP truth-candidate features via the SAME full-panel inference path
as probe_empty_fp.py (so size features are comparable -> no extraction-path artifact
in the separability diagnostic). Full-panel seg_model on the in-region stack-missed asteroid
panels; for each truth (x,y) keep the candidate whose materialized mask covers it.
Read-only on test_real. Sharded.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    compute_v2_features, materialize_label_mask_v2, RF_FEATURES_V2)

MODEL = REPO / "experiments/diffim_runs/pilot_seg_realistic/ckpts/seg_realistic_scripted.pt"
OUT = REPO / "DATA_DIFFIM/test_real"
RES = Path("/sdf/scratch/users/m/mrakovci/realistic/real_tp_fp")
FEATS = list(RF_FEATURES_V2)
R = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True); (RES / "parts").mkdir(exist_ok=True)
    reg = pd.read_csv(REPO / "experiments/explore_simreal_gap/inregion_real.csv")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    by_panel = {i: g for i, g in miss.groupby("image_id")}
    ids = sorted(by_panel)[args.shard::args.nshards]
    dev = torch.device("cuda"); model = torch.jit.load(str(MODEL), map_location=dev).eval()
    print(f"[real-tp] shard {args.shard}/{args.nshards}: {len(ids)} panels", flush=True)

    rows = []
    with h5py.File(OUT / "test.h5", "r") as f:
        for n, idx in enumerate(ids):
            img = f["images"][idx][:].astype(np.float32); rl = f["real_labels"][idx][:]
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            cand, ppd = compute_v2_features(
                {0: prob.astype(np.float32)}, {0: img}, {0: sin.astype(np.float32)},
                {0: cos.astype(np.float32)}, {0: agg.astype(np.float32)},
                real_labels={0: rl}, verbose=False)
            if not len(cand):
                continue
            lab = materialize_label_mask_v2(cand, ppd, (1,) + prob.shape)[0]
            for _, r in by_panel[idx].iterrows():
                ty, tx = int(round(r.y)), int(round(r.x))
                y0, y1 = max(0, ty - R), min(lab.shape[0], ty + R + 1)
                x0, x1 = max(0, tx - R), min(lab.shape[1], tx + R + 1)
                cids = np.unique(lab[y0:y1, x0:x1]); cids = cids[cids > 0]
                if not len(cids):
                    continue
                sub = cand[cand.candidate_id.isin(cids)]
                trow = sub.loc[sub.pcount.idxmax()] if "pcount" in sub else sub.iloc[0]
                rows.append({**{ff: float(trow[ff]) for ff in FEATS},
                             "ObjID": r.ObjID, "image_id": idx})
            if n % 10 == 0 or n == len(ids) - 1:
                print(f"  [{n+1}/{len(ids)}] panel={idx} cands={len(cand)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RES / "parts" / f"tp_{args.shard}.parquet")
    print(f"[done] shard {args.shard}: {len(df)} full-panel truth-cands", flush=True)


if __name__ == "__main__":
    main()
