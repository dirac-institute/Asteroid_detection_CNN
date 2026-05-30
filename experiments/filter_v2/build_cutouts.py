"""Build fresh cutouts + per-candidate catalog rows on the seg_v2 model.

The cached cut_train2 (experiments/heliolinc/rejecter_data/cut_train2/) was built from the May-26
deployed seg model + the OLD train2 panels (pre dataset rebuild). The May-29 rebuild produced a
NEW train2.h5 (467 panels), val.h5 (155), val2.h5 (112), and we ship the new seg_v2 segmentation
model -- so existing cutouts no longer match. This script extracts fresh cutouts AND the candidate
catalog columns we need to compute the val2 combined-FP metric without re-running the segmentation
model every iteration.

Outputs under experiments/filter_v2/cutouts/<set>/:
  part_XXXX.npz with arrays
    X       (N, 3, 48, 48) float32   the cutout stack [diffim/sigma, seg_prob, seg_agg]
    y       (N,)            int8     1=trail (injection overlap), 0=FP
    panel   (N,)            int32    panel id within the set's h5 (==csv image_id; 0..N-1)
    cand    (N, 8)          float32  catalog columns aligned with X
                                     [x_centroid, y_centroid, mf_beta, mf_length, mf_flux, mf_snr,
                                      area, max_p]   <- everything the downstream catalog needs
  done.txt resumable panel list.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CAT_COLS = ["x_centroid", "y_centroid", "mf_beta", "mf_length",
            "mf_flux", "mf_snr", "area", "max_p"]


def panel_cand(model, img, rl, panel_cat, *, pid, fp_cap, device, k):
    """Run seg_v2 on one panel + extract candidates + cutouts (at side `k`) + catalog columns."""
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import compute_v2_features, label_candidates_by_injection_overlap
    from ADCNN.inference.cnn_postproc import make_cutouts
    prob, sn, cs, agg = predict_panel_overlap_3ch_full(model, img, rl, device=device)
    prob = prob.astype(np.float32); agg = np.asarray(agg, np.float32)
    cand, _ = compute_v2_features({pid: prob}, {pid: img}, {pid: sn}, {pid: cs}, {pid: agg},
                                  real_labels={pid: rl.astype(np.uint16)}, verbose=False)
    if not len(cand):
        return (np.zeros((0, 3, k, k), np.float32),
                np.zeros((0,), np.int8), np.zeros((0, len(CAT_COLS)), np.float32))
    lab = label_candidates_by_injection_overlap(cand, panel_cat, {pid: prob})
    keep = np.ones(len(cand), bool)
    if fp_cap > 0:
        fp_i = np.where(lab == 0)[0]
        if len(fp_i) > fp_cap:
            drop = np.random.default_rng(pid).choice(fp_i, len(fp_i) - fp_cap, replace=False)
            keep[drop] = False
    cand = cand[keep].reset_index(drop=True); lab = lab[keep]
    X = make_cutouts(cand, img, prob, agg, k=k)
    feats = cand[CAT_COLS].to_numpy(np.float32)
    return X, lab.astype(np.int8), feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(REPO / "models/seg_v2_segmentation_scripted.pt"))
    ap.add_argument("--h5", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fp-cap", type=int, default=600, help="0 = keep ALL FP candidates")
    ap.add_argument("--k", type=int, default=96, help="cutout side length (px); 96 gives every iter room to center-crop down to 48 or 64")
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = torch.jit.load(a.seg, map_location=dev).eval()
    truth = pd.read_csv(a.csv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    done_file = out / "done.txt"
    done = set(int(x) for x in done_file.read_text().split()) if done_file.exists() else set()
    buf = {k: [] for k in ("X", "y", "panel", "cand")}
    chunk_pids: list[int] = []

    def flush():
        if not buf["X"]:
            return
        fn = out / f"part_{chunk_pids[0]:04d}.npz"
        np.savez_compressed(fn,
                             X=np.concatenate(buf["X"]).astype(np.float32),
                             y=np.concatenate(buf["y"]).astype(np.int8),
                             panel=np.array(buf["panel"], np.int32),
                             cand=np.concatenate(buf["cand"]).astype(np.float32))
        done.update(chunk_pids); done_file.write_text(" ".join(map(str, sorted(done))))
        for k in buf:
            buf[k].clear()
        print(f"  [flush] {fn.name}  total {len(done)} panels done", flush=True)
        chunk_pids.clear()

    with h5py.File(a.h5, "r") as f:
        npan = int(f["images"].shape[0])
        for pid in range(npan):
            if pid in done:
                continue
            img = f["images"][pid][:].astype(np.float32)
            rl = f["real_labels"][pid][:].astype(np.uint16)
            X, y, cand = panel_cand(model, img, rl, truth, pid=pid, fp_cap=a.fp_cap, device=dev, k=a.k)
            chunk_pids.append(pid)
            if len(X):
                buf["X"].append(X); buf["y"].append(y); buf["cand"].append(cand)
                buf["panel"].extend([pid] * len(X))
                print(f"  panel {pid}: {len(X)} cand  TP={int((y == 1).sum())}", flush=True)
            else:
                print(f"  panel {pid}: 0 candidates", flush=True)
            if len(chunk_pids) >= a.chunk:
                flush()
    flush()
    n_parts = len(list(out.glob("part_*.npz")))
    print(f"CUTOUTS DONE: {len(done)} panels -> {out}/ ({n_parts} parts)", flush=True)


if __name__ == "__main__":
    main()
