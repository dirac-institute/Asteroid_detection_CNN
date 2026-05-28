"""Extract the FULL RF feature vector of the seg_model candidate at each real
in-region stack-missed truth, so we can compare real trail-candidate features to
the SYNTHETIC positives the RF was trained on, and find which features push real
trails into the RF's 'background' region. Read-only on test_real.
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.data.diffim_dataset import build_3channel, diffim_mad_sigma
from ADCNN.inference.diffim_eval import hann2d
from ADCNN.inference.diffim_postproc_v2 import (
    compute_v2_features, apply_rf_v2, materialize_label_mask_v2,
    load_rf, RF_FEATURES_V2)

OUT = REPO / "experiments/explore_simreal_gap"
MODEL = REPO / "experiments/diffim_runs/pilot_seg/ckpts/segmentation_scripted.pt"
RF_PKL = REPO / "experiments/explore_rf_leakage/rf_postproc_v2_valtrain.pkl"
REAL_H5 = REPO / "DATA_DIFFIM/test_real/test.h5"
R = 12


def predict_window_heads(model, fimg, frl, idx, x, y, sigma, device, *,
                         half=192, tile=128, stride=64, clip=5.0):
    H, W = fimg.shape[1], fimg.shape[2]
    cy, cx = int(round(y)), int(round(x))
    y0 = max(0, min(cy - half, H - tile)); y1 = min(H, max(cy + half, y0 + tile))
    x0 = max(0, min(cx - half, W - tile)); x1 = min(W, max(cx + half, x0 + tile))
    img = fimg[idx, y0:y1, x0:x1].astype(np.float32)
    rl = frl[idx, y0:y1, x0:x1]
    h, w = img.shape
    hann = hann2d(tile)

    def starts(N):
        out = list(range(0, max(N - tile, 0) + 1, stride))
        if not out or out[-1] != N - tile:
            out.append(max(N - tile, 0))
        return sorted(set(out))

    acc = {k: np.zeros((h, w), np.float32) for k in "pscg"}
    wacc = np.zeros((h, w), np.float32)
    xs, locs = [], []
    for yy in starts(h):
        for xx in starts(w):
            xs.append(build_3channel(img[yy:yy + tile, xx:xx + tile],
                                     rl[yy:yy + tile, xx:xx + tile],
                                     panel_sigma=sigma, clip=clip))
            locs.append((yy, xx))
    xb = torch.from_numpy(np.stack(xs)).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        seg, sn, cs, _, ag = model(xb)
    P = torch.sigmoid(seg).detach().float().cpu().numpy()[:, 0]
    S = sn.detach().float().cpu().numpy()[:, 0]
    C = cs.detach().float().cpu().numpy()[:, 0]
    G = ag.detach().float().cpu().numpy()[:, 0]
    for i, (yy, xx) in enumerate(locs):
        sl = (slice(yy, yy + tile), slice(xx, xx + tile))
        acc["p"][sl] += P[i] * hann; acc["s"][sl] += S[i] * hann
        acc["c"][sl] += C[i] * hann; acc["g"][sl] += G[i] * hann
        wacc[sl] += hann
    wm = np.maximum(wacc, 1e-6)
    return (acc["p"] / wm, acc["s"] / wm, acc["c"] / wm, acc["g"] / wm,
            img, rl, y0, x0)


def main():
    reg = pd.read_csv(OUT / "inregion_real.csv")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(MODEL), map_location=dev).eval()
    rf = load_rf(str(RF_PKL))
    feats = list(RF_FEATURES_V2)

    rows = []
    with h5py.File(REAL_H5, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        for n, r in miss.iterrows():
            idx = int(r.image_id)
            s = min(1024, fimg.shape[1], fimg.shape[2])
            h0, w0 = (fimg.shape[1] - s) // 2, (fimg.shape[2] - s) // 2
            sig = diffim_mad_sigma(fimg[idx, h0:h0 + s, w0:w0 + s].astype(np.float32))
            prob, sin, cos, agg, img, rl, cy0, cx0 = predict_window_heads(
                model, fimg, frl, idx, r.x, r.y, sig, dev)
            cand, ppd = compute_v2_features(
                {0: prob.astype(np.float32)}, {0: img}, {0: sin.astype(np.float32)},
                {0: cos.astype(np.float32)}, {0: agg.astype(np.float32)},
                real_labels={0: rl}, verbose=False)
            if not len(cand):
                continue
            cand[feats] = cand[feats].replace([np.inf, -np.inf], np.nan)
            cand = apply_rf_v2(cand, rf)
            lab = materialize_label_mask_v2(cand, ppd, (1,) + prob.shape)[0]
            ty, tx = int(round(r.y - cy0)), int(round(r.x - cx0))
            y0w, y1w = max(0, ty - R), min(lab.shape[0], ty + R + 1)
            x0w, x1w = max(0, tx - R), min(lab.shape[1], tx + R + 1)
            cids = np.unique(lab[y0w:y1w, x0w:x1w]); cids = cids[cids > 0]
            if not len(cids):
                continue
            sub = cand[cand.candidate_id.isin(cids)]
            trow = sub.loc[sub.score_rf.idxmax()]   # the truth candidate
            rec = {f: float(trow[f]) for f in feats}
            rec.update(ObjID=r.ObjID, image_id=idx, score_rf=float(trow.score_rf),
                       trail_length=r.trail_length, snr=r.lsst_psf_snr)
            rows.append(rec)
            if n % 25 == 0 or n == len(miss) - 1:
                print(f"  [{n+1}/{len(miss)}] saved {len(rows)} truth-cands", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_real_features.csv", index=False)
    print(f"[done] wrote {len(df)} real truth-candidate feature rows", flush=True)
    print("FEATURES DONE", flush=True)


if __name__ == "__main__":
    main()
