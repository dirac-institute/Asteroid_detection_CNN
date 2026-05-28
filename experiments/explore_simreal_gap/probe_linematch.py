"""Does the line-overlap matcher (objectwise_confusion) explain anything beyond
the RF? Compare three ways of counting seg_model-alone recovery (NO RF) on the 119
real in-region stack-missed sightings:

  A. point-probe: max seg_model prob within +/-12px of catalog (x,y)  [already have]
  B. line-match : run the ACTUAL objectwise_confusion (draws trail from
     x,beta,trail_length, thickness=half_psf, hit = any seg_model component on the line)
     on the seg_model-only mask (prob>=thr), same matcher used for sim scoring.

If B >> A, my point-probe undercounted and the line matcher is fine/generous.
If B << A, the drawn line (beta/length/anchor) is mis-placed for real trails and
the matcher itself is dropping true seg_model hits. Either way this isolates matching
from the RF (which we already showed kills the candidates).
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
import ADCNN.evaluation.detection as evals

OUT = REPO / "experiments/explore_simreal_gap"
MODEL = REPO / "experiments/diffim_runs/pilot_seg/ckpts/segmentation_scripted.pt"
REAL_H5 = REPO / "DATA_DIFFIM/test_real/test.h5"
R = 12


def predict_window(model, fimg, frl, idx, x, y, sigma, device, *,
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

    pacc = np.zeros((h, w), np.float32); wacc = np.zeros((h, w), np.float32)
    xs, locs = [], []
    for yy in starts(h):
        for xx in starts(w):
            xs.append(build_3channel(img[yy:yy + tile, xx:xx + tile],
                                     rl[yy:yy + tile, xx:xx + tile],
                                     panel_sigma=sigma, clip=clip))
            locs.append((yy, xx))
    xb = torch.from_numpy(np.stack(xs)).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        seg = model(xb)[0]
    P = torch.sigmoid(seg).detach().float().cpu().numpy()[:, 0]
    for i, (yy, xx) in enumerate(locs):
        pacc[yy:yy + tile, xx:xx + tile] += P[i] * hann
        wacc[yy:yy + tile, xx:xx + tile] += hann
    return pacc / np.maximum(wacc, 1e-6), y0, x0


def main():
    reg = pd.read_csv(OUT / "inregion_real.csv")
    # need beta for the line matcher; pull from forced-phot table
    fz = pd.read_csv(REPO / "experiments/explore_rf_leakage/test_real_clean/"
                     "per_sighting_forced_lsst.csv")
    keys = ["ObjID", "image_id", "visit", "detector"]
    reg = reg.merge(fz[keys + ["beta"]], on=keys, how="left")
    miss = reg[~reg.stack_detected.astype(bool)].reset_index(drop=True)
    print(f"[linematch] {len(miss)} in-region stack-missed; beta non-null="
          f"{miss.beta.notna().sum()}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(MODEL), map_location=dev).eval()
    THRS = [0.5, 0.3, 0.1]
    rows = []
    with h5py.File(REAL_H5, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        for n, r in miss.iterrows():
            idx = int(r.image_id)
            s = min(1024, fimg.shape[1], fimg.shape[2])
            h0, w0 = (fimg.shape[1] - s) // 2, (fimg.shape[2] - s) // 2
            sig = diffim_mad_sigma(fimg[idx, h0:h0 + s, w0:w0 + s].astype(np.float32))
            prob, cy0, cx0 = predict_window(model, fimg, frl, idx, r.x, r.y, sig, dev)
            xw, yw = r.x - cx0, r.y - cy0
            # point-probe
            yy0, yy1 = max(0, int(yw) - R), int(yw) + R + 1
            xx0, xx1 = max(0, int(xw) - R), int(xw) + R + 1
            pmax = float(prob[yy0:yy1, xx0:xx1].max())
            rec = dict(ObjID=r.ObjID, image_id=idx, trail_length=r.trail_length,
                       snr=r.lsst_psf_snr, pmax_pt=pmax)
            # line-match via the real matcher, seg_model-only (no RF), per threshold
            cat1 = pd.DataFrame([{ "image_id": 0, "x": xw, "y": yw,
                                   "beta": r.beta, "trail_length": r.trail_length }])
            for thr in THRS:
                _, _, _, cm = evals.objectwise_confusion(
                    cat1, prob[None], thr, use_threads=True, max_workers=2)
                rec[f"line_thr{thr}"] = bool(cm["nn_detected"].iloc[0])
                rec[f"pt_thr{thr}"] = pmax >= thr
            rows.append(rec)
            if n % 25 == 0 or n == len(miss) - 1:
                print(f"  [{n+1}/{len(miss)}] idx={idx} pmax={pmax:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_linematch.csv", index=False)

    N = len(df)
    print(f"\n=== seg_model-ALONE recovery (NO RF), real in-region stack-missed n={N} ===")
    print("thr   point(±12px)   line-match(objectwise)   ")
    for thr in THRS:
        pt = int(df[f"pt_thr{thr}"].sum()); ln = int(df[f"line_thr{thr}"].sum())
        print(f"  {thr:.1f}   {pt:3d} ({100*pt/N:4.1f}%)      {ln:3d} ({100*ln/N:4.1f}%)")
    print("\n(pipeline w/ synthetic RF @0.5 scored 2/119=1.7%; synth stack-missed "
          "seg_model-fire 72%)")
    print("LINEMATCH DONE", flush=True)


if __name__ == "__main__":
    main()
