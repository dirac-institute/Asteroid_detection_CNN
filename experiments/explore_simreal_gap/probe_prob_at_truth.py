"""Diagnostic for the sim-to-real gain gap.

Decisive question: on faint trails the LSST 5sigma stack MISSES, does seg_model even
produce probability mass at the asteroid's true position? We compare the
distribution of `max seg_model prob within +/-R px of the true (x,y)` between:

  - REAL  in-region sightings (SNR 2-10, trail 6-60), split stack-detected vs missed
  - SYNTH in-region injections (SNR_estimation 2-10, trail 6-60), same split

The stack-detected vs stack-missed split is measurement-independent, so it
sidesteps the SNR-axis comparability problem (synthetic detection-scale SNR vs
real point-source PSF-fit lsst_psf_snr).

REAL probs are recomputed with seg_model (GPU). SYNTH probs come from the cached
test_probs.npy (no GPU needed) produced by improve_rf.py.
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

SC = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
OUT = REPO / "experiments/explore_simreal_gap"
MODEL = REPO / "experiments/diffim_runs/pilot_seg/ckpts/segmentation_scripted.pt"
REAL_H5 = REPO / "DATA_DIFFIM/test_real/test.h5"
R = 12  # half-window (px) around true position for max-prob extraction


def windowed_stats(prob, x, y, r=R):
    H, W = prob.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(W, xi + r + 1)
    y0, y1 = max(0, yi - r), min(H, yi + r + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan, np.nan, np.nan
    win = prob[y0:y1, x0:x1]
    return float(win.max()), float(win.mean()), float((win >= 0.5).sum())


def panel_sigma(dset, idx, stats_crop=1024):
    """MAD-sigma from the central crop of the FULL panel — identical to
    predict_panel_overlap_3ch_full so normalization matches the full scoring."""
    H, W = dset.shape[1], dset.shape[2]
    s = min(stats_crop, H, W)
    h0, w0 = (H - s) // 2, (W - s) // 2
    return diffim_mad_sigma(dset[idx, h0:h0 + s, w0:w0 + s].astype(np.float32))


def predict_window(model, fimg, frl, idx, x, y, sigma, device, *,
                   half=192, tile=128, stride=64, clip=5.0):
    """Run seg_model ONLY on tiles covering a (2*half) box around (x,y), Hann-blended.
    Reads just the crop from h5 (not the 65 MB panel). Returns (prob_crop, y0, x0)
    in panel coords. Normalization sigma is passed in (panel-level), so the
    output matches full-panel inference inside the window."""
    H, W = fimg.shape[1], fimg.shape[2]
    cy, cx = int(round(y)), int(round(x))
    # crop big enough that whole tiles fit and the truth window is interior
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

    prob_acc = np.zeros((h, w), np.float32)
    wacc = np.zeros((h, w), np.float32)
    xs, locs = [], []
    for yy in starts(h):
        for xx in starts(w):
            x3 = build_3channel(img[yy:yy + tile, xx:xx + tile],
                                rl[yy:yy + tile, xx:xx + tile],
                                panel_sigma=sigma, clip=clip)
            xs.append(x3); locs.append((yy, xx))
    xb = torch.from_numpy(np.stack(xs)).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        seg = model(xb)[0]
    probs = torch.sigmoid(seg).detach().float().cpu().numpy()[:, 0]
    for (yy, xx), p in zip(locs, probs):
        prob_acc[yy:yy + tile, xx:xx + tile] += p * hann
        wacc[yy:yy + tile, xx:xx + tile] += hann
    return prob_acc / np.maximum(wacc, 1e-6), y0, x0


def do_real():
    reg = pd.read_csv(OUT / "inregion_real.csv")
    # all 119 stack-missed + up to 120 stack-detected controls
    miss = reg[~reg.stack_detected.astype(bool)]
    det = reg[reg.stack_detected.astype(bool)].sample(
        min(120, int(reg.stack_detected.sum())), random_state=0)
    sub = pd.concat([miss, det]).reset_index(drop=True)
    print(f"[real] {len(sub)} sightings ({len(miss)} missed + {len(det)} det)",
          flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(MODEL), map_location=dev).eval()
    rows = []
    import time as _t
    t0 = _t.time()
    with h5py.File(REAL_H5, "r") as f:
        fimg, frl = f["images"], f["real_labels"]
        for n, (_, r) in enumerate(sub.iterrows()):
            idx = int(r.image_id)
            sig = panel_sigma(fimg, idx)
            prob_c, cy0, cx0 = predict_window(
                model, fimg, frl, idx, r.x, r.y, sig, dev)
            mx, mn, n50 = windowed_stats(prob_c, r.x - cx0, r.y - cy0)
            rows.append(dict(src="real", ObjID=r.ObjID, image_id=idx,
                             trail_length=r.trail_length, snr=r.lsst_psf_snr,
                             stack=bool(r.stack_detected), nn=bool(r.nn_detected),
                             pmax=mx, pmean=mn, n_over50=n50,
                             crop_pmax=float(prob_c.max())))
            if n % 25 == 0 or n == len(sub) - 1:
                el = _t.time() - t0
                print(f"  [real {n+1}/{len(sub)}] idx={idx} pmax={mx:.3f} "
                      f"({el:.0f}s, {el/(n+1):.2f}s/sighting)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_real.csv", index=False)
    print(f"[real] wrote {OUT/'probe_real.csv'}", flush=True)
    return df


def do_synth():
    tp = np.load(SC / "test_probs.npy")  # (50, H, W)
    syn = pd.read_csv(REPO / "DATA_DIFFIM/test_5sigma/test.csv")
    reg = syn[(syn.SNR_estimation >= 2) & (syn.SNR_estimation <= 10)
              & (syn.trail_length >= 6) & (syn.trail_length <= 60)].copy()
    print(f"[synth] {len(reg)} in-region injections "
          f"(stack-missed {int((~reg.stack_detection.astype(bool)).sum())})",
          flush=True)
    rows = []
    for _, r in reg.iterrows():
        iid = int(r.image_id)
        if iid >= tp.shape[0]:
            continue
        mx, mn, n50 = windowed_stats(tp[iid], r.x, r.y)
        rows.append(dict(src="synth", image_id=iid, trail_length=r.trail_length,
                         snr=r.SNR_estimation, stack=bool(r.stack_detection),
                         pmax=mx, pmean=mn, n_over50=n50))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_synth.csv", index=False)
    print(f"[synth] wrote {OUT/'probe_synth.csv'}", flush=True)
    return df


def summarize(real, synth):
    def block(df, name):
        out = [f"\n=== {name} (max seg_model prob within +/-{R}px of truth) ==="]
        for tag, g in [("stack-MISSED", df[~df.stack]),
                       ("stack-detected", df[df.stack])]:
            if not len(g):
                continue
            p = g.pmax.dropna()
            out.append(
                f"  {tag:15s} n={len(g):4d}  "
                f"pmax: med={p.median():.3f} mean={p.mean():.3f}  "
                f"frac(pmax>=0.5)={100*(p>=0.5).mean():.1f}%  "
                f"frac(pmax>=0.1)={100*(p>=0.1).mean():.1f}%")
        return "\n".join(out)
    rep = block(synth, "SYNTHETIC in-region") + "\n" + block(real, "REAL in-region")
    print(rep, flush=True)
    (OUT / "probe_summary.txt").write_text(rep + "\n")


def main():
    sp = OUT / "probe_synth.csv"
    synth = pd.read_csv(sp) if sp.exists() else do_synth()
    real = do_real()
    summarize(real, synth)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
