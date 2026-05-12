"""Production-style CPU inference for the diffim NN.

API:
    pp = predict_panel_cpu(panel_image, real_labels, model_path, ...)
returns a (H, W) float16 probability map; no model class definition is
required because the model is loaded via torch.jit.load().

Pipeline:
  1. Build the 3-channel input (signed diffim MAD-normalized, log1p local
     std, real_labels binary).
  2. Tile the panel at stride = tile (no overlap; the v5/v6 random-crop
     training already made the model translation-invariant, so the
     accuracy delta vs overlapping inference is minimal).
  3. Batch tiles through the TorchScript module.
  4. Stitch the per-tile probabilities back into a panel map.

If overlap is requested (`stride < tile`), a Hann-weighted average is used
to combine overlapping predictions — slower but matches the v5/v6 eval
exactly.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ----- channel construction (duplicated from ADCNN.data.diffim_dataset to
# avoid importing the training-only dataset class at inference time) -----
def diffim_mad_sigma(arr: np.ndarray) -> float:
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(good)) + 1e-8)


def local_std_tile(z_tile: np.ndarray, window: int = 11) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    mu = uniform_filter(z_tile, size=window, mode="nearest")
    sq = uniform_filter(z_tile * z_tile, size=window, mode="nearest")
    return np.sqrt(np.clip(sq - mu * mu, 0.0, None)).astype(np.float32)


def normalize_local_std(local_std: np.ndarray) -> np.ndarray:
    return np.clip(np.log1p(local_std) / np.log1p(5.0), 0.0, 1.0).astype(np.float32)


def build_3channel_panel(
    diffim_panel: np.ndarray,
    real_labels_panel: np.ndarray,
    *,
    sigma: Optional[float] = None,
    stats_crop: int = 1024,
    clip: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Returns (3, H, W) float32 + the panel sigma used."""
    H, W = diffim_panel.shape
    if sigma is None:
        s = min(stats_crop, H, W)
        h0 = (H - s) // 2; w0 = (W - s) // 2
        sigma = diffim_mad_sigma(diffim_panel[h0:h0 + s, w0:w0 + s])
    z = np.clip(diffim_panel.astype(np.float32) / sigma, -clip, clip)
    lstd = local_std_tile(z, window=11)
    lstd_n = normalize_local_std(lstd)
    rl = (real_labels_panel > 0).astype(np.float32)
    return np.stack([z.astype(np.float32), lstd_n, rl], axis=0), float(sigma)


# ----- inference -----
def _hann2d(tile: int) -> np.ndarray:
    w = np.hanning(tile + 2)[1:-1]
    return (w[:, None] * w[None, :]).astype(np.float32)


def _tile_starts(N: int, t: int, stride: int) -> list[int]:
    out = list(range(0, max(N - t, 0) + 1, stride))
    if not out or out[-1] != N - t:
        out.append(max(N - t, 0))
    return sorted(set(out))


@torch.no_grad()
def predict_panel_cpu(
    diffim_panel: np.ndarray,
    real_labels_panel: np.ndarray,
    model_path: str | Path,
    *,
    tile: int = 128,
    stride: Optional[int] = None,
    batch: int = 16,
    n_threads: Optional[int] = None,
    sigma: Optional[float] = None,
    use_hann: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run the diffim NN on a single (H, W) diffim + real_labels pair.

    Returns (probability_map (H, W) float16, timing_dict).
    `stride=None` defaults to `tile` (no overlap).
    """
    if n_threads is not None:
        torch.set_num_threads(int(n_threads))

    timing = {}
    t0 = time.time()
    chans, used_sigma = build_3channel_panel(diffim_panel, real_labels_panel, sigma=sigma)
    timing["channels_s"] = time.time() - t0

    t1 = time.time()
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    timing["load_model_s"] = time.time() - t1

    H, W = diffim_panel.shape
    s = int(stride or tile)
    ys = _tile_starts(H, tile, s)
    xs = _tile_starts(W, tile, s)

    prob_acc = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32) if use_hann else None
    hann = _hann2d(tile) if use_hann else None

    chans_t = torch.from_numpy(chans)  # (3, H, W)

    t2 = time.time()
    batch_tiles, batch_locs = [], []

    def flush():
        if not batch_tiles:
            return
        xb = torch.stack(batch_tiles, dim=0)  # (B, 3, T, T)
        seg_logits = model(xb)[0]
        probs = torch.sigmoid(seg_logits).numpy().astype(np.float32)
        for (y0, x0), p in zip(batch_locs, probs[:, 0]):
            if use_hann:
                prob_acc[y0:y0 + tile, x0:x0 + tile] += p * hann
                weight_acc[y0:y0 + tile, x0:x0 + tile] += hann
            else:
                prob_acc[y0:y0 + tile, x0:x0 + tile] = p
        batch_tiles.clear(); batch_locs.clear()

    for y0 in ys:
        for x0 in xs:
            batch_tiles.append(chans_t[:, y0:y0 + tile, x0:x0 + tile])
            batch_locs.append((y0, x0))
            if len(batch_tiles) >= batch:
                flush()
    flush()
    timing["forward_s"] = time.time() - t2

    if use_hann:
        out = prob_acc / np.maximum(weight_acc, 1e-6)
    else:
        out = prob_acc

    timing["panel_total_s"] = time.time() - t0
    timing["n_tiles"] = len(ys) * len(xs)
    timing["used_sigma"] = used_sigma
    return out.astype(np.float16), timing


# ----- CLI for benchmark -----
def _benchmark(args):
    import h5py
    from ADCNN.inference.diffim_candidates import (
        extract_candidates, CandidateExtractorConfig,
    )

    with h5py.File(args.h5, "r") as f:
        n = int(f["images"].shape[0])
        n = min(n, args.n_panels) if args.n_panels else n
        pids = list(range(n))
        diffims = [f["images"][p][:] for p in pids]
        rls = [f["real_labels"][p][:] for p in pids]

    per_panel = []
    for p, di, rl in zip(pids, diffims, rls):
        pp, t = predict_panel_cpu(
            di, rl, args.model,
            tile=args.tile, stride=args.stride,
            batch=args.batch, n_threads=args.n_threads,
            use_hann=args.use_hann,
        )
        # Optionally extract candidates (real production step).
        if args.extract_candidates:
            t_c = time.time()
            cand = extract_candidates(
                pp.astype(np.float32), real_labels=rl,
                cfg=CandidateExtractorConfig(t_low=0.05, min_area=4), panel_id=p,
            )
            t["extract_candidates_s"] = time.time() - t_c
            t["n_candidates"] = int(len(cand))
        t["panel_id"] = p
        per_panel.append(t)
        print(t)

    total = sum(t["panel_total_s"] for t in per_panel)
    print()
    print(f"Total wall: {total:.1f}s for {len(per_panel)} panels  ==> {total/len(per_panel):.2f}s/panel")
    print(f"Projected for 189 detectors (1 visit): {189 * total / len(per_panel):.1f}s "
          f"= {189 * total / len(per_panel) / 60:.2f} min")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Test HDF5 file (test_5sigma/test.h5 etc.)")
    ap.add_argument("--model", required=True, help="TorchScript .pt file")
    ap.add_argument("--n-panels", type=int, default=3)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128, help="Default tile size = no overlap")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-threads", type=int, default=None, help="torch.set_num_threads")
    ap.add_argument("--use-hann", action="store_true")
    ap.add_argument("--extract-candidates", action="store_true")
    args = ap.parse_args()
    _benchmark(args)


if __name__ == "__main__":
    main()
