"""v7 inference primitive — sliding-window prediction over a full diffim panel.

``predict_panel_overlap_3ch_full`` tiles a panel into 128px windows, runs the v7 model
(seg + orientation + line-aggregator heads), and Hann-blends the overlapping tiles into
full-panel maps (prob, sin2β, cos2β, aggregator). This is stage 1 of the detector and is
used by run_inference, the RF feature/training code, and the real-data evaluation.
"""
from __future__ import annotations
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from ADCNN.data.preprocessing import build_3channel, diffim_mad_sigma


def _tile_starts(N, t, sstep):
    """Sliding-window start offsets tiling [0, N) with `t`-wide windows at stride
    `sstep`, always including the flush-right final tile (N - t)."""
    if N <= t:          # panel smaller than a tile: one start at 0 (avoids a negative N-t start)
        return [0]
    out = list(range(0, N - t + 1, sstep))
    if out[-1] != N - t:
        out.append(N - t)
    return out


_TILE_BATCH = int(os.environ.get("ADCNN_TILE_BATCH", "24"))  # tiles/forward; env-tunable
# threads for the per-tile build_3channel preprocessing (CPU, GIL-releasing); overlaps with
# the GPU forward and removes the serial-prep bottleneck. env-tunable.
_PREP_WORKERS = int(os.environ.get("ADCNN_PREP_WORKERS", str(min(8, os.cpu_count() or 4))))


def hann2d(tile: int) -> np.ndarray:
    """2D Hann window for blending overlapping tile predictions."""
    w = np.hanning(tile + 2)[1:-1]
    return (w[:, None] * w[None, :]).astype(np.float32)


def predict_panel_overlap_3ch(
    model: torch.nn.Module,
    panel_image: np.ndarray,
    panel_real_labels: np.ndarray,
    *,
    device,
    tile: int = 128,
    stride: int = 64,
    clip: float = 5.0,
    stats_crop: int = 1024,
) -> np.ndarray:
    """Sliding-window inference with Hann-weighted averaging on 3-channel input."""
    H, W = panel_image.shape
    s = min(stats_crop, H, W)
    h0c = (H - s) // 2
    w0c = (W - s) // 2
    sigma = diffim_mad_sigma(panel_image[h0c:h0c + s, w0c:w0c + s])

    prob_acc = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    hann = hann2d(tile)

    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    batch_xs, batch_locs = [], []
    BATCH = _TILE_BATCH  # tiles/forward (env ADCNN_TILE_BATCH); batching does not change results

    def flush():
        if not batch_xs:
            return
        # Each entry is already (3, T, T); stack to (B, 3, T, T)
        xb = torch.from_numpy(np.stack(batch_xs)).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seg_logits, _, _, _, _ = model(xb)
        probs = torch.sigmoid(seg_logits).cpu().numpy().astype(np.float32)
        for (y0, x0), p in zip(batch_locs, probs[:, 0]):
            prob_acc[y0:y0 + tile, x0:x0 + tile] += p * hann
            weight_acc[y0:y0 + tile, x0:x0 + tile] += hann
        batch_xs.clear(); batch_locs.clear()

    for y0 in ys:
        for x0 in xs:
            diffim_tile = panel_image[y0:y0 + tile, x0:x0 + tile]
            rl_tile = panel_real_labels[y0:y0 + tile, x0:x0 + tile]
            x3 = build_3channel(diffim_tile, rl_tile, panel_sigma=sigma, clip=clip)
            batch_xs.append(x3)
            batch_locs.append((y0, x0))
            if len(batch_xs) >= BATCH:
                flush()
    flush()

    out = prob_acc / np.maximum(weight_acc, 1e-6)
    return out.astype(np.float16)


@torch.no_grad()


def predict_panel_overlap_3ch_full(
    model: torch.nn.Module,
    panel_image: np.ndarray,
    panel_real_labels: np.ndarray,
    *,
    device,
    tile: int = 128,
    stride: int = 64,
    clip: float = 5.0,
    stats_crop: int = 1024,
    tile_batch: int | None = None,
    prep_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window inference returning all auxiliary heads.

    Returns (prob, orient_sin, orient_cos, agg) — each (H, W) float16. `prob`
    is sigmoid(seg_logits); `orient_sin`/`orient_cos` are tanh-bounded
    sin(2β)/cos(2β); `agg` is the raw line-aggregator logit. All four maps
    use Hann-weighted overlap blending (same convention as
    `predict_panel_overlap_3ch`).

    `tile_batch` (tiles per GPU forward) and `prep_workers` (CPU threads building the
    per-tile 3-channel input, overlapping the GPU) are speed knobs only — they do not
    change the output. Both default to the module values (env ADCNN_TILE_BATCH /
    ADCNN_PREP_WORKERS).
    """
    H, W = panel_image.shape
    s = min(stats_crop, H, W)
    h0c = (H - s) // 2
    w0c = (W - s) // 2
    sigma = diffim_mad_sigma(panel_image[h0c:h0c + s, w0c:w0c + s])

    prob_acc = np.zeros((H, W), dtype=np.float32)
    sin_acc  = np.zeros((H, W), dtype=np.float32)
    cos_acc  = np.zeros((H, W), dtype=np.float32)
    agg_acc  = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    hann = hann2d(tile)

    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    BATCH = _TILE_BATCH if tile_batch is None else int(tile_batch)  # tiles/forward; batching is exact
    workers = _PREP_WORKERS if prep_workers is None else int(prep_workers)

    def flush(bx, bl):
        if not bx:
            return
        xb = torch.from_numpy(np.stack(bx)).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seg_logits, sn, cs, _, ag = model(xb)
        probs = torch.sigmoid(seg_logits).detach().float().cpu().numpy()
        sn = sn.detach().float().cpu().numpy()
        cs = cs.detach().float().cpu().numpy()
        ag = ag.detach().float().cpu().numpy()
        for (y0, x0), p, s_, c_, a_ in zip(bl, probs[:, 0], sn[:, 0], cs[:, 0], ag[:, 0]):
            prob_acc[y0:y0+tile, x0:x0+tile] += p * hann
            sin_acc[y0:y0+tile, x0:x0+tile]  += s_ * hann
            cos_acc[y0:y0+tile, x0:x0+tile]  += c_ * hann
            agg_acc[y0:y0+tile, x0:x0+tile]  += a_ * hann
            weight_acc[y0:y0+tile, x0:x0+tile] += hann

    coords = [(y0, x0) for y0 in ys for x0 in xs]

    def _build(loc):
        y0, x0 = loc
        return build_3channel(panel_image[y0:y0+tile, x0:x0+tile],
                              panel_real_labels[y0:y0+tile, x0:x0+tile],
                              panel_sigma=sigma, clip=clip)

    # Build the per-tile 3-channel inputs in parallel: build_3channel is ~40% of inference
    # wall time, CPU-bound (scipy uniform_filter releases the GIL), independent per tile, and
    # was otherwise serial with the GPU forward. Bit-identical to the serial build.
    if workers > 1 and len(coords) > BATCH:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            x3s = list(ex.map(_build, coords))
    else:
        x3s = [_build(c) for c in coords]
    for i in range(0, len(coords), BATCH):
        flush(x3s[i:i + BATCH], coords[i:i + BATCH])

    wmax = np.maximum(weight_acc, 1e-6)
    return (
        (prob_acc / wmax).astype(np.float16),
        (sin_acc  / wmax).astype(np.float16),
        (cos_acc  / wmax).astype(np.float16),
        (agg_acc  / wmax).astype(np.float16),
    )
