"""segmentation model inference primitive — sliding-window prediction over a full diffim panel.

``predict_panel_overlap_3ch_full`` tiles a panel into 128px windows, runs the segmentation
model, and Hann-blends the overlapping tiles into full-panel maps of the seg probability and the
line aggregator (the orientation heads are not used: see matched_filter). This is stage 1 of the
detector.
"""
from __future__ import annotations
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

import torch
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


TILE = 128      # window side (px)
STRIDE = 64     # window stride (px)
CLIP = 5.0      # ch0 clip (MAD-sigma)
_TILE_BATCH = int(os.environ.get("ADCNN_TILE_BATCH", "24"))  # tiles/forward; env-tunable
# threads for the per-tile build_3channel preprocessing (CPU, GIL-releasing); overlaps with
# the GPU forward and removes the serial-prep bottleneck. env-tunable.
_PREP_WORKERS = int(os.environ.get("ADCNN_PREP_WORKERS", str(min(8, os.cpu_count() or 4))))


def hann2d(tile: int) -> np.ndarray:
    """2D Hann window for blending overlapping tile predictions."""
    w = np.hanning(tile + 2)[1:-1]
    return (w[:, None] * w[None, :]).astype(np.float32)


@torch.no_grad()
def predict_panel_overlap_3ch_full(
    model: torch.nn.Module,
    panel_image: np.ndarray,
    panel_real_labels: np.ndarray,
    *,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window inference returning (prob, agg), each (H, W) float16.

    `prob` is sigmoid(seg_logits); `agg` is the raw line-aggregator logit. Both maps use
    Hann-weighted overlap blending.

    ADCNN_TILE_BATCH (tiles per GPU forward) and ADCNN_PREP_WORKERS (CPU threads building the
    per-tile 3-channel input, overlapping the GPU) are speed knobs only — they do not change the
    output.
    """
    H, W = panel_image.shape
    # Estimate on the WHOLE panel, not a centre crop. MEASURED: 3 of 120 sampled 0706 panels have
    # their centre-1024 crop 100% masked, so diffim_mad_sigma hits its `return 1.0` escape while the
    # true panel sigma is 31-47 -- build_3channel then does clip(img/1.0, +-5) and every noise pixel
    # saturates. On panel 10109 the crop is 40.9% zero: 17.0 vs 58.5 for the panel. The estimator is
    # a median over nonzero pixels, so the crop bought precision that was never the bottleneck.
    sigma = diffim_mad_sigma(panel_image)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"degenerate panel sigma {sigma!r}: panel is fully masked or non-finite")

    prob_acc = np.zeros((H, W), dtype=np.float32)
    agg_acc  = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    hann = hann2d(TILE)

    ys = _tile_starts(H, TILE, STRIDE)
    xs = _tile_starts(W, TILE, STRIDE)

    def flush(bx, bl):
        if not bx:
            return
        xb = torch.from_numpy(np.stack(bx)).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seg_logits, _, _, _, ag = model(xb)
        probs = torch.sigmoid(seg_logits).detach().float().cpu().numpy()
        ag = ag.detach().float().cpu().numpy()
        for (y0, x0), p, a_ in zip(bl, probs[:, 0], ag[:, 0]):
            prob_acc[y0:y0+TILE, x0:x0+TILE] += p * hann
            agg_acc[y0:y0+TILE, x0:x0+TILE]  += a_ * hann
            weight_acc[y0:y0+TILE, x0:x0+TILE] += hann

    coords = [(y0, x0) for y0 in ys for x0 in xs]

    def _build(loc):
        y0, x0 = loc
        return build_3channel(panel_image[y0:y0+TILE, x0:x0+TILE],
                              panel_real_labels[y0:y0+TILE, x0:x0+TILE],
                              panel_sigma=sigma, clip=CLIP)

    # Build the per-tile 3-channel inputs in parallel: build_3channel is ~40% of inference
    # wall time, CPU-bound (scipy uniform_filter releases the GIL), independent per tile, and
    # was otherwise serial with the GPU forward. Bit-identical to the serial build.
    if _PREP_WORKERS > 1 and len(coords) > _TILE_BATCH:
        with ThreadPoolExecutor(max_workers=_PREP_WORKERS) as ex:
            x3s = list(ex.map(_build, coords))
    else:
        x3s = [_build(c) for c in coords]
    for i in range(0, len(coords), _TILE_BATCH):
        flush(x3s[i:i + _TILE_BATCH], coords[i:i + _TILE_BATCH])

    wmax = np.maximum(weight_acc, 1e-6)
    return (
        (prob_acc / wmax).astype(np.float16),
        (agg_acc  / wmax).astype(np.float16),
    )
