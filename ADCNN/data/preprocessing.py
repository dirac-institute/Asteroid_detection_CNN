"""Diffim preprocessing primitives for inference.

The segmentation model detector consumes a 3-channel tile (``build_3channel``):
  ch0 = signed diffim / MAD-sigma, clipped to ±5
  ch1 = log1p-compressed local std of ch0 (noise/edge/saturation context)
  ch2 = DIA/artefact mask (real_labels > 0)
``diffim_mad_sigma`` is the robust per-panel noise scale used to normalise everything.
"""
from __future__ import annotations
import numpy as np


def diffim_mad_sigma(arr: np.ndarray) -> float:
    """Robust noise scale of a zero-mean diffim: 1.4826 * median(|x|) over the finite, nonzero pixels
    (1.0 when there are none).

    MASKED PIXELS ARE NOT NOISE SAMPLES. Some panels are majority exactly-zero (masked/no-data): on
    0706, 2 of 40 sampled panels are >80% zeros. There the all-pixel median is 0, this returned its
    1e-8 floor, and every consumer dividing by it exploded -- `mf_snr = flux/(sigma*sqrt(n))` reached
    1.38e12, with 22,221 detections (0.58%) above 1e5 across 770 panels.
    """
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return 1.0
    nz = good[good != 0]
    # UNCONDITIONAL. A majority test leaves a cliff: panels 25-50% masked keep the all-pixel median,
    # which is a low quantile of |x| by construction. MEASURED on real 0706 panels -- zero_frac 0.4-0.5
    # gives sig_all/sig_nonzero = 0.161, worst case 4.63 vs 58.49 (12.6x under), ~569 panels/night in
    # that band. Downstream, mf_snr = flux/(sigma*sqrt(n)) inflates. Exactly-zero
    # pixels are MASK, not noise samples, at every masking fraction.
    if nz.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(nz)) + 1e-8)


def local_std_panel(arr: np.ndarray, window: int = 11) -> np.ndarray:
    """Per-pixel std over a window-by-window box centered on the pixel.

    var = E[x^2] - E[x]^2 (clipped to 0). Uses uniform_filter for speed.
    """
    from scipy.ndimage import uniform_filter
    a = arr.astype(np.float32)
    mu = uniform_filter(a, size=window, mode="nearest")
    sq = uniform_filter(a * a, size=window, mode="nearest")
    var = np.clip(sq - mu * mu, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def normalize_local_std(local_std: np.ndarray) -> np.ndarray:
    """Map local_std into [0, 1]-ish for use as a NN channel.

    The diffim is already MAD-normalized (clipped to ±5), so its local std
    is bounded by ~5/sqrt(2) ≈ 3.5 for noise-only regions and much higher
    near saturated cores. We log1p-compress and clip.
    """
    return np.clip(np.log1p(local_std) / np.log1p(5.0), 0.0, 1.0).astype(np.float32)


def build_3channel(diffim_tile: np.ndarray, real_labels_tile: np.ndarray,
                   *, panel_sigma: float, clip: float = 5.0) -> np.ndarray:
    """Build the 3-channel input from one tile. Returns (3, H, W) float32."""
    z = np.clip(diffim_tile.astype(np.float32) / panel_sigma, -clip, clip)
    lstd = local_std_panel(z, window=11)
    lstd_n = normalize_local_std(lstd)
    rl = (real_labels_tile > 0).astype(np.float32)
    return np.stack([z.astype(np.float32), lstd_n, rl], axis=0)
