"""Diffim preprocessing primitives — shared by the dataset, training, and inference.

The segmentation model detector consumes a 3-channel tile (``build_3channel``):
  ch0 = signed diffim / MAD-sigma, clipped to ±5
  ch1 = log1p-compressed local std of ch0 (noise/edge/saturation context)
  ch2 = DIA/artefact mask (real_labels > 0)
plus per-pixel sin(2β)/cos(2β) orientation targets (``_panel_orient_maps``).
``diffim_mad_sigma`` is the robust per-panel noise scale used to normalise everything.
"""
from __future__ import annotations
import numpy as np


def diffim_mad_sigma(arr: np.ndarray) -> float:
    """Robust noise scale of a zero-mean diffim. median(|x|) * 1.4826.

    MASKED PIXELS ARE NOT NOISE SAMPLES. Some panels are majority exactly-zero (masked/no-data): on
    0706, 2 of 40 sampled panels are >80% zeros. There the all-pixel median is 0, this returned its
    1e-8 floor, and every consumer dividing by it exploded -- `mf_snr = flux/(sigma*sqrt(n))` reached
    1.38e12, with 22,221 detections (0.58%) above 1e5 across 770 panels. `mfsnr_min_2v` cannot reject
    those: they are ABOVE the gate by twelve orders of magnitude, so they enter the linkable set and
    manufacture chance links.

    When zeros are a MINORITY the all-pixel median is used exactly as before, so normal panels are
    bit-for-bit unchanged; only degenerate panels take the nonzero-pixel path.
    """
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return 1.0
    nz = good[good != 0]
    # UNCONDITIONAL. A majority test leaves a cliff: panels 25-50% masked keep the all-pixel median,
    # which is a low quantile of |x| by construction. MEASURED on real 0706 panels -- zero_frac 0.4-0.5
    # gives sig_all/sig_nonzero = 0.161, worst case 4.63 vs 58.49 (12.6x under), ~569 panels/night in
    # that band. Downstream, mf_snr = flux/(sigma*sqrt(n)) inflates: on one such panel 15 of 40
    # production detections passed mfsnr_min ONLY because sigma was under-estimated. Exactly-zero
    # pixels are MASK, not noise samples, at every masking fraction.
    if nz.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(nz)) + 1e-8)


def _panel_orient_maps(masks_panel: np.ndarray, csv_panel) -> tuple:
    """Per-pixel sin(2β), cos(2β) for a panel, derived from its truth mask
    and the per-injection β. Each truth pixel takes the β of the nearest
    injection in (x,y)."""
    H, W = masks_panel.shape
    sin_map = np.zeros((H, W), dtype=np.float32)
    cos_map = np.zeros((H, W), dtype=np.float32)
    if len(csv_panel) == 0 or not masks_panel.any():
        return sin_map, cos_map
    ys, xs = np.nonzero(masks_panel)
    inj_xs = csv_panel["x"].to_numpy().astype(np.float32)
    inj_ys = csv_panel["y"].to_numpy().astype(np.float32)
    inj_betas = csv_panel["beta"].to_numpy().astype(np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.stack([inj_xs, inj_ys], axis=1))
        _, idx = tree.query(np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1), k=1)
    except Exception:
        idx = np.empty_like(xs)
        for k in range(xs.size):
            dd = (inj_xs - xs[k]) ** 2 + (inj_ys - ys[k]) ** 2
            idx[k] = int(np.argmin(dd))
    betas_assigned = inj_betas[idx]
    sin_vals = np.sin(np.radians(2.0 * betas_assigned)).astype(np.float32)
    cos_vals = np.cos(np.radians(2.0 * betas_assigned)).astype(np.float32)
    sin_map[ys, xs] = sin_vals
    cos_map[ys, xs] = cos_vals
    return sin_map, cos_map


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
