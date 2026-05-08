"""Signed, zero-centered, variance-aware normalization for difference images.

The direct-image pipeline used MAD around the sky median and clipped to [-5σ, +5σ].
That is wrong for diffims: the diffim is already zero-mean by construction, its noise
is roughly symmetric, and the science signal lives equally in positive and negative
residuals (positive for new sources, negative for dipoles and for sources in the
template but not in the science visit).

This module provides three normalizations, all of which preserve sign and all of
which are zero-centered. Pick one per experiment and stick with it — never subtract
a running median from a diffim.
"""
from __future__ import annotations
import numpy as np

ArrayF = np.ndarray


def _masked_finite(arr: ArrayF, bad_mask: ArrayF | None) -> ArrayF:
    """Return the subset of `arr` that is finite and not masked."""
    good = np.isfinite(arr)
    if bad_mask is not None:
        good &= ~bad_mask.astype(bool, copy=False)
    return arr[good]


def diffim_mad_sigma(
    arr: ArrayF,
    *,
    bad_mask: ArrayF | None = None,
) -> float:
    """Scalar noise scale from the (masked) diffim pixels around zero.

    Uses median(|x|) rather than median(|x - median(x)|) because we want the
    scale of a zero-centered distribution, not a robust scale around an
    unknown center. The 1.4826 factor converts MAD→sigma for a Gaussian.
    """
    good = _masked_finite(arr.astype(np.float32, copy=False), bad_mask)
    if good.size == 0:
        return 1.0
    mad = float(np.median(np.abs(good)))
    return float(1.4826 * mad + 1e-8)


def normalize_diffim_mad(
    arr: ArrayF,
    *,
    bad_mask: ArrayF | None = None,
    clip: float = 5.0,
) -> ArrayF:
    """Panel-level MAD normalization for a diffim.

    No median subtraction. Divides by a robust scale estimated from the whole
    array (or from finite, non-masked pixels) and symmetrically clips to
    ±`clip`. Preserves sign.
    """
    sigma = diffim_mad_sigma(arr, bad_mask=bad_mask)
    z = arr.astype(np.float32, copy=False) / sigma
    z = np.clip(z, -clip, clip, out=None)
    return z


def normalize_diffim_variance(
    arr: ArrayF,
    variance: ArrayF,
    *,
    bad_mask: ArrayF | None = None,
    clip: float = 5.0,
    variance_floor_quantile: float = 0.02,
) -> ArrayF:
    """Pixel-wise SNR normalization.

    Divides the diffim by `sqrt(variance)` pixel by pixel, so the output is an
    S/N map with unit noise everywhere the variance plane is trustworthy. Very
    small variance values (bad pixels, masked regions) are floored at a low
    quantile to prevent blow-up.

    This is the strongest "variance-aware" normalization and the natural
    default for diffim training because the subtraction-matched variance plane
    already encodes the per-pixel noise budget of the (science, template, PSF-
    matching-kernel) combination.
    """
    v = variance.astype(np.float32, copy=False)
    finite = np.isfinite(v) & (v > 0)
    if bad_mask is not None:
        finite &= ~bad_mask.astype(bool, copy=False)
    if finite.any():
        floor = float(np.quantile(v[finite], variance_floor_quantile))
    else:
        floor = 1.0
    v_eff = np.maximum(v, floor)
    z = arr.astype(np.float32, copy=False) / np.sqrt(v_eff + 1e-12)
    # Replace non-finite (should be rare) with zero.
    z = np.where(np.isfinite(z), z, 0.0).astype(np.float32, copy=False)
    return np.clip(z, -clip, clip)


def normalize_variance_channel(
    variance: ArrayF,
    *,
    bad_mask: ArrayF | None = None,
) -> ArrayF:
    """Compress the variance plane for use as a NN input channel.

    Returns log1p(variance / median_variance). Robust to outliers.
    """
    v = variance.astype(np.float32, copy=False)
    finite = np.isfinite(v) & (v > 0)
    if bad_mask is not None:
        finite &= ~bad_mask.astype(bool, copy=False)
    if finite.any():
        med = float(np.median(v[finite]))
        med = max(med, 1e-12)
    else:
        med = 1.0
    out = np.log1p(np.maximum(v, 0.0) / med)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# A tiny self-test runnable as `python -m ADCNN.data.diffim_norm`.
# ---------------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)
    # fake diffim: zero-mean gaussian + a few trail pixels
    arr = rng.standard_normal((256, 256)).astype(np.float32) * 3.0
    arr[100:105, 20:200] += 15.0  # a trail residual
    var = np.full_like(arr, 9.0)

    sigma = diffim_mad_sigma(arr)
    z1 = normalize_diffim_mad(arr, clip=5.0)
    z2 = normalize_diffim_variance(arr, var, clip=5.0)
    print(f"sigma_MAD={sigma:.3f} (expected ~3.0)")
    print(f"z_mad: min={z1.min():.2f} max={z1.max():.2f} median={np.median(z1):.3f}")
    print(f"z_var: min={z2.min():.2f} max={z2.max():.2f} median={np.median(z2):.3f}")
    assert z1.min() >= -5 and z1.max() <= 5
    assert z2.min() >= -5 and z2.max() <= 5
    # zero-centered assertion: the noise part should be close to zero median
    assert abs(float(np.median(z1))) < 0.1
    print("OK")


if __name__ == "__main__":
    _self_test()
