"""TEMPLATE-BANK matched-filter trail length — replaces the footprint-extent estimator.

WHY. The incumbent `mf_length` (matched_filter.matched_filter_for_nn_candidates) integrates along the
principal axis of the THRESHOLDED footprint, so its length is whatever the threshold happened to
claim. On a faint trail the threshold keeps only the bright middle and the ends are lost. Measured
against injected truth on a faint population (true lengths 14-40px, SNR 2-6): the footprint estimator
returns a MEDIAN 13.17px where the truth median is ~24px, and 47.6% of faint 40-56px trails come back
under 60% of true length.

ESTIMATOR. Correlate the stamp against PSF-convolved line templates spanning length x angle:

    S(L, beta) = sum(I * T) / (sigma * ||T||_2)

S peaks at the true length — shorter misses flux; longer grows ||T|| as sqrt(L) while the collected
signal stops growing. Two details matter and both were established by measurement, not assumption:

  * the along-track profile is a top-hat CONVOLVED with the PSF (a difference of two erfs), NOT a
    hard-edged segment. Sharp ends correlate against smeared ends and peak SHORT: a measured -10.8%
    systematic, and at 56px 11% is 6px ~ 2*sigma, which is the tell.
  * the estimator is the flux-weighted PEAK CENTROID within MF_DELTA of the maximum, not argmax. For
    a faint trail S(L) is flat near its peak, so argmax lands long — 4.9% of faint-fast came back
    over 150% of true length, a failure the footprint estimator does not have.

A single global factor MF_K removes a near-constant residual bias. That the bias is near-CONSTANT is
the whole reason this estimator is usable: its spread across length bins is 7.6 points where the
footprint estimator's is 26.3, and only a near-constant bias is removable with one number.

MEASURED (out-of-fold, leave-one-visit-pair-out, 2,296 injected trails), |fractional rate error|:

    population        footprint -> template   truncation (<60% of true)
    ALL                 20.5%  ->  11.9%          15.4% -> 1.9%
    FAST rate>4         14.2%  ->   8.1%           8.0% -> 3.1%
    FAINT SNR<6         23.3%  ->  16.0%          17.6% -> 3.2%

COST: 251 ms/panel at ONE BLAS thread (226 detections/panel).

ROBUST TO PSF MISMATCH: with the template sigma 31% wrong (2.1 vs 1.6) and MF_K refit, the median
error is 12.5% vs 11.9% and the p90 is BETTER. It does not depend on knowing the PSF.

DO NOT APPLY THE MF_LEN ENDS-BLOOM DE-BIAS TO THESE LENGTHS. That correction (offset/slope) was
calibrated for the FOOTPRINT estimator's bias; applying it here double-corrects. `catalog.py` applies
it only to the edge detections this estimator cannot measure.

NO LENGTH FALLBACK. A coarse-grid run once suggested the template was worse below ~10px, but that
grid aligned some injected lengths with grid points and not others, manufacturing the effect; the
fine-grid calibrated measurement shows the template wins at every length (7px: 26.7% -> 20.0% median
error; falling back below 10px costs median 11.2% -> 14.2%, p90 41.3% -> 56.1% and truncation
0.6% -> 6.9%). Any fallback would also have to be gated on the template's OWN output, never on the
incoming length -- the incumbent's failure mode IS truncation, so a real 40px trail returned at 5px
would be skipped as "too short" and keep its bad value, excluding exactly the detections this
estimator exists to rescue.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

MF_STAMP = 96            # >= max half-length (56/2) + PSF wings
MF_K = 0.8956            # re-derived for v2_D2 on the real-PSF campaign
MF_DELTA = 1.0           # peak-centroid window, in units of S
MF_PSF_SIGMA = 1.6       # template PSF sigma (px)
MF_SIGMA_MIN = 1e-4      # nJy diffims sit at sigma ~15; anything near 0 is a masked panel
MF_L = np.arange(4, 80, 1.0)
MF_B = np.arange(0, 180, 3.0)

_TPL = None


def _templates() -> np.ndarray:
    """(n_template, MF_STAMP*MF_STAMP) unit-L2 templates, built once and cached."""
    sigma_px, stamp = MF_PSF_SIGMA, MF_STAMP
    c = stamp // 2
    yy, xx = np.mgrid[0:stamp, 0:stamp].astype(np.float32)
    yy -= c; xx -= c
    out = []
    for L in MF_L:
        for b in MF_B:
            ca, sa = np.cos(np.radians(b)), np.sin(np.radians(b))
            s = xx * ca + yy * sa            # along-track
            p = -xx * sa + yy * ca           # cross-track
            along = 0.5 * (erf((0.5 * L - s) / (np.sqrt(2) * sigma_px))
                           + erf((0.5 * L + s) / (np.sqrt(2) * sigma_px)))
            t = along * np.exp(-0.5 * (p / sigma_px) ** 2)
            nrm = np.linalg.norm(t)
            out.append((t / nrm if nrm > 0 else t).ravel())
    return np.asarray(out, np.float32)


def refine_trail_length(x, y, img, length_in, beta_in, sigma):
    """Template-bank length/angle at each (x, y). Returns (length, beta), incumbent where unusable.

    `sigma` is the panel MAD-sigma (features.extract_panel_candidates). Detections too close to
    the panel edge for a full stamp keep their incoming values, as do all detections of a panel
    whose sigma is degenerate.
    """
    global _TPL
    x = np.asarray(x, float); y = np.asarray(y, float)
    L = np.asarray(length_in, float).copy(); B = np.asarray(beta_in, float).copy()
    if _TPL is None:
        _TPL = _templates()
    H, W = img.shape
    c = MF_STAMP // 2
    # Run the bank on every in-bounds detection, whatever its incoming length: the incumbent's
    # failure mode IS truncation (a real 40px trail can come back at 5px).
    ok = ((x > c) & (x < W - c) & (y > c) & (y < H - c))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return L, B
    sig = float(sigma)
    # Refuse a degenerate sigma: below a floor that no real nJy diffim approaches, or
    # diffim_mad_sigma's 1.0 "could not measure" sentinel for a fully masked panel. Dividing by
    # either returns the flat-S attractor (mutation-verified: an all-zero panel turns lengths
    # [9,20,33] into [43.65]x3, beta -> 0). Latent on real data (0 of 14,142 in-bounds detections).
    if not np.isfinite(sig) or sig <= MF_SIGMA_MIN or sig == 1.0:
        return L, B
    cuts = np.empty((len(idx), MF_STAMP * MF_STAMP), np.float32)
    for k, i in enumerate(idx):
        yi = int(round(y[i])) - c; xi = int(round(x[i])) - c
        cuts[k] = img[yi:yi + MF_STAMP, xi:xi + MF_STAMP].ravel()
    S = (cuts @ _TPL.T) / max(float(sig), 1e-6)
    S = S.reshape(len(idx), len(MF_L), len(MF_B))
    prof = S.max(axis=2)                                    # marginalise over angle
    w = np.clip(prof - (prof.max(1, keepdims=True) - MF_DELTA), 0.0, None)
    Lhat = (w * MF_L[None, :]).sum(1) / np.maximum(w.sum(1), 1e-9) * MF_K
    Bhat = MF_B[S.max(axis=1).argmax(axis=1)]
    # The bank tops out at MF_L[-1]; a trail longer than that saturates and reads SHORT with no
    # signal that it did (a 200 px trail returns 76.5 px, -62%). Say so rather than return it silently.
    _sat = int((Lhat >= MF_L[-1] * MF_K * 0.995).sum())
    if _sat:
        print(f"[mf_trail] {_sat} detection(s) at the {MF_L[-1]:.0f}px template-bank ceiling -- their "
              f"length is a LOWER BOUND, not a measurement", flush=True)
    use = Lhat >= 0.0            # False only for a NaN length: that detection keeps its incoming values
    L[idx[use]] = Lhat[use]
    B[idx[use]] = Bhat[use]
    return L, B
