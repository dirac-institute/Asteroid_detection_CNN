"""TEMPLATE-BANK matched-filter trail length — replaces the footprint-extent estimator.

WHY. The incumbent `mf_length` (matched_filter.matched_filter_for_nn_candidates) integrates along the
principal axis of the THRESHOLDED footprint, so its length is whatever the threshold happened to
claim. On a faint trail the threshold keeps only the bright middle and the ends are lost. Measured
against injected truth on a faint population (true lengths 14-40px, SNR 2-6): the footprint estimator
returns a MEDIAN 13.17px where the truth median is ~24px, and 47.6% of faint 40-56px trails come back
under 60% of true length.

That matters because the linker checks whether a trail's length is consistent with how far the object
moved between visits. A faint FAST mover measured at half length looks like a SLOW one, the dspeed
chi2 term fires, and a real detection is discarded for a mismeasurement of its own making.

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

and the dspeed chi2 penalty every fast pair paid drops from 0.45 sigma to 0.01 sigma. End-to-end at
the binding 1k budget the delivered completeness goes 9.39% -> 12.70% (McNemar p < 1e-4).

ROBUST TO PSF MISMATCH: with the template sigma 31% wrong (2.1 vs 1.6) and MF_K refit, the median
error is 12.5% vs 11.9% and the p90 is BETTER. It does not depend on knowing the PSF.

DO NOT APPLY THE MF_LEN ENDS-BLOOM DE-BIAS TO THESE LENGTHS. That correction (offset/slope) was
calibrated for the FOOTPRINT estimator's bias; applying it here double-corrects. `catalog.py` skips
it when this estimator is active.

NO LENGTH FALLBACK (MF_MIN_LEN=0). A coarse-grid run once suggested the template was worse below
~10px, but that grid aligned some injected lengths with grid points and not others, manufacturing the
effect; the fine-grid calibrated measurement shows the template wins at every length. Any fallback
would also have to be gated on the template's OWN output, never on the incoming length -- the
incumbent's failure mode IS truncation, so a real 40px trail returned at 5px would be skipped as "too
short" and keep its bad value, excluding exactly the detections this estimator exists to rescue.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.special import erf

__all__ = ["refine_trail_length", "MF_K", "MF_STAMP"]

MF_STAMP = 96            # >= max half-length (56/2) + PSF wings
MF_K = 1.0518            # global calibration; leaves a residual signed bias of -0.00%
MF_DELTA = 1.0           # peak-centroid window, in units of S
# NO length fallback. An earlier COARSE-grid run (3px length steps) suggested the template was worse
# below ~10px, but that grid put the injected lengths 7/10/28/40 exactly on grid points and 14/20/56
# between them, manufacturing the effect. The fine-grid calibrated measurement contradicts it: the
# template beats the footprint at EVERY length, 7px included (26.7% -> 20.0% median error). Measured
# end to end here, falling back below 10px costs median 11.2% -> 14.2%, p90 41.3% -> 56.1% and
# truncation 0.6% -> 6.9%. Set >0 only with evidence from a fine grid.
MF_MIN_LEN = float(os.environ.get("ADCNN_MF_MIN_LEN", "0"))
MF_PSF_SIGMA = float(os.environ.get("ADCNN_MF_PSF_SIGMA", "1.6"))
MF_SIGMA_MIN = 1e-4      # nJy diffims sit at sigma ~15; anything near 0 is a masked panel
MF_L = np.arange(4, 80, 1.0)
MF_B = np.arange(0, 180, 3.0)

_TPL = None


def _templates(sigma_px: float = MF_PSF_SIGMA, stamp: int = MF_STAMP) -> np.ndarray:
    """(n_template, stamp*stamp) unit-L2 templates, built once and cached."""
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


def refine_trail_length(x, y, img, length_in, beta_in, sigma=None):
    """Template-bank length/angle at each (x, y). Returns (length, beta), incumbent where unusable.

    Detections too close to the panel edge for a full stamp keep their incoming values, as do those
    the TEMPLATE itself measures below MF_MIN_LEN (where a near-PSF source does not constrain the
    angle). The fallback is decided by the OUTPUT, never by the incoming length.
    """
    global _TPL
    x = np.asarray(x, float); y = np.asarray(y, float)
    L = np.asarray(length_in, float).copy(); B = np.asarray(beta_in, float).copy()
    if not len(x):
        return L, B
    if _TPL is None:
        _TPL = _templates()
    H, W = img.shape
    c = MF_STAMP // 2
    # Gate on the OUTPUT, never on the incoming length. The incumbent's failure mode IS truncation
    # (a real 40px trail can come back at 5px), so gating on it would skip exactly the detections
    # this estimator exists to rescue. Run the bank on every in-bounds detection, then fall back
    # only where the TEMPLATE says the trail is genuinely short -- the regime where a near-PSF source
    # does not constrain the angle and the scan finds spurious maxima.
    ok = ((x > c) & (x < W - c) & (y > c) & (y < H - c))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return L, B
    # Noise scale. PROFILED: recomputing this dominated the estimator -- 277 ms of a 318 ms call at
    # the real density of 226 detections/panel (87%), because it took TWO full 4kx4k medians while
    # the template bank itself costs 20 ms. The pipeline ALREADY computes a panel sigma during
    # candidate extraction (features.panel_sigmas via preprocessing.diffim_mad_sigma), so callers
    # should pass it in. The fallback uses that SAME canonical estimator -- median(|x|), one median,
    # not median(|x - median(x)|) -- so an explicitly supplied sigma and the fallback agree exactly.
    if sigma is None:
        from ADCNN.data.preprocessing import diffim_mad_sigma
        sig = diffim_mad_sigma(img)
    else:
        sig = float(sigma)
    # `sig <= 0` is NOT enough: diffim_mad_sigma adds a +1e-8 floor, so a fully masked panel yields
    # 1e-8, passes that guard, and the estimator divides by it -- returning garbage lengths (median
    # 43.6px where the incumbent said 25.0). Refuse below a floor that no real nJy diffim approaches.
    if not np.isfinite(sig) or sig <= MF_SIGMA_MIN:
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
    use = Lhat >= MF_MIN_LEN                     # fall back below the template bank's usable regime
    L[idx[use]] = Lhat[use]
    B[idx[use]] = Bhat[use]
    return L, B
