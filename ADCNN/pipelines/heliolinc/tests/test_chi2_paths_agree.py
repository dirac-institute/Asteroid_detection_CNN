"""The vectorised prefilter must agree with the scalar chi2, or it silently drops real pairs.

`_prefilter_pairs_2v` exists only to avoid running the expensive orbit solve on every candidate, so
it is REQUIRED to prune only what the full `pair_chi2` would reject anyway. If the two paths compute
different sigmas, the prefilter throws away pairs the full check accepts -- a recall loss that leaves
no trace in any log.

This is not hypothetical: the per-object dspeed sigma (FIX 2) was first added to the prefilter and
computed-but-unused in the scalar path, and the length-scaled PA sigma carries the same warning in
its own comment. A test is the only thing that keeps them in step.
"""
import numpy as np
import pandas as pd

import ADCNN.linking.link_2visit as L


PIX, EXPT, DAY = 0.2, 30.0, 86400.0


def _len_for_rate(rate_degday):
    """Trail length implied by a rate: the trail IS the motion smeared over one exposure."""
    return rate_degday * 3600.0 / PIX * (EXPT / DAY)


def _pair(mfsnr_a, mfsnr_b, length_a=None, length_b=None, rate=3.0):
    """Two epochs of a SELF-CONSISTENT linear mover, 37 min apart.

    The trail length must be the one the rate implies, or the pair is genuinely inconsistent and sits
    near the chi2 threshold, where the two code paths can straddle for reasons that have nothing to
    do with their sigma models. (First version of this test used 30px with a 3 deg/day motion -- a 60%
    speed mismatch -- and "failed" on its own construction.)
    """
    dt = 37.0 / 1440.0
    ra0, dec0 = 300.0, -20.0
    if length_a is None:
        length_a = _len_for_rate(rate)
    if length_b is None:
        length_b = _len_for_rate(rate)
    ra1, dec1 = ra0 + rate * dt / np.cos(np.radians(dec0)), dec0
    rows = []
    for i, (ra, dec, mjd, mf, ln) in enumerate([(ra0, dec0, 61228.0, mfsnr_a, length_a),
                                                (ra1, dec1, 61228.0 + dt, mfsnr_b, length_b)]):
        half = ln * 0.2 / 3600.0 / 2.0
        rows.append(dict(ra=ra, dec=dec, mjd=mjd, visit=1000 + i, detector=1,
                         ra0=ra - half / np.cos(np.radians(dec)), dec0=dec,
                         ra1=ra + half / np.cos(np.radians(dec)), dec1=dec,
                         len_db=ln, mf_snr=mf, score=0.9))
    return pd.DataFrame(rows)


def test_dspeed_sigma_is_snr_dependent_and_bounded():
    """FIX 2's sigma must vary with SNR and stay inside its clip range."""
    lo, hi = L.dspeed_sigma(3.0), L.dspeed_sigma(30.0)
    assert lo > hi, "sigma must LOOSEN for faint sources, not tighten"
    assert L.DSPEED_SIG_LO <= hi <= L.DSPEED_SIG_HI
    assert L.DSPEED_SIG_LO <= lo <= L.DSPEED_SIG_HI
    # a fixed 0.237 sits between the two extremes -- that is exactly why one constant cannot serve
    assert lo > 0.237 > hi


def test_scalar_and_vectorised_chi2_agree():
    """The prefilter's partial chi2 must not exceed the scalar chi2 for the same pair.

    partial omits the orbit-fit terms (resid), so it is a LOWER bound; if it ever came out higher,
    the prefilter would reject pairs the full chi2 keeps.
    """
    # The pair must have a NON-ZERO but acceptable dspeed, or the sigma never enters the arithmetic
    # and the test cannot see a divergence at all: 0 / any sigma is 0. A perfectly self-consistent
    # pair passes this test even with the prefilter's sigma deliberately broken 4x (verified by
    # mutation). Give the trail a 20% speed excess so both paths must actually divide by their sigma.
    for mfa, mfb in ((5.0, 4.0), (12.0, 11.0), (30.0, 25.0), (3.5, 3.2)):
        g = _pair(mfa, mfb, length_a=1.2 * _len_for_rate(3.0), length_b=1.2 * _len_for_rate(3.0))
        chi2, _ = L.pair_chi2(g, exptime_s=30.0)
        assert np.isfinite(chi2), f"scalar chi2 not finite at mfsnr {mfa}/{mfb}"
        # THE INVARIANT: anything the full chi2 accepts at a threshold must survive the prefilter at
        # that same threshold. The prefilter omits the orbit-fit `resid` term, so its partial chi2 is
        # a LOWER bound; if it ever exceeded the scalar value it would reject pairs the full check
        # keeps, and nothing downstream would record the loss.
        kept = L.prefilter_2v_pairs(g, [[0, 1]], chi2_max=float(chi2) + 1e-6, exptime_s=30.0)
        assert kept, (f"prefilter dropped a pair the scalar chi2 ACCEPTS at mfsnr {mfa}/{mfb} "
                      f"(chi2={chi2:.3f}) -- the two sigma models have diverged")


def test_prefilter_function_exists_with_expected_signature():
    """Guard against the test above silently passing if the function is renamed.

    An earlier version of this test referenced a non-existent name behind a hasattr() guard, so it
    passed while exercising nothing.
    """
    import inspect
    assert hasattr(L, "prefilter_2v_pairs")
    params = list(inspect.signature(L.prefilter_2v_pairs).parameters)
    assert params[:3] == ["dets", "pairs", "chi2_max"], params


def test_len_best_epoch_uses_the_brighter_epoch():
    """FIX 1: a disagreeing shallow epoch must not drive dspeed when the deep epoch is confident."""
    good = _pair(20.0, 20.0)
    # same object, but the FAINT epoch mismeasures its trail badly
    bad = _pair(20.0, 3.0, length_a=_len_for_rate(3.0), length_b=2 * _len_for_rate(3.0))
    c_good, _ = L.pair_chi2(good, exptime_s=30.0)
    c_bad, _ = L.pair_chi2(bad, exptime_s=30.0)
    assert np.isfinite(c_good)
    # with FIX 1 the confident epoch carries the speed, so the bad epoch must not blow the chi2 up
    # anywhere near as much as it would if both epochs had to agree.
    assert c_bad < 1e6, "a mismeasured faint epoch should not make the pair unlinkable"


def test_paths_agree_under_ASYMMETRY():
    """Symmetric pairs cannot expose min-vs-max or best-epoch divergences.

    With near-equal SNR, dspeed_sigma(min) ~= dspeed_sigma(max); with equal trail lengths, FIX 1's
    best-epoch speed equals the max-over-both. Both mutations were MISSED until these cases existed.
    A test that cannot fail is not a test.

    SCOPE: this checks the RECALL-CRITICAL direction only -- the prefilter must never be TIGHTER than
    the full chi2. A mutation making it LOOSER (e.g. sigma from min-SNR instead of max) is not caught,
    and should not be: a looser prefilter wastes orbit-solve time but cannot drop a real pair.
    """
    for mfa, mfb, la, lb in ((30.0, 3.2, 1.2, 1.2),    # huge SNR asymmetry -> min vs max sigma differ
                             (25.0, 4.0, 1.2, 0.6),    # asymmetric SNR *and* length -> FIX 1 bites
                             (4.0, 25.0, 0.6, 1.2),    # same, epochs swapped
                             (18.0, 5.0, 1.35, 0.9)):
        g = _pair(mfa, mfb, length_a=la * _len_for_rate(3.0), length_b=lb * _len_for_rate(3.0))
        chi2, _ = L.pair_chi2(g, exptime_s=30.0)
        if not np.isfinite(chi2):
            continue
        kept = L.prefilter_2v_pairs(g, [[0, 1]], chi2_max=float(chi2) + 1e-6, exptime_s=30.0)
        assert kept, (f"prefilter dropped a pair the scalar chi2 ACCEPTS "
                      f"(mfsnr {mfa}/{mfb}, len x{la}/x{lb}, chi2={chi2:.3f})")
