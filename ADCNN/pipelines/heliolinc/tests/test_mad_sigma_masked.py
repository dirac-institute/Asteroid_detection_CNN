"""A majority-masked panel must not produce a ~0 noise scale.

Some real panels are >80% exactly-zero (masked / no-data). The all-pixel median of |x| is then 0 and
diffim_mad_sigma returned its 1e-8 floor, so every consumer dividing by it exploded: in the shipped
0706 catalogue mf_snr reached 1.38e12, with 22,221 detections (0.58%) above 1e5 across 770 panels.
`mfsnr_min_2v` cannot reject those -- they sit ABOVE the gate by twelve orders of magnitude.
"""
import numpy as np

from ADCNN.data.preprocessing import diffim_mad_sigma
from ADCNN.inference.mf_trail_length import refine_trail_length, MF_SIGMA_MIN


def _panel(frac_zero, sigma=15.0, shape=(512, 512), seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sigma, shape).astype(np.float32)
    n = int(frac_zero * a.size)
    idx = rng.choice(a.size, n, replace=False)
    a.ravel()[idx] = 0.0
    return a


def test_normal_panel_is_unchanged():
    """Zeros in a minority -> the historical all-pixel path, exactly."""
    a = _panel(0.05)
    good = a[np.isfinite(a)]
    legacy = float(1.4826 * np.median(np.abs(good)) + 1e-8)
    assert abs(diffim_mad_sigma(a) - legacy) < 1e-9


def test_majority_masked_panel_recovers_a_sane_sigma():
    for frac in (0.60, 0.83, 0.95):
        s = diffim_mad_sigma(_panel(frac))
        assert s > 1.0, f"{frac:.0%} masked -> sigma {s:.3e}, still degenerate"
        assert 5.0 < s < 40.0, f"{frac:.0%} masked -> sigma {s:.3f} not a plausible noise scale"


def test_all_zero_panel_does_not_explode():
    assert diffim_mad_sigma(np.zeros((64, 64), np.float32)) >= 1.0


def test_trail_estimator_refuses_a_degenerate_sigma():
    """`sig <= 0` was not enough: the +1e-8 floor passes it and the estimator divides by ~0."""
    img = _panel(0.05)
    x = np.array([100.0, 200.0]); y = np.array([100.0, 200.0])
    Lin = np.array([25.0, 25.0]); Bin = np.zeros(2)
    L, B = refine_trail_length(x, y, img, Lin, Bin, sigma=1e-8)
    assert np.allclose(L, Lin), "a 1e-8 sigma must fall back to the incumbent, not divide by it"
    assert MF_SIGMA_MIN > 1e-8
