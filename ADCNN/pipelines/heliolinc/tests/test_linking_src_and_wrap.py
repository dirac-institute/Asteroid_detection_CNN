"""Three linking invariants, each written against a measured defect.

1. mf_snr is NOT commensurable across `src`. ingest_diasource writes the stack's PSF point-source snr
   into the column the linker reads as matched-filter TRAIL snr. Measured on 150,277 stack<->ADCNN
   detections matched within 1" on 0706: the stack value is a median 1.43x LARGER while its trail on
   the same source is 2-10x SHORTER. FIX 1 ("trust the epoch that resolved the trail") therefore
   picked the stack epoch on 86% of 125 real mixed pairs, and that was the WORSE-agreeing trail 59%
   of the time -- it selected exactly the member it was written to distrust.

2. The scalar pair_chi2 and the vectorised prefilter must agree. A divergence prunes pairs the full
   chi2 accepts -- a recall loss with no trace in any log. That has already happened once here.

3. RA differences must wrap. A raw subtraction reads a small step across RA=0 as ~360 deg.
"""
import numpy as np
import pandas as pd
import pytest

import ADCNN.linking.link_2visit as L

LEN_A, LEN_S = 13.09, 6.06      # measured median len_db: adcnn vs stack on the SAME sources
SNR_A, SNR_S = 4.62, 7.40       # measured median mf_snr -- the stack number is the LARGER one
PIX, EXP, DT_MIN = 0.2, 30.0, 40.0


def _mixed_pair(order="as", ra0=10.0):
    """One object, two epochs, different `src`, with self-consistent geometry: the trail lies ALONG
    the motion and the chord matches the ADCNN trail, so the ADCNN member reads dspeed ~ 0."""
    rate = (LEN_A * PIX / 3600.0) / (EXP / 86400.0)
    delta = rate * (DT_MIN / 1440.0)
    specs = [("adcnn", SNR_A, LEN_A), ("stack", SNR_S, LEN_S)]
    if order == "sa":
        specs = specs[::-1]
    rows = []
    for k, (src, snr, ln) in enumerate(specs):
        ra = ra0 + k * delta; dec = -20.0
        half = (ln * PIX / 3600.0) / 2 / np.cos(np.radians(dec))
        rows.append(dict(visit=900 + k, detector=1, ra=ra % 360.0, dec=dec,
                         mjd=60000.0 + k * DT_MIN / 1440.0, score=0.9, len_db=ln, mf_snr=snr,
                         beta=0.0, src=src, ra0=(ra - half) % 360.0, dec0=dec,
                         ra1=(ra + half) % 360.0, dec1=dec))
    return pd.DataFrame(rows)


def _chi2(g):
    c = L.pair_chi2(g)
    return float(np.ravel(c[0] if isinstance(c, tuple) else c)[0])


def test_mixed_source_pair_is_judged_on_the_adcnn_trail():
    """Source-blind, the stack member's larger mf_snr wins and its 0.46x trail reads as a big dspeed;
    source-aware, the ADCNN trail is used and the pair clears the shipped chi2_2v_max of 8.0."""
    if L.LEN_SRC_BLIND:
        pytest.skip("ADCNN_LEN_SRC_BLIND=1 selects the legacy source-blind comparison")
    assert _chi2(_mixed_pair("as")) < 8.0, "a well-measured mixed pair must clear the shipped gate"


def test_result_does_not_depend_on_which_epoch_is_listed_first():
    a, b = _chi2(_mixed_pair("as")), _chi2(_mixed_pair("sa"))
    assert abs(a - b) < 0.05, f"epoch order changed chi2: {a} vs {b}"


def test_scalar_and_vectorised_paths_agree_on_a_mixed_pair():
    g = _mixed_pair("as")
    c = _chi2(g)
    # ONE-SIDED by design: the prefilter omits the non-negative orbit-residual term, so its partial
    # chi2 is a LOWER bound. It may keep a pair the full chi2 later rejects (harmless); it must never
    # PRUNE one the full chi2 accepts (a silent recall loss). Only that direction is asserted.
    assert len(L.prefilter_2v_pairs(g, [[0, 1]], chi2_max=c)) == 1, \
        "the prefilter pruned a pair the scalar chi2 accepts at the same threshold"


def test_ra_wrap_gives_the_same_chi2_across_the_meridian():
    """Straddling the meridian must not change the verdict. Straddling visits are real: 4 each on
    0629, 0708 and 0711 (31,511 detections).

    NOT exactly invariant, and it should not be: the bound-orbit residual depends on the true sky
    direction relative to Earth, so chi2 drifts slightly with RA everywhere. The control is what
    separates physics from a wrap bug -- a GENUINE 180 deg move (RA 10 -> 190) shifts chi2 by 0.003,
    so anything far larger at the meridian is arithmetic, not sky. Before the fix this pair read
    82.25 at RA 359.98 against 4.52 at RA 10: 10x over the shipped chi2_2v_max of 8.0, i.e. a real
    mover rejected for crossing zero. Three sites were unwrapped -- pair_chi2's private tv() copy
    (the module-level trail_velocity had always wrapped), the collinearity reference, and the chord
    rate feeding orbit_check.fit_orbit.
    """
    base = _chi2(_mixed_pair("as", ra0=10.0))
    rotated = _chi2(_mixed_pair("as", ra0=190.0))          # control: a real 180 deg rotation
    straddle = _chi2(_mixed_pair("as", ra0=359.98))        # the meridian case
    drift = abs(rotated - base)
    assert drift < 0.05, f"control drift {drift} too large -- the control is not a control"
    assert abs(straddle - base) < 10 * max(drift, 1e-3), \
        f"meridian chi2 {straddle} vs {base} moves far more than the {drift} of a genuine rotation"


def test_dra_helper_wraps():
    assert abs(float(L._dra(0.05, 359.95)) - 0.10) < 1e-9
    assert abs(float(L._dra(359.95, 0.05)) + 0.10) < 1e-9
