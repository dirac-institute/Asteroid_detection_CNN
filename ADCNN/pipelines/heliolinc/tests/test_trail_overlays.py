"""Unit tests for the QA trail-overlay generator (ADCNN/qa/trail_overlays.py):
capsule_outline geometry on a synthetic TAN WCS (closed path, edge lengths, centring,
PA orientation, minimum-length floor) and the fail-loud dets-row lookup match_det_row.
Pure functions -- no GPU, no Butler, no data files."""
import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from ADCNN.linking.pixel_vet import PXSCALE
from ADCNN.qa.trail_overlays import EXPTIME_S, HALFW_PX, capsule_outline, match_det_row

RA0, DEC0 = 150.0, -20.0


def _tan_wcs():
    """0.2''/px TAN WCS, north up / east left (cdelt1 < 0), crpix (100, 100)."""
    w = WCS(naxis=2)
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [-PXSCALE / 3600.0, PXSCALE / 3600.0]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


# --------------------------------------------------------------- capsule_outline
def test_capsule_closed_and_centred():
    w = _tan_wcs()
    (xs, ys), (x0, y0) = capsule_outline(w, RA0, DEC0, rate_degday=3.0, pa_deg=30.0)
    assert len(xs) == 5 and len(ys) == 5
    assert xs[0] == xs[-1] and ys[0] == ys[-1]                  # closed path
    ex, ey = map(float, w.world_to_pixel_values(RA0, DEC0))
    assert x0 == pytest.approx(ex) and y0 == pytest.approx(ey)  # anchored at the position
    # rectangle centre == capsule centre
    assert np.mean(xs[:4]) == pytest.approx(x0, abs=1e-6)
    assert np.mean(ys[:4]) == pytest.approx(y0, abs=1e-6)


def test_capsule_edge_lengths_match_rate():
    w = _tan_wcs()
    rate = 4.32                                # deg/day -> L = 4.32*30/86400*3600/0.2 = 27 px
    (xs, ys), _ = capsule_outline(w, RA0, DEC0, rate_degday=rate, pa_deg=77.0)
    c = np.c_[xs, ys]
    short = np.linalg.norm(c[1] - c[0])
    long = np.linalg.norm(c[2] - c[1])
    L = rate * EXPTIME_S / 86400.0 * 3600.0 / PXSCALE
    assert short == pytest.approx(2 * HALFW_PX, rel=1e-6)
    assert long == pytest.approx(L, rel=1e-3)                   # 27 px on this WCS
    assert long == pytest.approx(27.0, rel=1e-3)


def test_capsule_pa_orientation():
    w = _tan_wcs()
    # PA=0 (due north): long axis along +y; PA=90 (due east): along -x (cdelt1 < 0)
    axes = {}
    for pa in (0.0, 90.0):
        (xs, ys), _ = capsule_outline(w, RA0, DEC0, rate_degday=3.0, pa_deg=pa)
        c = np.c_[xs, ys]
        u = (c[2] - c[1]) / np.linalg.norm(c[2] - c[1])
        axes[pa] = u
    assert abs(axes[0.0] @ np.array([0.0, 1.0])) == pytest.approx(1.0, abs=1e-3)
    assert abs(axes[90.0] @ np.array([1.0, 0.0])) == pytest.approx(1.0, abs=1e-3)
    assert abs(axes[0.0] @ axes[90.0]) < 1e-3                   # perpendicular


def test_capsule_min_length_floor():
    w = _tan_wcs()
    (xs, ys), _ = capsule_outline(w, RA0, DEC0, rate_degday=0.0, pa_deg=0.0)
    c = np.c_[xs, ys]
    assert np.linalg.norm(c[2] - c[1]) == pytest.approx(2.0, rel=1e-6)  # L floor = 2 px


# --------------------------------------------------------------- match_det_row
def _dets():
    return pd.DataFrame(dict(
        visit=[101, 101, 102],
        detector=[5, 5, 5],
        ra=[RA0, RA0 + 30.0 / 3600.0, RA0],                     # 2nd det 30" east
        dec=[DEC0, DEC0, DEC0],
        score=[0.9, 0.8, 0.7]))


def test_match_det_row_picks_nearest_within_tol():
    ep = {"visit": 101, "detector": 5, "ra": RA0 + 0.4 / 3600.0, "dec": DEC0}
    row = match_det_row(_dets(), ep, tol_arcsec=1.0)
    assert row.score == 0.9                                     # nearest, not the 30"-away one


def test_match_det_row_raises_beyond_tol():
    ep = {"visit": 101, "detector": 5, "ra": RA0 + 10.0 / 3600.0, "dec": DEC0}
    with pytest.raises(ValueError, match="nearest det"):
        match_det_row(_dets(), ep, tol_arcsec=1.0)


def test_match_det_row_raises_on_empty_panel():
    ep = {"visit": 999, "detector": 5, "ra": RA0, "dec": DEC0}
    with pytest.raises(ValueError, match="no dets"):
        match_det_row(_dets(), ep)
