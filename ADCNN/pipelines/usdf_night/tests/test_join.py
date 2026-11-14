"""Finish stage: stamps (stack-row pixel measurements) and join (merge, flags, vetoes, writers)."""
import hashlib
import json
import types

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from ADCNN.inference import detect_night as dn
from ADCNN.pipelines.usdf_night import join as J, stamps
from ADCNN.pipelines.usdf_night import stack_tables as st_mod
from ADCNN.pipelines.usdf_night.mask_planes import COLUMNS as MASK_COLUMNS

V1, V2 = 2026111300017, 2026111300486
D1, D2 = 42, 129
MJD1, MJD2 = 61358.009781924935, 61358.267722652665
S = 0.2 / 3600.0                       # deg per px of the synthetic sky
BIG = 171207156887978006               # > 2**53: must survive every step exactly


def sky(x, y):
    return 150.0 + np.asarray(x, float) * S, 2.0 + np.asarray(y, float) * S


def net_row(visit, det, x, y, beta, length, score=0.6, mjd=None):
    b = np.radians(beta)
    hx, hy = 0.5 * length * np.cos(b), 0.5 * length * np.sin(b)
    ra, dec = sky(x, y)
    ra0, dec0 = sky(x - hx, y - hy)
    ra1, dec1 = sky(x + hx, y + hy)
    r = {c: np.nan for c in dn.DET_COLUMNS}
    r.update(mjd=mjd if mjd is not None else (MJD1 if visit == V1 else MJD2), ra=ra, dec=dec, band="r",
             obscode="X05", visit=visit, detector=det, x=x, y=y, score=score, length=length, len_db=length,
             mf_snr=5.0, ra0=ra0, dec0=dec0, ra1=ra1, dec1=dec1, beta=beta, nn_pmax=0.9, m_elong=1.5,
             m_dipole=0.1, m_neg_ratio=0.0, m_neg_blob=0.0, is_dipole=False)
    r.update({c: 0.0 for c in MASK_COLUMNS})
    return r


def stack_row(i, visit, det, x, y, seg=None, L=np.nan, pa=np.nan, flux=100.0, neg=0, in_det=0, in_apdb=0,
              dobj=0, ssobj=0):
    """seg = (x0, y0, x1, y1) or None (no trail fit)."""
    r = {c: np.nan for c in st_mod.COLUMNS}
    ra, dec = sky(x, y)
    r.update(diaSourceId=BIG + i, visit=visit, detector=det, band="r", mjd_tai=MJD1 if visit == V1 else MJD2,
             x=x, y=y, psf_ra=ra, psf_dec=dec, ra_err=1e-6, dec_err=2e-6, ra_dec_cov=1e-13,
             psf_flux_njy=flux, psf_flux_err_njy=10.0, is_negative=neg, rubin_is_dipole=0, trail_flag_edge=0,
             in_detector=in_det, in_apdb=in_apdb, reliability=0.9 if in_det else np.nan, glint_trail=0,
             dia_object_id=dobj, ss_object_id=ssobj)
    r.update({c: 0 for c in st_mod.COLUMNS if c.startswith("pixelFlags_")})
    if seg is not None:
        x0, y0, x1, y1 = seg
        tra, tdec = sky(0.5 * (x0 + x1), 0.5 * (y0 + y1))
        r.update(trail_x0=x0, trail_y0=y0, trail_x1=x1, trail_y1=y1, trail_ra=tra, trail_dec=tdec,
                 trail_length_px=np.hypot(x1 - x0, y1 - y0), trail_angle_img=np.arctan2(y1 - y0, x1 - x0),
                 trail_length_arcsec=L, trail_length_err_arcsec=0.1, trail_pa_deg=pa, trail_pa_err_deg=2.0,
                 trail_flux_njy=flux, trail_flux_err_njy=20.0)
    return r


def synthetic_night():
    """Five network trails, eight stack sources: see test_build_rows_and_hl for the expected outcome."""
    man = pd.DataFrame({"visit": [V1, V2], "detector": [D1, D2], "band": ["r", "i"], "day_obs": 20261113,
                        "mjd_tai": [MJD1, MJD2], "fits_path": ["p1.fits", "p2.fits"], "repo": "embargo",
                        "dataset_type": "future_difference_image", "run": "r"})
    net = pd.DataFrame([net_row(V1, D1, 50.0, 50.0, 0.0, 20.0),       # A: absorbs s1 (passes) + s2 (point)
                        net_row(V1, D1, 150.0, 150.0, 90.0, 30.0),    # B: absorbs s3 + s4, both pass (split)
                        net_row(V1, D1, 50.0, 150.0, 45.0, 25.0),     # C: no stack, network passes
                        net_row(V1, D1, 150.0, 50.0, 0.0, 5.0),       # D: no stack, 1" fails the HL rule
                        net_row(V2, D2, 10.0, 10.0, 0.0, 2.0)])       # E: panel 2, no stack
    net.insert(0, "detid", range(len(net)))
    stack = pd.DataFrame([
        stack_row(1, V1, D1, 52.0, 51.0, (45.0, 51.0, 55.0, 51.0), L=3.0, pa=-100.0, dobj=7, ssobj=9),
        stack_row(2, V1, D1, 58.0, 52.0, in_det=1),
        stack_row(3, V1, D1, 151.0, 142.0, (151.0, 140.0, 151.0, 145.0), L=2.0, pa=10.0, in_det=1, in_apdb=1),
        stack_row(4, V1, D1, 150.5, 157.0, (150.5, 155.0, 150.5, 160.0), L=2.0, pa=10.0, in_det=1),
        stack_row(5, V1, D1, 100.0, 100.0, (95.0, 100.0, 105.0, 100.0), L=2.0, pa=-30.0, in_det=1, in_apdb=1),
        stack_row(6, V1, D1, 50.0, 50.0, (45.0, 50.0, 55.0, 50.0), L=2.0, pa=90.0, neg=1),
        stack_row(7, V1, D1, 100.0, 20.0),
        stack_row(8, V2, D2, 50.0, 50.0, (48.0, 50.0, 52.0, 50.0), L=0.8, pa=90.0)])[st_mod.COLUMNS]
    for c in ["diaSourceId", "visit", "detector", "dia_object_id", "ss_object_id"]:
        stack[c] = stack[c].astype(np.int64)
    ra5, dec5 = sky(100.0, 100.0)
    refcat = pd.DataFrame({"ra": [ra5 + 1.0 / 3600], "dec": [dec5], "gmag": [19.0]})
    return man, net, stack, refcat


# ------------------------------------------------------------------ geometry

def seg(x0, y0, x1, y1):
    return np.array([[[x0, y0], [x1, y1]]], float)


def test_segment_distance_regular_cases():
    assert J.segment_distance(seg(0, 0, 10, 0), seg(0, 3, 10, 3))[0, 0] == pytest.approx(3.0)
    assert J.segment_distance(seg(0, 0, 10, 10), seg(0, 10, 10, 0))[0, 0] == 0.0          # crossing
    assert J.segment_distance(seg(0, 0, 10, 0), seg(5, 0, 5, 7))[0, 0] == 0.0             # T-junction
    assert J.segment_distance(seg(0, 0, 10, 0), seg(13, 4, 20, 4))[0, 0] == pytest.approx(5.0)


def test_segment_distance_degenerate_cases():
    # two distinct points, a point on the extension of a segment, collinear disjoint segments
    assert J.segment_distance(seg(0, 0, 0, 0), seg(30, 40, 30, 40))[0, 0] == pytest.approx(50.0)
    assert J.segment_distance(seg(100, 0, 100, 0), seg(0, 0, 10, 0))[0, 0] == pytest.approx(90.0)
    assert J.segment_distance(seg(0, 0, 10, 0), seg(20, 0, 30, 0))[0, 0] == pytest.approx(10.0)
    assert J.segment_distance(seg(5, 0, 5, 0), seg(0, 0, 10, 0))[0, 0] == 0.0             # point on segment
    assert J.segment_distance(seg(0, 0, 0, 0), seg(0, 0, 0, 0))[0, 0] == 0.0
    assert np.isnan(J.segment_distance(seg(np.nan, 0, 1, 0), seg(0, 0, 10, 0))[0, 0])
    assert J.segment_distance(np.zeros((0, 2, 2)), seg(0, 0, 1, 0)).shape == (0, 1)


def test_network_segments_endpoints_and_point_fallback():
    net = pd.DataFrame({"x": [10.0, 5.0], "y": [20.0, 5.0], "beta": [30.0, 10.0], "length": [8.0, np.nan]})
    s = J.network_segments(net)
    h = 4.0 * np.cos(np.radians(30.0))                       # half-length 4 px at 30 deg: (h, 2) about (10, 20)
    assert np.allclose(s[0], [[10.0 - h, 18.0], [10.0 + h, 22.0]])
    assert np.array_equal(s[1], [[5.0, 5.0], [5.0, 5.0]])


def test_stack_segments_naive_or_point():
    st = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "trail_x0": [0.0, np.nan], "trail_y0": [1.0, 0.0],
                       "trail_x1": [5.0, 1.0], "trail_y1": [6.0, 1.0]})
    s = J.stack_segments(st)
    assert np.array_equal(s[0], [[0.0, 1.0], [5.0, 6.0]])
    assert np.array_equal(s[1], [[2.0, 4.0], [2.0, 4.0]])


def test_sky_sep_and_bearing():
    d = 1.0 / 3600
    assert J.sky_sep_arcsec(10.0, 0.0, 10.0, d) == pytest.approx(1.0, rel=1e-9)
    assert J.sky_sep_arcsec(10.0, 60.0, 10.0 + 2 * d, 60.0) == pytest.approx(1.0, rel=1e-6)
    b = J.bearing_deg([10.0] * 4, [0.0] * 4, [10.0, 10.0 + d, 10.0 - d, 10.0], [d, 0.0, 0.0, -d])
    assert np.allclose(b, [0.0, 90.0, -90.0, 180.0])
    assert b[3] == 180.0                                                 # (-180, 180]: south is +180
    assert J.bearing_deg(0.0, 0.0, -0.0, -d) == 180.0                    # arctan2(-0.0, x<0) = -180
    assert J.sky_sep_arcsec(10.0, 5.0, 10.0, 5.0) == 0.0


def test_fold_pa():
    pa = np.array([-180.0, -90.0, 0.0, 89.9, 90.0, 180.0, 269.9, 270.0, np.nan])
    f = J.fold_pa(pa)
    assert np.allclose(f[:-1], [180.0, 90.0, 180.0, 269.9, 90.0, 180.0, 269.9, 90.0])
    assert np.isnan(f[-1]) and ((f[:-1] >= 90) & (f[:-1] < 270)).all()


def test_mag():
    m = J.mag([3631e9, 1.0, 0.0, -5.0, np.nan])
    assert m[0] == pytest.approx(31.4 - 2.5 * np.log10(3631e9))
    assert m[1] == 31.4 and np.isnan(m[2:]).all()


def test_passes_hl_boundaries():
    L = [1.25, 1.2500001, 12.4999, 12.5, np.nan, 5.0, 5.0, 5.0, 5.0]
    neg = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    flux = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, np.nan, -1.0]
    assert J.passes_hl(L, neg, flux).tolist() == [False, True, True, False, False, False, False, False, False]
    assert J.passes_hl(L, neg).tolist() == [False, True, True, False, False, False, True, True, True]


# ------------------------------------------------------------------ merge

def test_network_ids_formula_and_order():
    visit = np.array([V1, V1, V1, V2, V1], np.int64)
    det = np.array([42, 42, 42, 7, 43])
    x = np.array([5.0, 1.0, 5.0, 0.0, 0.0])
    y = np.array([9.0, 9.0, 2.0, 0.0, 0.0])
    score = np.array([0.5, 0.9, 0.7, 0.5, 0.5])
    ids = J.network_ids(visit, det, x, y, score)
    base = (V1 % 10**11) * 10**8 + 42 * 10**5
    assert ids.dtype == np.int64
    assert ids.tolist() == [base + 2, base + 1, base + 0, (V2 % 10**11) * 10**8 + 7 * 10**5,
                            (V1 % 10**11) * 10**8 + 43 * 10**5]
    assert ids[0] == 2611130001704200002 and len(str(ids[0])) == 19


def test_network_ids_stable_on_ties_and_score():
    ids = J.network_ids([V1] * 4, [1] * 4, [3.0] * 4, [3.0] * 4, [0.5, 0.4, 0.5, 0.4])
    idx = ids % 10**5
    assert idx.tolist() == [2, 0, 3, 1]                  # score ascending, ties keep input order


def test_network_ids_index_limit():
    n = 100001
    with pytest.raises(AssertionError):
        J.network_ids(np.full(n, V1), np.zeros(n, int), np.arange(n, dtype=float), np.zeros(n), np.zeros(n))
    with pytest.raises(AssertionError):
        J.network_ids([V1], [1000], [0.0], [0.0], [0.0])


def test_match_nearest_negative_panel_and_tolerance():
    net = pd.DataFrame({"visit": [V1, V1], "detector": [D1, D1], "x": [50.0, 60.0], "y": [50.0, 50.0],
                        "beta": [90.0, 90.0], "length": [10.0, 10.0]})
    st = pd.DataFrame({"visit": [V1, V1, V2, V1, V1, V1], "detector": [D1, D1, D1, D1, D1, D1],
                       "x": [58.0, 50.0, 50.0, 40.0, 39.5, 61.0], "y": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
                       "is_negative": [0, 1, 0, 0, 0, 0],
                       "trail_x0": np.nan, "trail_y0": np.nan, "trail_x1": np.nan, "trail_y1": np.nan})
    trail, sep = J.match(net, st)
    # nearer of two trails / negative never matches / other panel / exactly 10 px / 10.5 px / tie-free
    assert trail.tolist() == [1, -1, -1, 0, -1, 1]
    assert sep[0] == pytest.approx(2.0) and sep[3] == pytest.approx(10.0) and np.isnan(sep[1])
    t0, s0 = J.match(net.iloc[:0], st)
    assert (t0 == -1).all() and np.isnan(s0).all()


def test_match_zero_length_trail_does_not_absorb_every_point():
    net = pd.DataFrame({"visit": [V1], "detector": [D1], "x": [10.0], "y": [10.0], "beta": [0.0], "length": [0.0]})
    st = pd.DataFrame({"visit": [V1, V1], "detector": [D1, D1], "x": [15.0, 500.0], "y": [10.0, 900.0],
                       "is_negative": [0, 0], "trail_x0": np.nan, "trail_y0": np.nan, "trail_x1": np.nan,
                       "trail_y1": np.nan})
    assert J.match(net, st)[0].tolist() == [0, -1]


def test_absorb_representative_ladder():
    # trail 0: pass+det (near) loses to pass+apdb (far); trail 1: no source passes -> apdb > det > nearest
    # trail 2: only non-flagged, non-passing -> nearest; trail 3: nothing absorbed
    trail = np.array([0, 0, 1, 1, 1, 2, 2, -1])
    sep = np.array([1.0, 5.0, 0.5, 4.0, 3.0, 2.0, 1.0, np.nan])
    ids = np.arange(8, dtype=np.int64) + BIG
    passes = np.array([1, 1, 0, 0, 0, 0, 0, 1], bool)
    in_det = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    in_apdb = np.array([0, 1, 0, 0, 1, 0, 0, 1])
    rep, n_stack, n_pass, reliable, apdb, joined = J.absorb(trail, sep, ids, passes, in_det, in_apdb, 4)
    assert rep.tolist() == [1, 4, 6, -1]
    assert n_stack.tolist() == [2, 3, 2, 0] and n_pass.tolist() == [2, 0, 0, 0]
    assert reliable.tolist() == [True, True, False, False] and apdb.tolist() == [True, True, False, False]
    assert joined.tolist() == [f"{BIG};{BIG + 1}", f"{BIG + 2};{BIG + 3};{BIG + 4}", f"{BIG + 5};{BIG + 6}", "0"]


def test_absorb_ladder_pass_beats_flags():
    trail = np.array([0, 0, 0])
    rep = J.absorb(trail, np.array([3.0, 1.0, 2.0]), np.arange(3, dtype=np.int64),
                   np.array([1, 0, 0], bool), np.array([0, 1, 1]), np.array([0, 1, 0]), 1)[0]
    assert rep.tolist() == [0]                  # passes the HL rule > in APDB > reliable


def test_take():
    v = J._take(np.array([1.0, 2.0, 3.0]), np.array([2, -1, 0]))
    assert v[0] == 3.0 and np.isnan(v[1]) and v[2] == 1.0


def test_vetoes():
    d = 1.0 / 3600
    ra = np.array([10.0, 20.0, 30.0, 40.0, 50.0, np.nan])
    dec = np.zeros(6)
    refcat = pd.DataFrame({"ra": [10.0 + 1 * d, 20.0 + 2 * d, 30.0 + 5 * d, 40.0 + 20 * d, 50.0 + 1 * d, 50.0 + 2.5 * d],
                           "gmag": [19.5, 20.5, 18.0, 15.0, 20.5, 19.9], "dec": 0.0})
    v = J.vetoes(ra, dec, refcat)
    assert np.allclose(v["refcat_sep_arcsec"][:3], [1.0, 2.0, 5.0], rtol=1e-6)
    assert np.isnan(v["refcat_sep_arcsec"][3]) and np.isnan(v["refcat_gmag"][3])
    assert v["refcat_gmag"][:3].tolist() == [19.5, 20.5, 18.0]
    assert v["ring_flag"][:5].tolist() == [1, 1, 0, 0, 1]
    # static uses its own G < 20 tree: row 4's nearest star is G 20.5 at 1", a G 19.9 star sits at 2.5"
    assert v["static_flag"][:5].tolist() == [1, 0, 0, 0, 1]
    assert np.isnan([v[k][5] for k in v]).all()
    e = J.vetoes(ra[:2], dec[:2], refcat.iloc[:0])
    assert e["ring_flag"].tolist() == [0, 0] and np.isnan(e["refcat_sep_arcsec"]).all()


def test_build_rows_and_hl():
    man, net, stack, refcat = synthetic_night()
    det, hl, counts = J.build(man, net, stack, stamps.empty(stack), refcat)
    pix = [c for c in stack.columns if c.startswith("pixelFlags_")]
    assert list(det.columns[:-2]) == J.detection_columns(pix)
    assert len(det) == 9 and counts["network_rows"] == 5 and counts["stack_only_rows"] == 4
    ids = J.network_ids(net.visit, net.detector, net.x, net.y, net.score)
    by = det.set_index("det_id")
    A, B, C, Dn, E = (by.loc[i] for i in ids)
    assert (A.n_stack, A.stack_dia_source_id, A.dia_source_ids) == (2, BIG + 1, f"{BIG + 1};{BIG + 2}")
    # s2 is in dia_source_detector but not in the APDB: reliable comes from in_detector alone
    assert (A.adcnn_detected, A.stack_detected, A.stack_reliable, A.stack_in_apdb) == (1, 1, 1, 0)
    assert A.match_sep_px == pytest.approx(1.0) and A.is_negative == 0 and A.score == 0.6
    assert A.length_px == 20.0 and A.beta_img_deg == 0.0 and A.pa_sky_deg == pytest.approx(90.0, abs=1e-4)
    assert A.length_arcsec == pytest.approx(20 * 0.2 * np.cos(np.radians(2.0)), rel=1e-3)
    assert (B.n_stack, B.stack_dia_source_id, B.stack_reliable, B.stack_in_apdb) == (2, BIG + 3, 1, 1)
    assert B.dia_source_ids == f"{BIG + 3};{BIG + 4}"
    assert (C.n_stack, C.stack_dia_source_id, C.dia_source_ids, C.stack_detected) == (0, 0, "0", 0)
    assert pd.isna(C.is_negative) and pd.isna(C.reliability) and pd.isna(C.pixelFlags_bad)
    s5, s6, s7, s8 = (by.loc[BIG + i] for i in (5, 6, 7, 8))
    assert (s5.adcnn_detected, s5.stack_detected, s5.stack_reliable, s5.stack_in_apdb, s5.n_stack) == (0, 1, 1, 1, 1)
    assert s5.dia_source_ids == str(BIG + 5) and s5.stack_dia_source_id == BIG + 5
    assert s5.ring_flag == 1 and s5.static_flag == 1 and s5.refcat_gmag == 19.0
    assert s6.is_negative == 1 and s6.n_stack == 1                       # negative on trail A: own row
    assert s7.ra == stack.set_index("diaSourceId").loc[BIG + 7, "psf_ra"]   # no trail fit: PSF position
    assert counts["stack_only_psf_position"] == 1 and counts["stack_only_negative"] == 1
    assert s5.beta_img_deg == 0.0 and s5.pa_sky_deg == -30.0 and s5.length_arcsec == 2.0   # raw PA, unfolded
    assert counts["split_trails"] == 1 and counts["split_trails_network_fails_hl"] == 0
    assert counts["hl_rows"] == 4 and counts["hl_trail_stack_values"] == 1 and counts["hl_stack_only"] == 1
    assert counts["hl_split_network_values"] == 1 and counts["hl_trail_network_values"] == 1
    for c in ["det_id", "visit", "detector", "stack_dia_source_id", "n_stack"]:
        assert det[c].dtype == np.int64
    assert det.mjd_tai.is_monotonic_increasing
    # hl: A -> s1's stack values, B -> network values (split), C -> network values, s5 -> own values
    assert list(hl.columns) == J.HL_COLUMNS
    h = hl.set_index("#stringID")
    assert sorted(h.index) == sorted([BIG + 1, BIG + 5, ids[1], ids[2]])
    a = h.loc[BIG + 1]
    s1 = stack.set_index("diaSourceId").loc[BIG + 1]
    assert (a.RA, a.DEC, a.traillength, a.trailpa) == (s1.trail_ra, s1.trail_dec, 3.0, 260.0)
    assert a.RA != s1.psf_ra                                             # trail centre, not the PSF position
    assert a.Mag == J.mag(100.0) and a.sigmag == pytest.approx(1.0857 * 0.2)
    assert (a.det_id, a.dia_source_ids, a.n_stack, a.dia_object_id, a.ss_object_id) == (ids[0], A.dia_source_ids, 2, 7, 9)
    assert (a.MJD, a.Obscode, a.band, a.RA_err, a.trailpa_err, a.traillength_err) == (MJD1, "X05", "r", 1e-6, 2.0, 0.1)
    assert (a.stack_reliable, a.stack_in_apdb) == (1, 0)
    assert h.loc[BIG + 5].trailpa == 150.0                              # -30 folded into [90, 270)
    b = h.loc[ids[1]]
    assert b.dia_source_ids == f"{BIG + 3};{BIG + 4}" and np.isnan([b.Mag, b.RA_err, b.sigmag, b.traillength_err]).all()
    assert (b.RA, b.DEC, b.dia_object_id) == (B.ra, B.dec, 0) and b.trailpa == pytest.approx(180.0, abs=1e-4)
    assert b.traillength == B.length_arcsec and b.reliability == 0.9
    assert hl.MJD.is_monotonic_increasing


def test_build_rejects_mjd_mismatch_and_foreign_panel():
    man, net, stack, refcat = synthetic_night()
    bad = net.copy()
    bad.loc[0, "mjd"] += 2.0 / 86400000
    with pytest.raises(AssertionError):
        J.build(man, bad, stack, stamps.empty(stack), refcat)
    bad = net.copy()
    bad.loc[0, "detector"] = 7
    with pytest.raises(AssertionError):
        J.build(man, bad, stack, stamps.empty(stack), refcat)


def test_build_stack_only_night():
    man, net, stack, refcat = synthetic_night()
    det, hl, counts = J.build(man, net.iloc[:0], stack, stamps.empty(stack), refcat)
    assert len(det) == 8 and (det.adcnn_detected == 0).all() and counts["split_trails"] == 0
    assert sorted(hl["#stringID"]) == [BIG + 1, BIG + 3, BIG + 4, BIG + 5]


def test_split_trail_outside_hl_window_keeps_network_values():
    """A trail Rubin split into two passing sources is one hl row with the network values, also when the
    network length (16") is outside the HL window: dropping the row would drop both APDB sources."""
    man, _, _, refcat = synthetic_night()
    net = pd.DataFrame([net_row(V1, D1, 100.0, 100.0, 0.0, 80.0)])
    net.insert(0, "detid", 0)
    stack = pd.DataFrame([
        stack_row(1, V1, D1, 72.5, 100.0, (70.0, 100.0, 75.0, 100.0), L=2.0, pa=90.0, in_det=1, in_apdb=1),
        stack_row(2, V1, D1, 127.5, 100.0, (125.0, 100.0, 130.0, 100.0), L=2.0, pa=90.0, in_det=1, in_apdb=1),
    ])[st_mod.COLUMNS]
    for c in ["diaSourceId", "visit", "detector", "dia_object_id", "ss_object_id"]:
        stack[c] = stack[c].astype(np.int64)
    det, hl, counts = J.build(man, net, stack, stamps.empty(stack), refcat)
    assert len(det) == 1 and det.n_stack[0] == 2 and det.length_arcsec[0] > J.HL_MAX_ARCSEC
    assert counts["split_trails"] == counts["split_trails_network_fails_hl"] == counts["hl_rows"] == 1
    h = hl.iloc[0]
    assert h["#stringID"] == det.det_id[0] and h.dia_source_ids == f"{BIG + 1};{BIG + 2}"
    assert h.traillength == det.length_arcsec[0] and np.isnan(h.Mag) and h.dia_object_id == 0


def test_hl_rows_assertions():
    man, net, stack, refcat = synthetic_night()
    det, hl, _ = J.build(man, net, stack, stamps.empty(stack), refcat)
    st = stack.reset_index(drop=True).merge(man[["visit", "detector", "band", "mjd_tai"]], on=["visit", "detector"],
                                            suffixes=("_stack", ""))
    st["trail_mag"] = J.mag(st.trail_flux_njy)
    bad = det.copy()
    bad.loc[bad["_hl"] == 2, "mjd_tai"] = np.nan
    with pytest.raises(AssertionError):
        J.hl_rows(bad, st)


def test_column_lists():
    pix = ["pixelFlags_bad", "pixelFlags_cr"]
    cols = J.detection_columns(pix)
    spec = ("det_id visit detector band mjd_tai adcnn_detected stack_detected stack_reliable stack_in_apdb "
            "is_negative ra dec x y length_px length_arcsec beta_img_deg pa_sky_deg ra0 dec0 ra1 dec1 score "
            "mf_snr nn_pmax stack_dia_source_id n_stack match_sep_px dia_source_ids reliability psf_ra psf_dec "
            "ra_err dec_err ra_dec_cov psf_flux_njy psf_flux_err_njy psf_mag trail_flux_njy trail_flux_err_njy "
            "trail_mag trail_length_err_arcsec trail_pa_err_deg rubin_is_dipole glint_trail trail_flag_edge "
            "pixelFlags_bad pixelFlags_cr ring_flag refcat_sep_arcsec refcat_gmag static_flag is_dipole m_elong "
            "m_dipole m_neg_ratio m_neg_blob").split() + MASK_COLUMNS
    assert cols == spec
    assert set(J.nullable_flags(pix)) <= set(cols)
    assert ",".join(J.HL_COLUMNS) == (
        "#stringID,MJD,RA,DEC,Mag,band,Obscode,trailpa,traillength,RA_err,DEC_err,RA_DEC_cov,sigmag,trailpa_err,"
        "traillength_err,visit,detector,det_id,dia_source_ids,adcnn_detected,stack_detected,stack_reliable,"
        "stack_in_apdb,score,reliability,ring_flag,static_flag,is_dipole,rubin_is_dipole,art_frac,n_stack,"
        "dia_object_id,ss_object_id")


# ------------------------------------------------------------------ writers

def test_writers_round_trip_and_text(tmp_path):
    man, net, stack, refcat = synthetic_night()
    det, hl, _ = J.build(man, net, stack, stamps.empty(stack), refcat)
    J.write_csv(det.drop(columns=["_hl", "_src"]), tmp_path / "detections.csv")
    J.write_csv(hl, tmp_path / "hl_dets.csv")
    J.check_hl_text(tmp_path / "hl_dets.csv")
    J.write_text(J.COLFORMAT, tmp_path / "colformat.txt")
    assert (tmp_path / "colformat.txt").read_bytes() == (
        b"IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\nTRAILPACOL 8\n"
        b"TRAILLENCOL 9\nSIGMAGCOL 13\nSIGALONGCOL 15\n")
    text = (tmp_path / "hl_dets.csv").read_text()
    assert text.split("\n")[0] == ",".join(J.HL_COLUMNS) and text.endswith("\n") and ",," not in text
    assert ",nan," in text and f",{BIG + 3};{BIG + 4}," in text
    d2, h2 = pd.read_csv(tmp_path / "detections.csv"), pd.read_csv(tmp_path / "hl_dets.csv")
    for c in ["det_id", "visit", "detector", "stack_dia_source_id", "n_stack"]:
        assert d2[c].dtype == np.int64 and (d2[c].to_numpy() == det[c].to_numpy()).all()
    for c in ["#stringID", "visit", "detector", "det_id", "n_stack", "dia_object_id", "ss_object_id"]:
        assert h2[c].dtype == np.int64 and (h2[c].to_numpy() == hl[c].to_numpy()).all()
    assert set(h2["#stringID"]) >= {BIG + 1, BIG + 5}
    for f, ref in (("detections.csv", det), ("hl_dets.csv", hl)):
        s = pd.read_csv(tmp_path / f, dtype={"dia_source_ids": str})["dia_source_ids"]
        assert s.tolist() == ref["dia_source_ids"].tolist()
    assert d2.ring_flag.dropna().isin([0, 1]).all() and "1.0" not in text.split("\n")[1].split(",")[25:29]
    assert not list(tmp_path.glob("*.tmp"))


def test_write_csv_check_runs_before_rename(tmp_path):
    def reject(p):
        assert p.name == "hl_dets.csv.tmp"
        raise AssertionError("bad")
    with pytest.raises(AssertionError):
        J.write_csv(pd.DataFrame({"a": [1]}), tmp_path / "hl_dets.csv", check=reject)
    assert not (tmp_path / "hl_dets.csv").exists()
    J.write_csv(pd.DataFrame({"a": [1, np.nan]}), tmp_path / "x.csv", check=lambda p: None)
    assert (tmp_path / "x.csv").read_text() == "a\n1.0\nnan\n"


def test_check_hl_text_rejects(tmp_path):
    p = tmp_path / "hl.csv"
    p.write_text(",".join(J.HL_COLUMNS) + "\n" + "1,,2\n")
    with pytest.raises(AssertionError):
        J.check_hl_text(p)
    p.write_text(",".join(J.HL_COLUMNS) + "\n1,2")
    with pytest.raises(AssertionError):
        J.check_hl_text(p)
    p.write_text(",".join(J.HL_COLUMNS[:-1]) + "\n")
    with pytest.raises(AssertionError):
        J.check_hl_text(p)


def test_md5_commit_job_record(tmp_path, monkeypatch):
    (tmp_path / "f").write_bytes(b"abc")
    assert J.md5(tmp_path / "f") == hashlib.md5(b"abc").hexdigest()
    assert J.commit(tmp_path) == "uncommitted"
    out = "36468681|COMPLETED|01:11:00|billing=36,cpu=36,gres/gpu=5,mem=350G,node=1|sdfada006\n"   # S3DF sacct
    monkeypatch.setattr(J.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out, returncode=0))
    assert J.job_record(36468681) == {"job_id": "36468681", "state": "COMPLETED", "elapsed": "01:11:00",
                                      "alloc_tres": "billing=36,cpu=36,gres/gpu=5,mem=350G,node=1",
                                      "nodes": "sdfada006"}
    monkeypatch.setattr(J.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="", returncode=1))
    assert J.job_record(5) == {"job_id": "5"}

    def boom(*a, **k):
        raise FileNotFoundError("sacct")
    monkeypatch.setattr(J.subprocess, "run", boom)
    assert J.job_record(6) == {"job_id": "6"}


def test_stage_summary():
    assert J.stage_summary({}) == (None, {})
    gpu, wall = J.stage_summary({"detect": [{"seconds": 10, "gpus": ["NVIDIA H100", "NVIDIA L40S"]},
                                            {"seconds": None, "gpus": []}], "prep": [{"seconds": None}]})
    assert gpu == {"type": "NVIDIA H100,NVIDIA L40S", "count": 2} and wall == {"detect": 10}


def test_covered_panels_and_unreadable(tmp_path):
    man, *_ = synthetic_night()
    assert J.covered_panels(tmp_path, man) == {(V1, D1), (V2, D2)}
    (tmp_path / "_shard_adcnn_dets_0.csv.done").write_text(f"{V1},{D1}\n2026111399999,1\n")
    assert J.covered_panels(tmp_path, man) == {(V1, D1)}
    (tmp_path / "adcnn_dets.csv.unreadable").write_text(
        f"{V2},{D2},OSError: first\n{V1},{D1},OSError: stale\n{V2},{D2},\"TimeoutError: a, b\"\n")
    assert J.unreadable(tmp_path, {(V2, D2)}) == [[V2, D2, "TimeoutError: a, b"]]
    assert J.unreadable(tmp_path / "missing", {(V2, D2)}) == []


# ------------------------------------------------------------------ stamps (synthetic diffim FITS)

def write_diffim(path, shape=(120, 160)):
    """Diffim FITS as read_fits_panel expects: HDU1 image + TAN WCS, MASK HDU with MSKN/MSKM cards."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [80.5, 60.5]
    w.wcs.cdelt = [-S, S]
    img = np.random.default_rng(1).normal(0, 1, shape).astype(np.float32)
    img[40, 30:51] += 50.0                               # a horizontal trail y=40, x=30..50
    mask = np.zeros(shape, np.int32)
    mask[40, 30:51] = 2                                  # SAT along that trail only
    mask[10:13, 10:13] = 8                               # CR box around (11, 11)
    mask[99:102, 99:102] = 1                             # BAD box around (100, 100) only
    mask[50:71, 120] = 2                                 # SAT along a vertical trail x=120, y=50..70
    p = fits.PrimaryHDU()
    p.header["DATE-AVG"] = "2026-11-13T06:25:00.000"
    mh = fits.Header()
    for i, (n, v) in enumerate([("BAD", 1), ("SATURATED", 2), ("COSMIC_RAY", 8)]):
        mh[f"MSKN{i:04d}"], mh[f"MSKM{i:04d}"] = n, v
    fits.HDUList([p, fits.ImageHDU(img, header=w.to_header(), name="IMAGE"),
                  fits.ImageHDU(mask, header=mh, name="MASK")]).writeto(path)
    return img, w, mask


def stamp_sources():
    return pd.DataFrame({"diaSourceId": [BIG, BIG + 1, BIG + 2, BIG + 3],
                         "x": [43.0, 11.0, 100.0, 123.0], "y": [44.0, 11.0, 100.0, 60.0],   # centroids off the segments
                         "trail_x0": [30.0, np.nan, 104.0, 120.0], "trail_y0": [40.0, np.nan, 100.0, 50.0],
                         "trail_x1": [50.0, np.nan, 104.4, 120.0], "trail_y1": [40.0, np.nan, 100.0, 70.0],
                         "trail_length_px": [20.0, np.nan, 0.4, 20.0],
                         "trail_angle_img": [0.0, np.nan, 0.0, np.pi / 2]})


def test_stamps_empty():
    e = stamps.empty(stamp_sources())
    assert list(e.columns) == stamps.COLUMNS and e.diaSourceId.dtype == np.int64
    assert e.drop(columns="diaSourceId").isna().all().all()


def test_stamps_measure(tmp_path):
    from ADCNN.inference.catalog import _attach_dipole_morphology
    img, w, mask, values = None, None, None, None
    write_diffim(tmp_path / "d.fits")
    img, w, _mjd, mask, values = dn.read_fits_panel(str(tmp_path / "d.fits"))
    src = stamp_sources()
    m = stamps.measure(img, w, mask, values, src)
    assert m.m_SAT[0] == 1.0                     # sampled along the Naive segment (midpoint 40,40), not at (43,44)
    assert m.m_SAT[3] == 1.0                     # vertical trail: Naive_angle (rad) -> beta in degrees
    assert m.m_CR[1] == 1.0 and m.m_SAT[1] == 0.0  # no segment: 3x3 box at (x, y)
    assert m.m_BAD[2] == 1.0 and m.m_CR[2] == 0.0      # 0.4 px segment: 3x3 box at (x, y), not at its midpoint
    assert np.isnan(m.m_STREAK).all() and (m.art_frac == m[["m_SAT", "m_CR", "m_BAD"]].max(1)).all()
    ref = pd.DataFrame({"x_centroid": src.x, "y_centroid": src.y})
    _attach_dipole_morphology(ref, img)
    for c in stamps.MORPH_COLUMNS:
        assert np.array_equal(m[c].to_numpy(), ref[c].to_numpy(np.float64))
    e0 = w.all_pix2world([[30.0, 40.0]], 0)[0]
    assert (m.ra0[0], m.dec0[0]) == (e0[0], e0[1]) and np.isnan(m.loc[1, stamps.ENDPOINT_COLUMNS]).all()
    assert np.isfinite(m.loc[2, stamps.ENDPOINT_COLUMNS].to_numpy(float)).all()
    assert np.isnan(stamps.measure(img, w, None, None, src)[MASK_COLUMNS]).all().all()


def test_stamps_panel_and_run(tmp_path):
    write_diffim(tmp_path / "d.fits")
    src = stamp_sources()
    ok, err = stamps._panel((str(tmp_path / "d.fits"), src))
    assert err is None and ok.m_SAT[0] == 1.0
    bad, err = stamps._panel((str(tmp_path / "missing.fits"), src))
    assert err.startswith("FileNotFoundError") and bad.drop(columns="diaSourceId").isna().all().all()
    stack = pd.concat([src.assign(visit=V1, detector=D1), src.assign(visit=V2, detector=D2,
                                                                       diaSourceId=src.diaSourceId + 10)])
    man = pd.DataFrame({"visit": [V1, V2], "detector": [D1, D2],
                        "fits_path": [str(tmp_path / "d.fits"), str(tmp_path / "missing.fits")]})
    m, unread = stamps.run(stack, man, 2)
    assert len(m) == 8 and m.diaSourceId.tolist() == stack.diaSourceId.tolist()
    assert unread == [[V2, D2, unread[0][2]]] and unread[0][2].startswith("FileNotFoundError")
    assert np.isfinite(m.m_SAT[:4]).all() and m.iloc[4:].drop(columns="diaSourceId").isna().all().all()
    assert stamps.run(stack.iloc[:0], man, 2)[0].empty


# ------------------------------------------------------------------ end to end

def write_night(N, done_only=None):
    man, net, stack, refcat = synthetic_night()
    work = N / "work"
    work.mkdir(parents=True)
    for i, p in enumerate(["p1.fits", "p2.fits"]):
        write_diffim(work / p)
        man.loc[i, "fits_path"] = str(work / p)
    man.to_csv(work / "manifest.csv", index=False)
    net.to_csv(work / "adcnn_dets.csv", index=False)
    stack.to_parquet(work / "stack_sources.parquet", index=False)
    refcat.to_parquet(work / "refcat.parquet", index=False)
    (work / "prep_info.json").write_text(json.dumps({
        "manifest": {"repo": "embargo", "collection": "LSSTCam/runs/prompt-20261113",
                     "dataset_type": "future_difference_image", "runs": ["r"], "lsst_weekly": "w_2026_46"},
        "stack_tables": {"panels_without_unfiltered": [], "validation_failed_columns": []},
        "refcat": {"refcat": "the_monster_20250219"}}))
    if done_only:
        net[(net.visit == done_only[0]) & (net.detector == done_only[1])].to_csv(work / "adcnn_dets.csv", index=False)
        (work / "_shard_adcnn_dets_0.csv.done").write_text(f"{done_only[0]},{done_only[1]}\n")
        (work / "adcnn_dets.csv.unreadable").write_text(f"{V2},{D2},OSError: gone\n")
    return man, net, stack, refcat


def test_main_end_to_end(tmp_path):
    N = tmp_path / "20261113"
    man, net, stack, refcat = write_night(N)
    # stages as status.finish_step writes them before join: a later detect attempt without nvidia-smi names
    # and the running finish job (no seconds yet)
    stages = {"prep": [{"stage": "prep", "job_id": "41", "rc": 0, "seconds": 600}],
              "detect": [{"stage": "detect", "job_id": "42", "rc": 3, "seconds": 5400, "gpus": ["NVIDIA L40S"] * 4},
                         {"stage": "detect", "job_id": "44", "rc": 0, "seconds": 300, "gpus": []}],
              "finish": [{"stage": "finish", "job_id": "43", "rc": 0, "seconds": 20},
                         {"stage": "finish", "job_id": "45", "rc": None, "seconds": None}]}
    (N / "state.json").write_text(json.dumps({"night": "20261113", "floor": 0.43, "user": "someone", "jobs": {},
                                              "stages": stages}))
    assert J.main([str(N), "--workers", "2"]) == 0
    for f in ["detections.csv", "hl_dets.csv", "colformat.txt", "run_info.json"]:
        assert (N / f).exists()
    assert (N / "work").exists()                          # the driver removes work/, not join
    info = json.loads((N / "run_info.json").read_text())
    assert info["status"] == "complete" and info["panels_expected"] == info["panels_done"] == 2
    assert (info["floor"], info["user"], info["mf_k"], info["lsst_weekly"]) == (0.43, "someone", 0.8956, "w_2026_46")
    assert info["models"]["seg"]["md5"] == "147cef8bb4e872b3fa0560a264e4748e"
    assert info["models"]["cnn"]["md5"] == "0ba43c29fc1c76c63d992b871c83474b"
    assert info["counts"]["hl_rows"] == 4 and info["counts"]["split_trails"] == 1
    assert info["gpu"] == {"type": "NVIDIA L40S", "count": 4}
    w = info["wall_seconds"]
    assert (w["prep"], w["detect"], w["finish"]) == (600, 5700, 20) and w["join"] > 0
    assert info["stack_validation_failed_columns"] == []
    det = pd.read_csv(N / "detections.csv")
    s5 = det[det.det_id == BIG + 5].iloc[0]
    assert np.isfinite([s5.m_elong, s5.m_SAT, s5.ra0]).all()   # stamps values reach stack-only rows
    a = det[det.det_id == J.network_ids([V1], [D1], [50.0], [50.0], [0.6])[0]].iloc[0]
    assert a.m_elong == 1.5                                     # matched rows keep the network's morphology


def test_main_partial_night(tmp_path):
    N = tmp_path / "20261113"
    write_night(N, done_only=(V1, D1))
    assert J.main([str(N), "--workers", "1"]) == 0
    info = json.loads((N / "run_info.json").read_text())
    assert info["status"] == "partial" and info["panels_missing"] == [[V2, D2]]
    assert info["panels_unreadable"] == [[V2, D2, "OSError: gone"]] and info["panels_done"] == 1
    det = pd.read_csv(N / "detections.csv")
    assert set(det.visit) == {V1} and not (det.det_id == BIG + 8).any()


def test_load_header_only_network_and_not_done_rows(tmp_path):
    N = tmp_path / "20261113"
    write_night(N)
    pd.DataFrame(columns=["detid"] + dn.DET_COLUMNS).to_csv(N / "work" / "adcnn_dets.csv", index=False)
    man, net, stack, refcat, prep, covered, man_all = J.load(N)
    assert len(net) == 0 and net.x.dtype == np.float64 and net.visit.dtype == np.int64
    det, hl, counts = J.build(man, net, stack, stamps.empty(stack), refcat)
    assert counts["network_rows"] == 0 and len(det) == 8
    N2 = tmp_path / "b" / "20261113"
    write_night(N2, done_only=(V1, D1))
    _, net_all, *_ = synthetic_night()
    net_all.to_csv(N2 / "work" / "adcnn_dets.csv", index=False)       # rows of a panel that is not done
    with pytest.raises(AssertionError):
        J.load(N2)
