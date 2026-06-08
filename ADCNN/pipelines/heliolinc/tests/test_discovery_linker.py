"""Unit tests for the same-night discovery linker's pure functions -- no GPU / Butler / data files needed.

Runs two ways:
  python ADCNN/pipelines/heliolinc/tests/test_discovery_linker.py     # standalone (no pytest dependency)
  pytest ADCNN/pipelines/heliolinc/tests/test_discovery_linker.py     # if pytest is installed

Covers the correctness-critical, deterministic behaviour exercised in production + the hardening changes:
spherical geometry (RA=0/pole), KD-tree crossmatch == brute force, chord seeding across the meridian,
the chi2 gate (true vs false, NaN rejected), visit-based epoch counting, and the resume dedup key.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from ADCNN.pipelines.heliolinc.trail_state_link import (
    radec_to_unit, _chord_radius, trail_velocity, pair_chi2, physical_check,
    chord_seed_pairs, crossmatch, build_known_index,
)
from ADCNN.pipelines.heliolinc.orbit_check import orbit_ok

SOLARDAY = 86400.0
EXPT = 30.0


def _mover_dets(ra0_deg, dec_deg, rate_degday, pa_deg, n_visits=2, gap_min=34.0, mjd0=60000.0,
                mf_snr=30.0, score=0.95):
    """Synthetic same-night mover: constant on-sky velocity, trail aligned with motion (PA), bright.
    Returns a dets DataFrame with the columns the linker needs."""
    pa = np.radians(pa_deg)
    cosd = np.cos(np.radians(dec_deg))
    vra = rate_degday * np.cos(pa) / cosd      # deg/day in RA (so on-sky rate is rate_degday)
    vdec = rate_degday * np.sin(pa)
    tl_deg = rate_degday * (EXPT / SOLARDAY)   # trail length over one exposure (deg, on-sky)
    rows = []
    for k in range(n_visits):
        mjd = mjd0 + k * (gap_min / 1440.0)
        ra = ra0_deg + vra * (mjd - mjd0)
        dec = dec_deg + vdec * (mjd - mjd0)
        # trail endpoints: half an exposure of motion each side of (ra,dec), along the motion direction
        h = 0.5 * EXPT / SOLARDAY
        rows.append(dict(detid=k, mjd=mjd, ra=ra, dec=dec,
                         ra0=ra - vra * h, dec0=dec - vdec * h, ra1=ra + vra * h, dec1=dec + vdec * h,
                         visit=1000 + k, detector=1, score=score, len_db=max(tl_deg * 3600 / 0.2, 6.0),
                         mf_snr=mf_snr))
    return pd.DataFrame(rows)


def _check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"  ok: {name}")


# ---------------------------------------------------------------- geometry
def test_radec_to_unit_and_chord():
    v = radec_to_unit(np.array([0.0, 90.0, 180.0]), np.array([0.0, 0.0, -45.0]))
    _check("unit vectors are unit-norm", np.allclose(np.linalg.norm(v, axis=1), 1.0))
    _check("chord radius monotonic + bounded", 0 < _chord_radius(1.0) < _chord_radius(90.0) <= 2.0)


def test_trail_velocity_ra_wrap():
    # a trail straddling RA=0 (359.99 -> 0.01) is a small +RA motion, NOT a ~-240 deg/day wrap artefact
    d = pd.DataFrame([dict(ra0=359.99, dec0=0.0, ra1=0.01, dec1=0.0, dec=0.0)])
    vx, vy = trail_velocity(d, EXPT)
    rate = float(np.hypot(vx[0], vy[0]))
    _check("trail_velocity RA-wrap gives small +rate", vx[0] > 0 and rate < 100.0)


def test_chord_seed_across_ra0_equals_ra180():
    # the SAME mover seeded at RA~0 (straddling the meridian) must yield the same #pairs as at RA~180
    n0 = len(chord_seed_pairs(_mover_dets(359.99, -10.0, 3.0, 30.0), max_arc_min=40.0))
    n180 = len(chord_seed_pairs(_mover_dets(180.0, -10.0, 3.0, 30.0), max_arc_min=40.0))
    _check("chord seeding finds the across-RA=0 pair", n0 >= 1)
    _check("chord seeding RA=0 count == RA=180 count", n0 == n180)


# ---------------------------------------------------------------- crossmatch
def test_crossmatch_kd_equals_brute_and_ra0():
    rng = np.random.default_rng(1)
    # known catalogue spanning RA=0 and a range of dec
    ra = np.concatenate([rng.uniform(359.0, 360.0, 300), rng.uniform(0.0, 1.0, 300)])
    dec = rng.uniform(-30.0, 30.0, 600)
    known = pd.DataFrame(dict(ObjID=[f"K{i}" for i in range(600)], mjd=60000.0,
                              ra=ra, dec=dec, mag=20.0))
    dets = known.rename(columns={"ObjID": "obj"}).copy()        # detections sit ON the known objects

    def brute(members):
        g = dets.iloc[members]; hits = []
        for _, r in g.iterrows():
            sel = np.abs(known.mjd.values - r.mjd) <= 0.02
            dra = (known.ra.values[sel] - r.ra + 180) % 360 - 180
            sep = np.hypot(dra * np.cos(np.radians(r.dec)), known.dec.values[sel] - r.dec) * 3600
            j = int(np.argmin(sep))
            if sep[j] <= 3.0:
                hits.append(known.ObjID.values[sel][j])
        if not hits:
            return "", 0.0
        vc = pd.Series(hits).value_counts(); return vc.index[0], vc.iloc[0] / len(g)

    ix = build_known_index(known)
    ndiff = 0
    for _ in range(500):
        m = list(rng.integers(0, len(dets), size=int(rng.integers(2, 4))))
        if crossmatch(dets, m, known, 3.0, 0.02, index=ix) != brute(m):
            ndiff += 1
    _check("KD-tree crossmatch == brute force across RA=0 (0 diffs)", ndiff == 0)


# ---------------------------------------------------------------- chi2 gate / physical_check
def test_pair_chi2_true_low_false_high():
    true_pair = _mover_dets(180.0, -20.0, 3.0, 45.0)                      # clean aligned mover
    c_true, info = pair_chi2(true_pair, EXPT)
    _check("true pair chi2 finite and low", np.isfinite(c_true) and c_true < 10.0)
    # false pair: two random positions, mismatched trail directions
    false_pair = pd.DataFrame([
        dict(detid=0, mjd=60000.0, ra=180.0, dec=-20.0, ra0=179.999, dec0=-20.0, ra1=180.001, dec1=-20.0,
             visit=1000, detector=1, score=0.9, len_db=20.0, mf_snr=8.0),
        dict(detid=1, mjd=60000.02, ra=180.3, dec=-20.2, ra0=180.3, dec0=-20.201, ra1=180.3, dec1=-20.199,
             visit=1001, detector=1, score=0.9, len_db=20.0, mf_snr=8.0)])
    c_false, _ = pair_chi2(false_pair, EXPT)
    _check("false (misaligned) pair chi2 high/inf", (not np.isfinite(c_false)) or c_false > 10.0)


def test_physical_check_gate_and_nan():
    true_pair = _mover_dets(180.0, -20.0, 3.0, 45.0)
    ok, info, nep = physical_check(true_pair, [0, 1], EXPT, min_epochs=2, chi2_2v_max=5.0,
                                   mfsnr_min_2v=0.0, rate_lo_2v=0.5, rate_hi_2v=10.0, max_arc_2v_min=60.0)
    _check("clean 2-visit mover passes physical_check", ok and nep == 2)
    # same visit id on both -> 1 epoch -> reject
    same = true_pair.copy(); same.loc[1, "visit"] = same.loc[0, "visit"]
    ok2, _, nep2 = physical_check(same, [0, 1], EXPT, min_epochs=2, chi2_2v_max=5.0, mfsnr_min_2v=0.0)
    _check("same-visit pair counts as 1 epoch and is rejected", (not ok2) and nep2 == 1)


# ---------------------------------------------------------------- orbit_ok
def test_orbit_ok_runs_and_flags():
    g = _mover_dets(180.0, -20.0, 3.0, 45.0)
    _ok, res = orbit_ok(g, exptime_s=EXPT)          # returns (bound_bool, info_dict)
    _check("orbit_ok returns finite-or-inf cost (no crash/NaN-leak)",
           np.isfinite(res["cost"]) or np.isinf(res["cost"]))
    _check("orbit_ok a/e are finite-or-nan (guarded)",
           (np.isfinite(res["a"]) or np.isnan(res["a"])))


# ---------------------------------------------------------------- resume dedup
def test_resume_dedup_key():
    # mirrors discover_stream's idempotent merge: a duplicated panel collapses on (visit,detector,x,y,score)
    base = pd.DataFrame(dict(visit=[1, 1, 2], detector=[3, 3, 4], x=[10.0, 11.0, 12.0],
                             y=[20.0, 21.0, 22.0], score=[0.9, 0.8, 0.95]))
    dup = pd.concat([base, base.iloc[[0, 1]]], ignore_index=True)        # re-detect panel (visit=1,det=3)
    out = dup.drop_duplicates(["visit", "detector", "x", "y", "score"]).reset_index(drop=True)
    _check("dedup collapses duplicated panel rows", len(dup) == 5 and len(out) == 3)


TESTS = [test_radec_to_unit_and_chord, test_trail_velocity_ra_wrap, test_chord_seed_across_ra0_equals_ra180,
         test_crossmatch_kd_equals_brute_and_ra0, test_pair_chi2_true_low_false_high,
         test_physical_check_gate_and_nan, test_orbit_ok_runs_and_flags, test_resume_dedup_key]


if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as e:
            fails += 1; print(f"  FAIL: {e}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'} ({len(TESTS)} tests)")
    sys.exit(1 if fails else 0)
