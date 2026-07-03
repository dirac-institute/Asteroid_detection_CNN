"""Unit tests for the 2v confidence veto stack (alert schema 1.5, INVESTIGATION_2V_CONFIDENCE.md
sections 6-9): catalog stationarity flag, per-alert FPP, sigma_rate / dt^2 ranking terms, the
pixel-vet formal kill rule (mask-clean snr_at0, combined-OR-single, defect demotion), the
template-footprint static veto (static-static seed exclusion + single-static FLAG demotion), and
the shared-great-circle train/line veto (satellite-train glint chains + static template lines).
Pure functions + synthetic panels -- no GPU, no Butler, no data files."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from ADCNN.linking import pixel_vet as pv
from ADCNN.linking import rank_alerts as ra
from ADCNN.linking.link_2visit import (_finalize_fpp, drop_static_static_pairs, fpp_block,
                                       radec_to_unit, stationarity_check, train_veto_check)

MIN39 = 39.2 / 1440.0                      # the audited WFD pair gap, days


def _members(rate_degday=1.5, dt_days=MIN39, ra0=150.0, dec0=-20.0, v1=101, v2=102):
    """Two member rows moving east at `rate` for `dt` -- the minimal 2v track slice."""
    cosd = np.cos(np.radians(dec0))
    return pd.DataFrame(dict(
        mjd=[61221.0, 61221.0 + dt_days],
        ra=[ra0, ra0 + rate_degday * dt_days / cosd], dec=[dec0, dec0],
        visit=[v1, v2]))


def _trees(points):
    """{visit: (cKDTree, n)} from {visit: [(ra, dec), ...]}."""
    return {v: (cKDTree(radec_to_unit(np.array([p[0] for p in pts]),
                                      np.array([p[1] for p in pts]))), len(pts))
            for v, pts in points.items()}


# --------------------------------------------------------------- catalog stationarity
def test_stationarity_veto_fires_on_counterpart():
    g = _members()                                             # 147" displacement: testable
    st = _trees({101: [(0.0, 0.0)], 102: [(150.0, -20.0)]})    # e1's position repeats in v102
    out = stationarity_check(g, st, {})
    assert out["testable"] and out["vetoStationary"]
    assert out["e1"]["counterpart"] and out["e1"]["sep_arcsec"] < 0.1
    assert not out["e2"]["counterpart"]


def test_stationarity_clean_when_no_counterpart():
    g = _members()
    st = _trees({101: [(0.0, 0.0)], 102: [(150.5, -20.5)]})    # nothing within 3"
    out = stationarity_check(g, st, {})
    assert out["testable"] and not out["vetoStationary"]


def test_stationarity_motion_guard_blocks_self_veto():
    # 49-s companion pair at 4.5 deg/day moves only ~9" -- a real mover's own other-epoch det
    # WOULD be a "counterpart"; the guard must mark the test not-testable instead of vetoing.
    g = _members(rate_degday=4.5, dt_days=49.0 / 86400.0)
    st = _trees({101: [(float(g.ra[1]), float(g.dec[1]))],
                 102: [(float(g.ra[0]), float(g.dec[0]))]})    # each epoch "repeats" in the other
    out = stationarity_check(g, st, {})
    assert out["expectedDispArcsec"] < 10.0
    assert not out["testable"] and not out["vetoStationary"]


# --------------------------------------------------------------- per-alert FPP
CALIB = dict(k_per_det2=3.273e-7, dt_ref_min=39.2, dt_power=2.0,
             calibrated_on="test", calib_quality="test")


def test_fpp_lambda_formula():
    f = fpp_block(CALIB, 4211, 7999, 39.2, visits=[101, 102])
    assert f["lambdaPair"] == pytest.approx(3.273e-7 * 4211 * 7999, abs=5e-4)
    # dt^2 law: half the gap -> quarter the chance area
    f2 = fpp_block(CALIB, 4211, 7999, 19.6)
    assert f2["lambdaPair"] == pytest.approx(f["lambdaPair"] / 4, abs=5e-4)
    assert fpp_block(None, 1, 1, 1.0) is None


def test_finalize_fpp_shares():
    mk = lambda lam, vv: dict(tier="2visit",
                              fpp=dict(lambdaPair=lam, visits=vv, n1=1, n2=1, dtMin=39.0,
                                       perAlertShare=None))
    a1, a2 = mk(0.6, [101, 102]), mk(0.6, [101, 102])      # same pair: share = lambda/2
    a3 = mk(2.5, [103, 104])                               # single alert: min(1, 2.5) clamps
    a4 = dict(tier="3+visit")                              # untouched
    _finalize_fpp([a1, a2, a3, a4])
    assert a1["fpp"]["perAlertShare"] == a2["fpp"]["perAlertShare"] == 0.3
    assert a1["fpp"]["nAlertsPair"] == 2
    assert a3["fpp"]["perAlertShare"] == 1.0
    assert "fpp" not in a4


# --------------------------------------------------------------- sigma_rate + dt^2 bonus
def test_rate_sigma_short_arc():
    # 49-s pair: sqrt(2)*0.4"/49s ~ 0.29 deg/day (the audit's short-arc caveat);
    # 39-min pair: ~0.006 (negligible). rms==0 must NOT claim zero rate error.
    assert ra.rate_sigma_degday(0.0, 49.0 / 86400.0) == pytest.approx(0.288, abs=0.02)
    assert ra.rate_sigma_degday(0.0, MIN39) == pytest.approx(0.0058, abs=0.001)
    # a measured rms above the floor takes over
    assert ra.rate_sigma_degday(0.8, MIN39) == pytest.approx(2 * 0.0058, rel=0.05)


def test_priority_dt_bonus_preserves_tiers():
    # bonus saturates at DT_BONUS_MAX; 2v NEW can never reach the 3+visit base of 3.0
    hi = ra.priority_score("NEW", "2visit", 1.0, 1.0, 10.0, dt_min=0.8)
    assert hi == pytest.approx(2.0 + 0.95 + ra.DT_BONUS_MAX)
    assert hi < 3.0 <= ra.priority_score("NEW", "3+visit", 1.0, 0.0, 10.0)
    # 39-min WFD pair: +0.0026 -- the 2026-06-10 recalibrated ranking is essentially untouched
    wfd = ra.priority_score("NEW", "2visit", 1.0, 0.85, 10.0, dt_min=39.2)
    assert wfd - ra.priority_score("NEW", "2visit", 1.0, 0.85, 10.0) == pytest.approx(0.0026, abs=3e-4)
    # no bonus for known recoveries / missing dt (API-stable default)
    assert ra.priority_score("CONFIRMED", "2visit", 1.0, 0.85, 10.0, dt_min=0.8) == \
        ra.priority_score("CONFIRMED", "2visit", 1.0, 0.85, 10.0)


def test_build_alert_publishes_veto_blocks():
    g = _members()
    g["mag"], g["mf_snr"], g["len_db"], g["score"], g["detector"], g["art_frac"] = \
        22.5, 6.0, 8, 0.85, 5, 0.0
    stat = dict(testable=True, vetoStationary=False, expectedDispArcsec=147.0)
    fpp = dict(lambdaPair=0.5, perAlertShare=0.25)
    al = ra.build_alert(g, alert_id="t", night=61220, obscode="X05", status="NEW", tier="2visit",
                        chi2=1.0, rms_arcsec=0.0, stationarity=stat, fpp=fpp, rate_lo=1.0)
    assert al["schema"].endswith("/1.5")
    assert al["stationarity"] is stat and al["fpp"] is fpp
    m = al["motion"]
    assert m["rate_sigma_degday"] == pytest.approx(0.0058, abs=0.001)
    assert m["neoRateGate"] is True                        # 1.5 - 3*0.006 >> 1.0
    json.dumps(al)                                         # fully serializable
    # 49-s pair at ~1.2 deg/day: rate NOT 3-sigma above the floor -> gate False
    g2 = _members(rate_degday=1.2, dt_days=49.0 / 86400.0)
    al2 = ra.build_alert(g2, alert_id="t2", night=61220, obscode="X05", status="NEW",
                         tier="2visit", chi2=1.0, rms_arcsec=0.0, rate_lo=1.0)
    assert al2["motion"]["neoRateGate"] is False


def test_write_alerts_demotes_flagged_and_killed(tmp_path):
    clean = dict(priorityScore=2.5, priority=2)
    flagged = dict(priorityScore=2.9, priority=2, stationarity=dict(vetoStationary=True))
    pixflag = dict(priorityScore=2.95, priority=2, pixelVet=dict(verdict="FLAGGED", killed=False))
    killed = dict(priorityScore=2.99, priority=2, pixelVet=dict(verdict="STATIC_E1", killed=True))
    p = tmp_path / "a.jsonl"
    ra.write_alerts([killed, flagged, clean, pixflag], p)
    scores = [json.loads(l)["priorityScore"] for l in open(p)]
    # clean first; the two class-1 flags sort among themselves by score; killed dead last
    assert scores == [2.5, 2.95, 2.9, 2.99]
    # the follow-up cap is applied AFTER demotion: a killed alert never crowds out a clean one
    ra.write_alerts([killed, flagged, clean, pixflag], p, top_n=2)
    scores = [json.loads(l)["priorityScore"] for l in open(p)]
    assert scores == [2.5, 2.95]


# --------------------------------------------------------------- template-footprint static veto
def test_static_static_pairs_excluded_single_kept():
    # seed exclusion is BOTH-members-only: static-static pairs die, single/zero-static pairs live
    flag = np.array([True, True, False, True])
    pairs = [[0, 1], [0, 2], [2, 3], [1, 3]]                   # SS, S-, -S, SS
    assert drop_static_static_pairs(pairs, flag) == [[0, 2], [2, 3]]
    assert drop_static_static_pairs([], flag) == []            # empty in, empty out
    # no flagged dets = exact no-op (the --static-catalog-off equivalence at the pair level)
    assert drop_static_static_pairs(pairs, np.zeros(4, bool)) == pairs


def test_static_veto_rank_class_demotion():
    # nStaticMembers>=1 demotes to class 1 (FLAG level); 0 / absent stays clean; pixel-kill wins
    assert ra._rank_class(dict(staticVeto=dict(nStaticMembers=1))) == 1
    assert ra._rank_class(dict(staticVeto=dict(nStaticMembers=0))) == 0
    assert ra._rank_class(dict(staticVeto=None)) == 0
    assert ra._rank_class(dict(staticVeto=dict(nStaticMembers=1),
                               pixelVet=dict(killed=True))) == 2


def test_static_veto_flag_demotes_never_drops(tmp_path):
    clean = dict(priorityScore=2.5, priority=2)
    single = dict(priorityScore=2.9, priority=2, staticVeto=dict(nStaticMembers=1))
    zero = dict(priorityScore=2.7, priority=2, staticVeto=dict(nStaticMembers=0))
    p = tmp_path / "a.jsonl"
    n = ra.write_alerts([single, clean, zero], p)
    assert n == 3                                              # FLAG, never drop
    scores = [json.loads(l)["priorityScore"] for l in open(p)]
    assert scores == [2.7, 2.5, 2.9]                           # clean by score first; flagged one last


def test_build_alert_publishes_static_veto_block():
    g = _members()
    sv = dict(nStaticMembers=1, magMax=20.0, radiusArcsec=3.0,
              members=[dict(visit=101, isStatic=True, sepArcsec=1.8, staticMag=17.2),
                       dict(visit=102, isStatic=False, sepArcsec=45.1, staticMag=19.0)])
    al = ra.build_alert(g, alert_id="t", night=61220, obscode="X05", status="NEW", tier="2visit",
                        chi2=1.0, rms_arcsec=0.0, static_veto=sv, rate_lo=1.0)
    assert al["staticVeto"] is sv and ra._rank_class(al) == 1
    json.dumps(al)                                             # fully serializable
    # default (no catalog): the block is null and the alert stays clean
    al2 = ra.build_alert(g, alert_id="t2", night=61220, obscode="X05", status="NEW", tier="2visit",
                         chi2=1.0, rms_arcsec=0.0, rate_lo=1.0)
    assert al2["staticVeto"] is None and ra._rank_class(al2) == 0


# --------------------------------------------------------------- shared-great-circle train veto
def _circle_frame(g):
    """(n, mid, e2) great-circle basis through the two member positions (as in train_veto_check)."""
    a, b = g.iloc[0], g.iloc[-1]
    p1, p2 = radec_to_unit(a.ra, a.dec), radec_to_unit(b.ra, b.dec)
    n = np.cross(p1, p2); n /= np.linalg.norm(n)
    mid = p1 + p2; mid /= np.linalg.norm(mid)
    return n, mid, np.cross(n, mid)


def _knots(g, alongs_as, *, perp_as=0.0, aligned=True, length=12.0):
    """(u, u0, u1, length) train_arrays entry: dets at the given along-track positions (arcsec from
    the member midpoint), `perp_as` off the members' great circle, with 6\"-long trails ALONG the
    circle (aligned=True) or PERPENDICULAR to it (False)."""
    n, mid, e2 = _circle_frame(g)
    th = np.radians(np.asarray(alongs_as, float) / 3600.0)
    u = np.cos(th)[:, None] * mid + np.sin(th)[:, None] * e2      # exactly on the circle
    ph = np.radians(perp_as / 3600.0)
    u = np.cos(ph) * u + np.sin(ph) * n                           # rotate off-plane by perp_as
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = np.cross(np.broadcast_to(n, u.shape), u)                  # local circle direction
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    v = t if aligned else np.broadcast_to(n, u.shape)             # trail direction
    eps = np.radians(3.0 / 3600.0)
    u0 = u - eps * v; u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    u1 = u + eps * v; u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
    return u, u0, u1, np.full(len(u), float(length))


# visit-101 glint chain (arcsec along-track). NB: the _members() pair sits at along ~ +-73.5"
# (147" separation), so the chain (and its +15" visit-102 drift) stays >2" clear of both members --
# otherwise the 2" self-exclusion eats a knot and the counts are off by one.
ALONGS_1 = np.array([-170.0, -140.0, -110.0, 100.0, 130.0, 160.0])


def test_train_veto_fires_on_glint_train():
    # 6 aligned on-line knots per visit (the train: member B glints near where A's were) -> 12 >= 10
    g = _members()
    ta = {101: _knots(g, ALONGS_1), 102: _knots(g, ALONGS_1 + 15.0)}   # glints DRIFT between visits
    out = train_veto_check(g, ta)
    assert out["tested"] and out["vetoTrain"]
    assert out["nCollinear"] == 12 and out["nAligned"] == 12
    assert [p["nAligned"] for p in out["perVisit"]] == [6, 6]
    assert out["nRepeats"] == 0                                    # drifting glints never repeat


def test_train_veto_clean_for_isolated_mover():
    # background 30" OFF the circle: nothing collinear, no veto (the golden-NEO case scored 1)
    g = _members()
    ta = {101: _knots(g, ALONGS_1, perp_as=30.0), 102: _knots(g, ALONGS_1 + 15.0, perp_as=30.0)}
    out = train_veto_check(g, ta)
    assert out["tested"] and not out["vetoTrain"]
    assert out["nCollinear"] == 0 and out["nAligned"] == 0


def test_train_veto_alignment_and_length_gates():
    g = _members()
    # on-line but trails PERPENDICULAR to the circle: collinear yes, aligned no -> no veto
    ta = {101: _knots(g, ALONGS_1, aligned=False), 102: _knots(g, ALONGS_1 + 15.0, aligned=False)}
    out = train_veto_check(g, ta)
    assert out["nCollinear"] == 12 and out["nAligned"] == 0 and not out["vetoTrain"]
    # on-line and aligned but too SHORT to vote (length <= 5 px): a short trail's PA is noise
    ta = {101: _knots(g, ALONGS_1, length=3.0), 102: _knots(g, ALONGS_1 + 15.0, length=3.0)}
    out = train_veto_check(g, ta)
    assert out["nCollinear"] == 12 and out["nAligned"] == 0 and not out["vetoTrain"]


def test_train_veto_window_and_self_exclusion():
    g = _members()
    # knots beyond the +-1800" along-track window don't count; dets within 2" of the member
    # (its own trail re-detection) are self-excluded
    far = np.array([2200.0, 2500.0, 2800.0, -2200.0, -2500.0, -2800.0])
    n, mid, e2 = _circle_frame(g)
    ta = {101: _knots(g, far), 102: _knots(g, far)}
    assert train_veto_check(g, ta)["nCollinear"] == 0
    # the members themselves sit ON their own circle at some along position -- recover it and
    # place a "det" there: it must be self-excluded
    p1 = radec_to_unit(g.iloc[0].ra, g.iloc[0].dec)
    al1 = np.degrees(np.arctan2(p1 @ e2, p1 @ mid)) * 3600.0
    ta = {101: _knots(g, [al1]), 102: _knots(g, [al1 + 500.0])}
    out = train_veto_check(g, ta)
    assert out["perVisit"][0]["nCollinear"] == 0                   # v101's det = member 1: excluded
    assert out["perVisit"][1]["nCollinear"] == 1                   # 500" away in v102: counted


def test_train_veto_static_line_repeats():
    # the 0630 rank-0 pathology: on-line residuals at IDENTICAL along positions both visits --
    # vetoed by nAligned like a train, and nRepeats tells the vetter it is a STATIC line
    g = _members()
    ta = {101: _knots(g, ALONGS_1), 102: _knots(g, ALONGS_1)}      # sub-arcsec repeats
    out = train_veto_check(g, ta)
    assert out["vetoTrain"] and out["nRepeats"] == 6


def test_train_veto_untested_without_coverage():
    # a member visit missing from the arrays (no pre-floor rows) -> that side untested (None),
    # and with NO covered visit the alert can never be vetoed (fail-safe)
    g = _members()
    out = train_veto_check(g, {101: _knots(g, ALONGS_1)})
    assert out["tested"] and out["perVisit"][1]["nCollinear"] is None
    out = train_veto_check(g, {})
    assert not out["tested"] and not out["vetoTrain"]


def test_train_veto_rank_class_demotion_and_alert_block():
    # vetoTrain demotes to class 1 (FLAG level, same as stationarity/static); pixel-kill still wins
    assert ra._rank_class(dict(trainVeto=dict(vetoTrain=True))) == 1
    assert ra._rank_class(dict(trainVeto=dict(vetoTrain=False))) == 0
    assert ra._rank_class(dict(trainVeto=None)) == 0
    assert ra._rank_class(dict(trainVeto=dict(vetoTrain=True), pixelVet=dict(killed=True))) == 2
    g = _members()
    tv = train_veto_check(g, {101: _knots(g, ALONGS_1), 102: _knots(g, ALONGS_1 + 15.0)})
    al = ra.build_alert(g, alert_id="t", night=61220, obscode="X05", status="NEW", tier="2visit",
                        chi2=1.0, rms_arcsec=0.0, train_veto=tv, rate_lo=1.0)
    assert al["trainVeto"] is tv and ra._rank_class(al) == 1
    json.dumps(al)                                                 # fully serializable
    # default (veto off): the block is null and the alert stays clean -- exact no-op
    al2 = ra.build_alert(g, alert_id="t2", night=61220, obscode="X05", status="NEW", tier="2visit",
                         chi2=1.0, rms_arcsec=0.0, rate_lo=1.0)
    assert al2["trainVeto"] is None and ra._rank_class(al2) == 0


# --------------------------------------------------------------- pixel vet: capsule statistic
class _FlatWCS:
    """0.2"/px linear WCS around (150, -20) -- enough for capsule geometry."""

    def world_to_pixel_values(self, ra, dec):
        cosd = np.cos(np.radians(-20.0))
        return ((np.asarray(ra) - 150.0) * cosd * 3600 / pv.PXSCALE + 100.0,
                (np.asarray(dec) + 20.0) * 3600 / pv.PXSCALE + 100.0)


def _panel(img=None, mask=None, badval=1):
    img = np.zeros((200, 200), np.float32) if img is None else img
    mask = np.zeros((200, 200), np.int32) if mask is None else mask
    return (img, np.ones((200, 200), np.float32), mask, _FlatWCS(), badval)


def test_forced_at0_null_and_blob():
    assert pv.forced_at0(_panel(), 150.0, -20.0, 3.0, 0.0)["snr"] == 0.0
    img = np.zeros((200, 200), np.float32)
    img[98:103, 98:103] = 10.0                             # bright static blob at the position
    r = pv.forced_at0(_panel(img=img), 150.0, -20.0, 3.0, 0.0)
    assert r["snr"] > 5.0 and r["valid"]


def test_forced_at0_mask_clean_excludes_defect_flux():
    # the 000010 retraction: flux living ONLY in BADBITS pixels must not kill
    img = np.zeros((200, 200), np.float32)
    mask = np.zeros((200, 200), np.int32)
    img[98:103, 98:103] = 10.0
    mask[98:103, 98:103] = 1                               # the blob is entirely defect-flagged
    r = pv.forced_at0(_panel(img=img, mask=mask), 150.0, -20.0, 3.0, 0.0)
    assert r["snr"] == 0.0                                 # mask-clean sum sees none of it
    assert 0 < r["badfrac"] <= pv.BADFRAC_MAX              # small capsule fraction -> still valid


def test_forced_at0_defect_dominated_cannot_kill():
    mask = np.zeros((200, 200), np.int32)
    mask[80:120, 80:101] = 1                               # >50% of the capsule defect-flagged
    r = pv.forced_at0(_panel(mask=mask), 150.0, -20.0, 3.0, 0.0)
    assert r is not None and not r["valid"] and r["badfrac"] > pv.BADFRAC_MAX
    mask[:, :] = 1                                         # everything bad: no measurement at all
    assert pv.forced_at0(_panel(mask=mask), 150.0, -20.0, 3.0, 0.0) is None


# --------------------------------------------------------------- pixel vet: kill logic
class _FakeNP:
    """NightPixels stand-in: measure() dispatches on (visit, ra-window)."""

    def __init__(self, visits, vmjd, fn):
        self._v, self.vmjd, self._fn = list(visits), dict(vmjd), fn

    def visits_for_night(self, night):
        return self._v

    def measure(self, visit, ra, dec, rate, pa, *, exptime_s, halfw_px):
        return self._fn(int(visit), float(ra))


def _meas(snr, valid=True):
    return dict(flux=float(snr), var=1.0, snr=float(snr), badfrac=0.0 if valid else 0.9,
                n_good=50, n_tot=60, valid=valid)


EP = dict(ra=150.0, dec=-20.0, mjd=61221.0, visit=101)
KW = dict(exptime_s=30.0, halfw_px=2.0, margin_arcsec=3.0, kill_sigma=5.0, flag_sigma=3.0,
          max_stat_visits=8)


def _np3(fn, dts_min=(39.2, 50.2, 78.4)):
    """3 covering test visits at the given gaps after EP."""
    vm = {101: 61221.0}
    vm.update({102 + i: 61221.0 + d / 1440.0 for i, d in enumerate(dts_min)})
    return _FakeNP(sorted(vm), vm, fn)


def test_stat_epoch_or_rule_single_visit_kill():
    # one 6-sigma static + two quiet visits: combined 6/sqrt(3)=3.5 would NOT kill --
    # the single-visit arm of the OR must (the dilution case the audit called out).
    fn = lambda v, r: _meas(6.0) if v == 102 else _meas(0.0)
    pe = pv._stat_epoch(_np3(fn), EP, 1.5, 90.0, 61220, **KW)
    assert pe["snrCombined"] < 5.0 <= pe["snrMaxSingle"] and pe["static"]


def test_stat_epoch_combined_stack_kill():
    # three 3-sigma repeats: no single visit kills, the sqrt(N) stack does (9/sqrt(3)=5.2) --
    # the sub-5-sigma design point (doc section 8).
    pe = pv._stat_epoch(_np3(lambda v, r: _meas(3.0)), EP, 1.5, 90.0, 61220, **KW)
    assert pe["snrMaxSingle"] == 3.0 < 5.0 <= pe["snrCombined"] and pe["static"]
    assert pe["nValid"] == 3


def test_stat_epoch_displacement_guard():
    # a 1-min gap at 1.5 deg/day moves 3.75" < guard (~4.3"): the mover could still be in the
    # capsule, so the visit is skipped as INVALID -- even though it would read 10 sigma.
    pe = pv._stat_epoch(_np3(lambda v, r: _meas(10.0), dts_min=(1.0,)), EP, 1.5, 90.0, 61220, **KW)
    assert pe["nGuardSkipped"] == 1 and pe["nValid"] == 0 and not pe["static"]


def test_stat_epoch_defect_flags_never_kills():
    # a defect-dominated capsule at 10 sigma may only FLAG (badfrac>0.5 excluded from the stack)
    pe = pv._stat_epoch(_np3(lambda v, r: _meas(10.0, valid=False)), EP, 1.5, 90.0, 61220, **KW)
    assert pe["nValid"] == 0 and not pe["static"] and pe["flagZone"]


def _alert(rate=1.5, dt_days=MIN39):
    g = _members(rate_degday=rate, dt_days=dt_days)
    cosd = np.cos(np.radians(-20.0))
    return dict(alertId="t", night=61220, tier="2visit", status="NEW",
                epochs=[dict(ra=float(r.ra), dec=float(r.dec), mjd=float(r.mjd),
                             visit=int(r.visit), snr=6.0) for _, r in g.iterrows()],
                motion=dict(rate_degday=rate, pa_deg=90.0, dra_degday=rate, ddec_degday=0.0))


E2_RA = float(_members().ra[1])


def _dispatch(e1=0.0, e2=0.0, conf=None):
    """measure() responses by position window: e1 member / e2 member / anywhere else (CONF)."""
    def fn(v, r):
        if abs(r - 150.0) < 0.005:
            return _meas(e1)
        if abs(r - E2_RA) < 0.005:
            return _meas(e2)
        return _meas(conf) if conf is not None else None
    return fn


VKW = dict(exptime_s=30.0, halfw_px=2.0, margin_arcsec=3.0, kill_sigma=5.0, flag_sigma=3.0,
           conf_sigma=5.0, max_stat_visits=8, confident_fpp_max=0.01)


def test_vet_alert_verdicts_and_confident_bit():
    vm = {101: 61221.0, 102: 61221.0 + MIN39}
    np2 = _FakeNP([101, 102], vm, _dispatch())
    al = _alert()
    assert pv.vet_alert(np2, al, **VKW) == "CLEAN"
    assert al["confident"] and not al["pixelVet"]["killed"]

    al = _alert()
    assert pv.vet_alert(_FakeNP([101, 102], vm, _dispatch(e1=8.0)), al, **VKW) == "STATIC_E1"
    assert al["pixelVet"]["killed"] and not al["confident"]

    al = _alert()
    assert pv.vet_alert(_FakeNP([101, 102], vm, _dispatch(e1=8.0, e2=8.0)), al,
                        **VKW) == "STATIC_BOTH"

    al = _alert()
    assert pv.vet_alert(_FakeNP([101, 102], vm, _dispatch(e1=4.0)), al, **VKW) == "FLAGGED"
    assert not al["pixelVet"]["killed"] and not al["confident"]

    al = _alert()
    assert pv.vet_alert(_FakeNP([101, 102], vm, lambda v, r: None), al, **VKW) == "NO_COVERAGE"
    assert not al["confident"]

    # catalog veto / fpp share flow into `confident` even when pixels are clean
    al = _alert()
    al["stationarity"] = dict(vetoStationary=True)
    pv.vet_alert(_FakeNP([101, 102], vm, _dispatch()), al, **VKW)
    assert al["pixelVet"]["verdict"] == "CLEAN" and not al["confident"]
    al = _alert()
    al["fpp"] = dict(perAlertShare=0.5)
    pv.vet_alert(_FakeNP([101, 102], vm, _dispatch()), al, **VKW)
    assert not al["confident"]


def test_vet_alert_third_visit_confirmation():
    # v103 arrives 11 min after e2: the predicted position is 41" past e2 (outside the guard),
    # prediction error ~0.6" (<1"): a 7-sigma clean source there = the mover arriving.
    vm = {101: 61221.0, 102: 61221.0 + MIN39, 103: 61221.0 + MIN39 + 11.0 / 1440.0}
    al = _alert()
    v = pv.vet_alert(_FakeNP([101, 102, 103], vm, _dispatch(conf=7.0)), al, **VKW)
    assert v == "CONFIRMED" and al["confident"]
    c = al["pixelVet"]["confirm"]
    assert c["confirmed"] and c["visit"] == 103 and c["snr"] == 7.0


# --------------------------------------------------------------- graceful no-op
def test_pixel_vet_cli_skips_without_fits_path(tmp_path, monkeypatch):
    alerts = tmp_path / "alerts.jsonl"
    alerts.write_text(json.dumps(dict(alertId="x", priorityScore=2.5, priority=2)) + "\n")
    dets = tmp_path / "dets.csv"
    pd.DataFrame(dict(visit=[1], ra=[0.0], dec=[0.0], mjd=[61221.0])).to_csv(dets, index=False)
    out = tmp_path / "vetted.jsonl"
    monkeypatch.setattr("sys.argv", ["pixel_vet", "--alerts", str(alerts), "--dets", str(dets),
                                     "--out", str(out)])
    pv.main()
    assert out.read_text() == alerts.read_text()          # pass-through, nothing dropped
