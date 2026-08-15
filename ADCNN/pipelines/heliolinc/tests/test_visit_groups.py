"""Visit-group selection for the injection harness — the fix that makes the 3+visit tier testable.

WHY THIS EXISTS. The 3+visit tier had never been validated against injected truth, and the reason
was not effort: `inject_night` selected visit PAIRS by BORESIGHT PROXIMITY (two boresights within
0.3 deg). Applied to triples that criterion yields ZERO groups on every embargo night measured --
and zero at 1.0 deg as well -- so no 3-epoch mover could ever be injected. Meanwhile the pipeline
finds 23 genuine 3+visit tracks across those same nine nights.

Both facts are true because three ~3.5 deg fields can share a footprint while their boresights sit
up to 3.5 deg apart. Boresight proximity is the wrong test for a triple; a COMMON FOOTPRINT is the
right one. Measured with the correct test the same nights offer 353-850 triples each.

These tests pin the distinction down, because it is exactly the kind of mis-posed criterion that
silently produces "no data" and reads as "not possible".
"""
import numpy as np
import pandas as pd
import pytest

from ADCNN.analysis.truth_harness.inject_night import FOV_DEG, visit_groups


def _vc(rows):
    """rows: (visit, ra, dec, minutes_from_start)."""
    return pd.DataFrame(
        [dict(ra=r, dec=d, mjd=61000.0 + m / 1440.0) for _, r, d, m in rows],
        index=[v for v, _, _, _ in rows],
    ).rename_axis("visit")


# ---------------------------------------------------------------- pairs: unchanged behaviour

def test_pairs_reproduce_the_boresight_criterion():
    """The validated 2-epoch truth sets must reproduce, so this path stays exactly as it was."""
    vc = _vc([(1, 10.0, -5.0, 0), (2, 10.1, -5.0, 30), (3, 40.0, -5.0, 30)])
    got = visit_groups(vc, 2)
    assert [g for g, _ in got] == [(1, 2)]          # 3 is co-timed but 30 deg away


# 51.9/52.1 rather than exactly 52: the boundary is a float compare on (mjd_j-mjd_i)*1440
# and testing equality there asserts on rounding, not on behaviour.
@pytest.mark.parametrize("dt,keep", [(5, False), (30, True), (51.9, True), (52.1, False)])
def test_pairs_respect_the_time_window(dt, keep):
    vc = _vc([(1, 10.0, -5.0, 0), (2, 10.05, -5.0, dt)])
    assert bool(visit_groups(vc, 2)) is keep


# ---------------------------------------------------------------- triples: the actual fix

def _shared_footprint_triple():
    """Three fields whose BORESIGHTS are ~1.5-2.6 deg apart -- far outside the 0.3 deg pair rule --
    but which all cover a common point. This is the real survey geometry."""
    return _vc([(1, 10.0, -5.0, 0), (2, 11.5, -5.0, 20), (3, 10.75, -3.7, 40)])


def test_triples_are_found_by_common_footprint_not_boresight_proximity():
    vc = _shared_footprint_triple()
    pos = np.array([[10.75, -4.6]])                  # inside all three fields
    got = visit_groups(vc, 3, positions=pos)
    assert [g for g, _ in got] == [(1, 2, 3)]


def test_the_boresight_criterion_would_have_found_nothing_here():
    """The precise reason the tier was never validated: same geometry, old rule, no groups."""
    vc = _shared_footprint_triple()
    seps = [np.hypot((vc.ra[i] - vc.ra[j]) * np.cos(np.radians(vc.dec[i])), vc.dec[i] - vc.dec[j])
            for i, j in ((1, 2), (1, 3), (2, 3))]
    assert min(seps) > 0.3, "fixture must have boresights outside the pair rule"
    assert visit_groups(vc, 2) == []


def test_triples_need_a_point_covered_by_ALL_THREE():
    vc = _shared_footprint_triple()
    far = np.array([[9.0, -5.5]])                    # inside field 1, outside field 3
    assert np.hypot((9.0 - 10.0) * np.cos(np.radians(-5.5)), -5.5 + 5.0) < FOV_DEG
    assert np.hypot((9.0 - 10.75) * np.cos(np.radians(-5.5)), -5.5 + 3.7) > FOV_DEG
    assert visit_groups(vc, 3, positions=far) == []


def test_triples_respect_the_window_on_the_WIDEST_leg():
    """A triple whose outer legs straddle the window must not be offered: the linker would not
    build it, so injecting into it would manufacture truth the pipeline cannot recover."""
    vc = _vc([(1, 10.0, -5.0, 0), (2, 11.5, -5.0, 20), (3, 10.75, -3.7, 100)])
    assert visit_groups(vc, 3, positions=np.array([[10.75, -4.6]])) == []


def test_triples_are_ordered_shortest_arc_first():
    vc = _vc([(1, 10.0, -5.0, 0), (2, 11.5, -5.0, 5), (3, 10.75, -3.7, 10),
              (4, 10.75, -4.0, 45)])
    got = visit_groups(vc, 3, positions=np.array([[10.75, -4.6]]))
    arcs = [dt for _, dt in got]
    assert arcs == sorted(arcs) and len(got) >= 2


def test_triples_refuse_without_observed_positions():
    """Silently returning [] would look identical to 'no triples exist' -- the failure this whole
    module is about. Refuse loudly instead."""
    with pytest.raises(ValueError):
        visit_groups(_shared_footprint_triple(), 3)


def test_limit_is_applied_after_sorting():
    vc = _vc([(1, 10.0, -5.0, 0), (2, 11.5, -5.0, 5), (3, 10.75, -3.7, 10),
              (4, 10.75, -4.0, 45)])
    got = visit_groups(vc, 3, positions=np.array([[10.75, -4.6]]), limit=1)
    assert len(got) == 1 and got[0][1] == min(
        dt for _, dt in visit_groups(vc, 3, positions=np.array([[10.75, -4.6]])))
