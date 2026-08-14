"""The delivered-budget fill rule: chi2 auto-selection and the score-floor relink ladder.

These guard the two halves of the 2026-08-13 cemented op point:

  * chi2_2v_max = "auto" -- a FIXED chi2 does not transfer across cadence (the measured flagship
    optimum is 8 on 20260706 and 16 on 20260713), but the BUDGET FILL RATIO does. filter_op picks
    the chi2 reaching target_fill x budget.
  * the score floor is chosen by LINK-CHECK-RELINK, not by predicting from detection density. The
    prediction under-filled 3 of 3 sparse nights (20260712 shipped 644 of 1,000 slots), and the
    scan measured score_min 0.50/0.60/0.70 to be completeness-IDENTICAL -- so relaxing is free and
    the ladder only ever steps down.

Every test RUNS the shipped functions on constructed alerts; none asserts on source text.
"""
import json

import pytest

from ADCNN.pipelines.run_night import SCORE_FLOORS, relink_ladder
from ADCNN.qa.filter_op import (CHI2_GRID, TARGET_FILL, _auto_chi2, _passes_cheap,
                                survivors_at)

OP = {"mfsnr_min_2v": 3.0, "len_db_min": 6.0, "rate_lo_2v": 1.0, "rate_hi_2v": 8.0,
      "budget": 1000, "target_fill": TARGET_FILL}


def _alert(chi2=1.0, mfsnr=5.0, length=20.0, rate=3.0):
    return {"orbit": ({"chi2": chi2} if chi2 is not None else {}),
            "vetting": {"mfsnr_min": mfsnr, "trail_len_px": [length, length]},
            "motion": {"rate_degday": rate}}


def _write(tmp_path, alerts):
    p = tmp_path / "alerts.jsonl"
    p.write_text("".join(json.dumps(a) + "\n" for a in alerts))
    return str(p)


# ---------------------------------------------------------------- the cheap gates

def test_chi2_none_passes_it_is_the_3visit_discovery_tier():
    """A 3+visit alert has no 2-visit orbit chi2. Treating None as failing once dropped the whole
    tier silently -- the exact regression this asserts against."""
    assert _passes_cheap(_alert(chi2=None), OP, chi2_max=3.0)
    assert survivors_at([_alert(chi2=None)] * 7, OP, 3.0) == 7


@pytest.mark.parametrize("kw,ok", [
    (dict(chi2=2.0), True),
    (dict(chi2=99.0), False),          # above chi2_max
    (dict(mfsnr=1.0), False),          # below mfsnr_min_2v
    (dict(length=2.0), False),         # below len_db_min
    (dict(rate=0.2), False),           # below rate_lo_2v
    (dict(rate=25.0), False),          # above rate_hi_2v
])
def test_each_cheap_gate_is_actually_applied(kw, ok):
    # bool() because the gate chain ends in a numpy scalar (np.mean on trail_len_px); it is only
    # ever consumed for truthiness, so np.False_ is correct behaviour, not a defect.
    assert bool(_passes_cheap(_alert(**kw), OP, chi2_max=8.0)) is ok


def test_survivors_at_is_monotone_in_chi2():
    al = [_alert(chi2=c) for c in (1, 4, 7, 9, 13, 17, 21, 40)]
    counts = [survivors_at(al, OP, c) for c in CHI2_GRID]
    assert counts == sorted(counts)
    assert survivors_at(al, OP, 1e9) == len(al)


def test_survivors_at_accepts_a_path_and_a_list_identically(tmp_path):
    al = [_alert(chi2=c) for c in (1, 5, 9, 20)]
    assert survivors_at(_write(tmp_path, al), OP, 10.0) == survivors_at(al, OP, 10.0)


# ---------------------------------------------------------------- auto chi2

def test_auto_chi2_picks_the_smallest_chi2_reaching_the_target(tmp_path):
    """1,000 alerts at chi2=2 and 1,000 at chi2=13, target 1.9x1000=1900. chi2<=12 yields only
    1,000, so auto must step past it to 14."""
    p = _write(tmp_path, [_alert(chi2=2.0)] * 1000 + [_alert(chi2=13.0)] * 1000)
    assert _auto_chi2(p, OP, budget=1000, target_fill=1.9) == 14


def test_auto_chi2_is_a_plateau_not_a_knife_edge(tmp_path):
    """MEASURED on both injected nights: 1.6/1.75/1.9 all pick chi2<=8 on the tune night and
    1.75/1.9/2.0 all pick chi2<=16 on the held-out one. Insensitivity is why the rule is usable."""
    p = _write(tmp_path, [_alert(chi2=2.0)] * 1500 + [_alert(chi2=13.0)] * 2000)
    assert len({_auto_chi2(p, OP, budget=1000, target_fill=t) for t in (1.6, 1.75, 1.9)}) == 1


def test_auto_chi2_returns_the_loosest_grid_point_when_the_night_cannot_fill(tmp_path):
    """A short night must not silently pick a TIGHT chi2 -- it should open all the way up, and the
    caller learns the budget was not the binding constraint."""
    p = _write(tmp_path, [_alert(chi2=2.0)] * 50)
    assert _auto_chi2(p, OP, budget=1000, target_fill=1.9) == CHI2_GRID[-1]


def test_auto_chi2_agrees_with_survivors_at(tmp_path):
    """The refactor that gave run_night its own view of fill must not fork the gate logic."""
    # 200 alerts at each integer chi2 1..39, so chi2<=c admits 200*c and the target 1,900 is
    # reachable partway up the grid rather than off its end.
    p = _write(tmp_path, [_alert(chi2=c) for c in range(1, 40)] * 200)
    pick = _auto_chi2(p, OP, budget=1000, target_fill=1.9)
    prev = [c for c in CHI2_GRID if c < pick]
    assert pick < CHI2_GRID[-1], "fixture must fill before the grid runs out"
    assert survivors_at(p, OP, pick) >= 1900
    if prev:
        assert survivors_at(p, OP, prev[-1]) < 1900


# ---------------------------------------------------------------- the relink ladder

@pytest.mark.parametrize("start,expect", [
    (0.70, [0.70, 0.60, 0.50]),
    (0.60, [0.60, 0.50]),
    (0.50, [0.50]),
    (None, [0.70, 0.60, 0.50]),
])
def test_ladder_only_ever_steps_down(start, expect):
    assert relink_ladder(start) == expect


def test_ladder_never_raises_the_floor_above_the_prediction():
    """Raising is the one direction the scan does NOT license: 0.80 is the first floor measured to
    cost delivered completeness."""
    for start in (0.50, 0.60, 0.70):
        assert max(relink_ladder(start)) <= start


def test_ladder_never_offers_0_80_whatever_the_inputs():
    for start in (None, 0.50, 0.60, 0.70, 0.90):
        assert max(relink_ladder(start, prev=None)) <= 0.70


@pytest.mark.parametrize("prev,expect", [
    (0.70, [0.60, 0.50]),
    (0.60, [0.50]),
    (0.50, [0.50]),          # exhausted: hand back the lowest so the caller links once, not never
])
def test_a_floor_already_tried_is_not_repeated(prev, expect):
    """Relinking at the floor the existing stream was built with would reproduce it exactly."""
    assert relink_ladder(0.70, prev=prev) == expect


def test_ladder_respects_whichever_of_prediction_and_prev_is_lower():
    assert relink_ladder(0.60, prev=0.70) == [0.60, 0.50]
    assert relink_ladder(0.70, prev=0.60) == [0.50]


def test_ladder_is_never_empty():
    """An empty ladder would silently skip the link entirely and leave the night with no stream."""
    for start in (None, 0.50, 0.60, 0.70):
        for prev in (None, 0.50, 0.60, 0.70):
            assert relink_ladder(start, prev)


def test_a_night_that_fills_stops_at_the_first_rung(tmp_path):
    """The loop is self-limiting: it fires only on SHORT nights, which are sparse, which is exactly
    where the lower floor is cheap. It must never fire on a dense night where 0.50 is intractable."""
    p = _write(tmp_path, [_alert(chi2=2.0)] * 5000)
    want = OP["target_fill"] * OP["budget"]
    assert survivors_at(p, OP, CHI2_GRID[-1]) >= want


def test_a_short_night_is_detected_as_short(tmp_path):
    """20260712's failure mode: 1,125 alerts is under the 1,900 the budget needs, so the loop must
    see it and relax rather than ship 644 of 1,000 slots."""
    p = _write(tmp_path, [_alert(chi2=2.0)] * 1125)
    want = OP["target_fill"] * OP["budget"]
    assert survivors_at(p, OP, CHI2_GRID[-1]) < want
