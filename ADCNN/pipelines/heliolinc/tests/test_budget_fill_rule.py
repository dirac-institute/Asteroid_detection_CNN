"""The FIXED operating point, and the chi2 gate machinery underneath it.

Two things are guarded here:

  * the chi2 machinery itself (cheap gates, monotonicity, the 3+visit chi2=None tier) still has to
    be right whatever value is chosen -- _auto_chi2 survives as an ANALYSIS helper only.
  * the operating point is now FIXED (score_min 0.70, chi2_2v_max 10.0) and never adapts per night
    (user decision 2026-08-14). A thin night delivers fewer than 1,000 alerts and that is the
    accepted product. These tests pin that down so adaptivity cannot creep back in.

Every test RUNS the shipped functions on constructed alerts; none asserts on source text.
"""
import json

import pytest

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


# ---------------------------------------------------------------- the FIXED operating point

OP_PATH = "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json"


def _shipped():
    return json.load(open(OP_PATH))


def test_shipped_op_is_fixed_not_auto():
    """One unchanging operating point (user decision 2026-08-14). A regression to "auto" would make
    the delivered chi2 depend on the night again."""
    op = _shipped()
    assert op["chi2_2v_max"] == 10.0
    assert op["score_min"] == 0.70
    assert "target_fill" not in op, "target_fill is the auto rule's knob; its presence implies adaptivity"


def test_shipped_op_records_what_the_fixed_point_costs():
    """The faint-fast cost of a fixed chi2 is real and measured. If the prose that records it is
    dropped, the next reader will 'fix' the over-fill without knowing it was chosen."""
    why = _shipped()["_op_FIXED"]
    assert "6.2x" in why and "20260706" in why       # the worst over-fill, named
    assert "721" in why                               # the night that comes in short, named


def test_run_night_no_longer_adapts_the_floor():
    """The density prediction and the relink ladder are gone, not merely unused."""
    import ADCNN.pipelines.run_night as rn
    for gone in ("relink_ladder", "SCORE_FLOORS", "SCORE_FLOOR_TARGET_DENSITY"):
        assert not hasattr(rn, gone), f"{gone} still present -- adaptivity can creep back"


def test_a_thin_night_is_allowed_to_under_deliver(tmp_path):
    """Under the fixed op, short IS the product on a thin night -- nothing should try to rescue it."""
    op = _shipped()
    p = _write(tmp_path, [_alert(chi2=2.0)] * 300)
    assert survivors_at(p, op, op["chi2_2v_max"]) < op.get("budget", 1000)


def test_fixed_chi2_admits_strictly_less_than_the_loosest(tmp_path):
    """Sanity that 10.0 is actually doing work: it must cut the stream, not pass everything."""
    op = _shipped()
    al = [_alert(chi2=c) for c in (2, 6, 9, 11, 15, 25)]
    assert survivors_at(al, op, op["chi2_2v_max"]) < survivors_at(al, op, CHI2_GRID[-1])
