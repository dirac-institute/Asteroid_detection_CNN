"""Tests for the regenerate-and-confirm calibration stages.

Run with pytest from the repo root, or standalone:
    python -m ADCNN.calibration.tests.test_threshold_selection
The threshold tests load the committed 82-field validation caches; they SKIP (not fail) if those
caches are absent from the checkout. The decision-rule and fail-loud tests run pure functions and
always execute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ADCNN.calibration import threshold_selection as TS
from ADCNN.calibration import calibrate_mflen as MF

REPO = Path(__file__).resolve().parents[3]
CACHE = TS.DEFAULT_CACHE_DIR
FROZEN_OP = REPO / "ADCNN/pipelines/heliolinc/op_2v_alert.json"
_HAVE_CACHE = Path(CACHE).exists() and any(Path(CACHE).glob("*_smin0.6_v3exact.json"))


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except Exception:
        print(f"SKIP: {msg}")
    return True


# ------------------------------------------------------------------ decision rule on synthetic curves
def test_score_rule_is_lowest_S_meeting_purity_floor():
    # monotone-increasing purity in S; completeness ~flat. Floor 75% -> lowest S with P>=75.
    P_vals = {0.75: 56.0, 0.775: 67.0, 0.80: 77.0, 0.825: 85.0, 0.85: 90.0}
    C_vals = {s: 6.0 for s in TS.S_GRID}

    def C(S, mf):
        return C_vals.get(round(S, 4), 6.0)

    def P(S, mf):
        return P_vals.get(round(S, 4), 100.0 if S > 0.85 else 0.0)

    sel, band = TS._select_score(C, P, mf=5, floor=75.0)
    assert sel == 0.80, f"purity-floor 75% should select 0.80, got {sel}"
    # the stable band is (P just below, P at) -> any floor in (67, 77] selects 0.80
    assert band[0] < 75.0 <= band[1]


def test_score_rule_rejects_larger_S_framings():
    # several S clear the 75% floor (0.80, 0.825, 0.85). "largest S on plateau" would pick a
    # larger S; the purity-floor rule must take the LOWEST feasible (0.80).
    def P(S, mf):
        return {0.775: 67.0, 0.80: 77.0, 0.825: 85.0, 0.85: 90.0}.get(round(S, 4),
                                                                       95.0 if S > 0.85 else 0.0)
    sel, _ = TS._select_score(lambda S, mf: 6.0, P, mf=5, floor=75.0)
    assert sel == 0.80, f"must take the lowest S clearing the floor, not a larger one; got {sel}"


def test_mfsnr_rule_is_largest_retaining_completeness():
    # completeness retention vs uncut drops sharply after mf=5.
    Cret = {0: 7.0, 3: 6.95, 5: 6.07, 6: 5.08, 7: 3.86, 10: 1.68}

    def C(S, mf):
        return Cret.get(mf, 7.0 if mf < 3 else (6.0 if mf < 6 else 1.0))

    sel, band = TS._select_mfsnr(C, 0.80, retention=0.80)
    assert sel == 5, f"retention 0.80 should select mfsnr=5, got {sel}"
    assert band[0] < 0.80 <= band[1]


# ------------------------------------------------------------------ fail-loud confirm
def test_confirm_raises_on_drift(tmp_path=None):
    import json
    selected = {"score_min": 0.70, "mfsnr_min": 5.0, "chi2_max": 5.0, "rate_lo": 1.0, "rate_hi": 8.0}
    raised = False
    try:
        TS.confirm_against_frozen(selected, FROZEN_OP)
    except TS.ThresholdSelectionError:
        raised = True
    assert raised, "confirm must FAIL LOUD when the regenerated score disagrees with the frozen op"


def test_confirm_passes_on_match():
    selected = {"score_min": 0.80, "mfsnr_min": 5.0, "chi2_max": 5.0, "rate_lo": 1.0, "rate_hi": 8.0}
    TS.confirm_against_frozen(selected, FROZEN_OP)  # must not raise


# ------------------------------------------------------------------ end-to-end regenerate-and-confirm
def test_regenerate_selects_080_5_from_committed_caches():
    if not _HAVE_CACHE:
        return _skip("validation caches absent from checkout")
    selected, _ = TS.run(cache_dir=CACHE, frozen_op=FROZEN_OP, confirm=True)
    assert selected["score_min"] == 0.80
    assert selected["mfsnr_min"] == 5.0
    # documented metrics at the op (purity-floor methodology): C~6.07%, P~76.9%
    assert abs(selected["at_op"]["faint_fast_completeness_pct"] - 6.07) < 0.2
    assert abs(selected["at_op"]["in_sample_purity_pct"] - 76.9) < 0.5


# ------------------------------------------------------------------ MF_LEN re-fit reproduces frozen
def test_mflen_fit_reproduces_frozen():
    csv = MF.DEFAULT_FIT_CSV
    if not Path(csv).exists():
        return _skip("MF_LEN fit-pairs CSV absent from checkout")
    import pandas as pd
    fitted = MF.fit(pd.read_csv(csv))
    MF.confirm_against_frozen(fitted)  # must not raise (within tol of 7.67/0.9425)
    assert abs(fitted["offset"] - 7.67) < MF.TOL_OFFSET
    assert abs(fitted["slope"] - 0.9425) < MF.TOL_SLOPE


def test_mflen_confirm_raises_on_drift():
    bad = {"offset": 33.4, "slope": 0.887, "fit_n": 100, "residual_px": 1.0}  # v1 constants
    raised = False
    try:
        MF.confirm_against_frozen(bad)
    except MF.MFLenCalibrationError:
        raised = True
    assert raised, "MF_LEN confirm must fail loud when the re-fit reproduces v1 constants, not current"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all threshold-selection tests passed")
