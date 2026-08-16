"""The op file is a SPECIFICATION: every non-prose key must be consumed by some stage.

The 2026-08-15 audit found op_2v_stream_1k.json declaring three boolean "cuts" that no code read
(drop_confident_fp / drop_stationary_single / morphology_dipole_veto): toggling each changed
survivors 38->38 while a real knob (chi2 10->5) changed 38->7, and 5.4% of delivered alerts carried
vetoStationary "despite" drop_stationary_single -- the switch was fiction, the FLAG-not-drop policy
was the reality. A key nobody reads is worse than no key: it documents behaviour the product does
not have. This test pins the contract: the op's parameter keys must be a subset of the keys some
stage actually consumes, so a new dead key fails CI instead of shipping as false documentation.

F11 (the verifier's hardcoded render cap) is pinned here too; F7/F10 live in
test_run_night_guards.py as real-CLI subprocess checks.
"""
import json
import os

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OP_1K = os.path.join(REPO, "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json")
OP_FULL = os.path.join(REPO, "ADCNN/pipelines/heliolinc/op_2v_stream_fullcadence.json")

# THE AUTHORITATIVE CONSUMER MAP. A key may be added here ONLY together with the code that reads
# it. Linker keys are the FLAGS map in link_2visit._main; filter_op keys are its op[...] /
# op.get(...) reads; budget/target_fill are filter_op's auto-chi2 machinery plus run_night.
CONSUMED = {
    # link_2visit FLAGS
    "score_min", "score_candidate_min", "score_hiconf", "chi2_2v_max", "mfsnr_min_2v",
    "rate_lo_2v", "rate_hi_2v", "pa_tol", "pa_tol_2v", "max_rms", "pos_tol_3v",
    "max_arc_2v_min", "promote_3v", "promote_tol_arcsec", "alerts_top_n",
    "seed_3v_arc_min", "promote_from", "len_db_min",
    # filter_op
    "bright_star_proximity", "bright_star_radius_arcsec", "bright_star_mag_max",
    "budget", "target_fill",
}

DEAD_KEYS = {"drop_confident_fp", "drop_stationary_single", "morphology_dipole_veto"}


@pytest.mark.parametrize("path", [OP_1K, OP_FULL])
def test_every_op_parameter_is_consumed_somewhere(path):
    op = json.load(open(path))
    params = {k for k in op if not k.startswith("_")}
    dead = params - CONSUMED
    assert not dead, (
        f"{os.path.basename(path)} declares parameter key(s) no stage reads: {sorted(dead)}. "
        f"Either implement the key or move its content into a _prose field -- an unread switch "
        f"is false documentation (see the 2026-08-15 audit / test docstring).")


@pytest.mark.parametrize("path", [OP_1K, OP_FULL])
def test_the_dead_booleans_stay_dead(path):
    """The three specific keys the audit falsified must never reappear as parameters."""
    op = json.load(open(path))
    back = DEAD_KEYS & set(op)
    assert not back, f"{sorted(back)} reappeared in {os.path.basename(path)}"


def test_chi2_prose_no_longer_claims_auto():
    op = json.load(open(OP_1K))
    assert "'auto'" not in op["_chi2_TRUTH"], (
        "_chi2_TRUTH still points readers at chi2='auto'; the shipped op is FIXED 10.0 (_op_FIXED)")
    assert op["chi2_2v_max"] == 10.0 and op["score_min"] == 0.7, "THE fixed op-point changed"


def test_image_cap_reads_recorded_top_n(tmp_path):
    """night_status must verify against the RECORDED render cap, not a hardcoded CLI default."""
    from ADCNN.pipelines.night_status import _image_cap, IMAGE_CAP
    sd = tmp_path / "stream"
    sd.mkdir()
    assert _image_cap(sd) == IMAGE_CAP, "no record -> the legacy fallback"
    (sd / "pairs_top_n.json").write_text(json.dumps({"top_n": 123}))
    assert _image_cap(sd) == 123, "a recorded cap must win over the fallback"
    (sd / "pairs_top_n.json").write_text("not json{")
    assert _image_cap(sd) == IMAGE_CAP, "a corrupt record degrades to the fallback, not a crash"
