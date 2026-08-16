"""night_status must NEVER raise on a damaged artifact, and must always return the status dict.

run_night calls status() on ENTRY, before preflight, so anything that raises here wedges the night
permanently: every retry crashes identically on the same byte. This has now happened TWICE --
first an unguarded json.loads (JSONDecodeError), then a guard that returned a bare string while
every caller indexes the result (TypeError). Same wedge, different exception.
"""
import json

import pytest

from ADCNN.pipelines.night_status import status

CONTRACT = ("night", "complete", "first_missing", "detail")


def _night(tmp_path, alerts_bytes=None, dets=True):
    d = tmp_path / "run_night_20260706"
    (d / "stream").mkdir(parents=True)
    if dets:
        (d / "adcnn_dets_masked.csv").write_text("detid,ra,dec\n1,10.0,-20.0\n")
    if alerts_bytes is not None:
        (d / "stream" / "alerts.jsonl").write_bytes(alerts_bytes)
    return d


@pytest.mark.parametrize("payload", [
    b'{"alertId": "a", "epochs": [',                  # truncated mid-object
    b'{"alertId": "a"}\n{"alertId": "b", "epo',       # truncated mid-line, valid first line
    b'\x00\x01\x02 not json at all\n',                # binary garbage
    b'{"no_alert_id": 1}\n',                          # valid json, missing the key
    b'',                                              # empty
])
def test_damaged_alerts_never_raises_and_returns_a_dict(tmp_path, payload):
    d = _night(tmp_path, payload)
    s = status(str(d))
    assert isinstance(s, dict), f"returned {type(s).__name__}; callers index this and will TypeError"
    for k in CONTRACT:
        assert k in s, f"missing contract key {k!r}"
    assert s["complete"] is False


def test_every_return_path_satisfies_the_contract(tmp_path):
    """Including the healthy-but-incomplete and no-detections paths."""
    for kwargs in ({"alerts_bytes": None, "dets": True},      # no stream yet
                   {"alerts_bytes": None, "dets": False}):    # nothing at all
        s = status(str(_night(tmp_path / str(id(kwargs)), **kwargs)))
        assert isinstance(s, dict) and all(k in s for k in CONTRACT)


# ---------------------------------------------------------------- the deliver stage

def _complete_night(tmp_path, with_1k=None):
    """A night whose STREAM chain fully verifies; with_1k adds a stream_1k in the given state."""
    import json
    d = tmp_path / "run_night_20260706"
    sd = d / "stream"
    (sd / "pairs").mkdir(parents=True)
    (sd / "sheets").mkdir()
    (d / "adcnn_dets_masked.csv").write_text("detid,ra,dec\n1,10.0,-20.0\n")
    alerts = [{"alertId": f"2v_61227_{i:06d}", "epochs": [{"visit": 1, "detector": 2}]}
              for i in range(3)]
    (sd / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in alerts))
    for i, a in enumerate(alerts):
        (sd / "pairs" / f"alert_{i:05d}_p0.90_{a['alertId']}_CLEAN.png").write_bytes(b"png")
    (sd / "sheets" / "index.html").write_text("x")
    (sd / "sheets" / "sheet_0000.png").write_bytes(b"png")
    (sd / "stream_summary.json").write_text(json.dumps({"n_alerts": 3}))
    if with_1k is not None:
        kd = d / "stream_1k"
        (kd / "pairs").mkdir(parents=True)
        (kd / "sheets").mkdir()
        kalerts = alerts[:2]
        (kd / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in kalerts))
        for i, a in enumerate(kalerts):
            (kd / "pairs" / f"alert_{i:05d}_p0.90_{a['alertId']}_CLEAN.png").write_bytes(b"png")
        (kd / "sheets" / "index.html").write_text("x")
        (kd / "sheets" / "sheet_0000.png").write_bytes(b"png")
        (kd / "stream_summary.json").write_text(json.dumps({"n_alerts": 2}))
        if with_1k == "missing_pair":
            next(iter((kd / "pairs").iterdir())).unlink()
        elif with_1k == "summary_mismatch":
            (kd / "stream_summary.json").write_text(json.dumps({"n_alerts": 99}))
        elif with_1k == "empty_alerts":
            (kd / "alerts.jsonl").write_text("")
    return d


def test_absent_stream_1k_is_not_a_failure(tmp_path):
    """run_night alone does not build the 1k product; its absence must not block completion."""
    s = status(str(_complete_night(tmp_path)))
    assert s["complete"] is True


def test_consistent_stream_1k_certifies(tmp_path):
    s = status(str(_complete_night(tmp_path, with_1k="ok")))
    assert s["complete"] is True and "deliver" in s["detail"]


@pytest.mark.parametrize("state", ["missing_pair", "summary_mismatch", "empty_alerts"])
def test_half_built_stream_1k_never_certifies(tmp_path, state):
    """THE HOLE THIS STAGE CLOSES: the campaign wrote .regen_complete off this module's verdict
    while the 1k build's rc was logged and ignored -- a night whose 1k chain died mid-way was
    marked VERIFIED COMPLETE and skipped by every later re-entry."""
    s = status(str(_complete_night(tmp_path, with_1k=state)))
    assert s["complete"] is False and s["first_missing"] == "deliver", s["detail"]


def test_stale_sentinel_cannot_bypass_the_deliver_stage(tmp_path):
    """A sentinel older than a rebuilt stream_1k must force re-verification, or deleting/replacing
    the delivered product after certification would go unnoticed."""
    import os, time
    d = _complete_night(tmp_path, with_1k="ok")
    (d / ".complete").write_text("")
    past = time.time() - 3600
    os.utime(d / ".complete", (past, past))          # sentinel predates the 1k product
    (d / "stream_1k" / "alerts.jsonl").write_text("")   # ...which is now broken
    s = status(str(d))
    assert s["complete"] is False and s["first_missing"] == "deliver"
