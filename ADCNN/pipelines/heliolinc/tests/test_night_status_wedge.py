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
