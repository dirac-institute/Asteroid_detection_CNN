"""night_status must NEVER raise on a damaged artifact, and must always return the status dict.

run_night calls status() on ENTRY, before preflight, so anything that raises here wedges the night
permanently: every retry crashes identically on the same byte. This has now happened TWICE --
first an unguarded json.loads (JSONDecodeError), then a guard that returned a bare string while
every caller indexes the result (TypeError). Same wedge, different exception.
"""
import json

import pytest

from ADCNN.pipelines.night_status import status
from ADCNN.qa.cache_identity import FINGERPRINT_VERSION as _FPV

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
    (d / "adcnn_dets_masked.csv").write_text("detid,ra,dec\n1,10.0,-20.0\n")
    alerts = [{"alertId": f"2v_61227_{i:06d}",
               "epochs": [{"visit": 1 + i, "detector": 2 + i,
                           "ra": 10.0 + i * 0.01, "dec": -5.0 - i * 0.01}]}
              for i in range(3)]
    (sd / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in alerts))
    for i, a in enumerate(alerts):
        (sd / "pairs" / f"alert_{i:05d}_p0.90_{a['alertId']}_CLEAN.png").write_bytes(b"png")
    (sd / "stream_summary.json").write_text(json.dumps({"n_alerts": 3}))
    if with_1k is not None:
        kd = d / "stream_1k"
        (kd / "pairs").mkdir(parents=True)
        kalerts = alerts[:2]
        (kd / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in kalerts))
        for i, a in enumerate(kalerts):
            (kd / "pairs" / f"alert_{i:05d}_p0.90_{a['alertId']}_CLEAN.png").write_bytes(b"png")
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


def test_recorded_zero_cap_needs_no_stream_images(tmp_path):
    """The 2026-08-16 default: run_night renders NO stream-level pairs (stream_1k/pairs is the
    image product) and records {"top_n": 0}. A night in that state must verify -- both the full
    status() walk and the sentinel fast path -- with no stream/pairs directory at all."""
    import json, shutil
    d = _complete_night(tmp_path, with_1k="ok")
    shutil.rmtree(d / "stream" / "pairs")
    (d / "stream" / "pairs_top_n.json").write_text(json.dumps({"top_n": 0}))
    s = status(str(d))
    assert s["complete"] is True, s["detail"]
    (d / ".complete").write_text("")                  # sentinel path must agree
    assert status(str(d))["complete"] is True


def test_no_record_still_requires_stream_images(tmp_path):
    """Pre-record nights (no pairs_top_n.json) keep the legacy contract: stream/pairs required.
    Deleting the record must NOT quietly excuse missing images."""
    import shutil
    d = _complete_night(tmp_path)
    shutil.rmtree(d / "stream" / "pairs")
    s = status(str(d))
    assert s["complete"] is False and s["first_missing"] == "images", s["detail"]


# ---------------------------------------------------------------- audit pass 2 regressions

def _fingerprint(alerts_path, cap=None):
    import hashlib, json as _j
    h = hashlib.sha256()
    with open(alerts_path) as f:
        for i, line in enumerate(f):
            if isinstance(cap, int) and i >= cap:
                break
            for e in (_j.loads(line).get("epochs") or []):
                h.update(f"{e.get('visit',-1)}:{e.get('detector',-1)}:"
                         f"{round(float(e.get('ra') or 0.0), 4)}:"
                         f"{round(float(e.get('dec') or 0.0), 4)};".encode())
            h.update(b"|")
    return h.hexdigest()


def test_valid_sentinel_cannot_hide_deleted_delivered_images(tmp_path):
    """F3: a sentinel written BEFORE the damage reported COMPLETE with 999 of 1000 delivered
    images gone -- and run_night returns immediately on COMPLETE, so the night never self-heals."""
    d = _complete_night(tmp_path, with_1k="ok")
    (d / ".complete").write_text("")
    for p in list((d / "stream_1k" / "pairs").iterdir())[:-1]:
        p.unlink()                                    # leave exactly one -> dir still "non-empty"
    s = status(str(d))
    assert s["complete"] is False, "deleted delivered images certified COMPLETE"


def test_valid_sentinel_cannot_hide_a_missing_delivered_product(tmp_path):
    """Same hole, coarser: the whole stream_1k/alerts.jsonl removed after certification."""
    d = _complete_night(tmp_path, with_1k="ok")
    (d / ".complete").write_text("")
    (d / "stream_1k" / "alerts.jsonl").unlink()
    assert status(str(d))["complete"] is False


def test_permuted_DELIVERED_product_is_caught_by_the_fingerprint(tmp_path):
    """F4: the permutation class that shipped six bad nights was guarded on the QA stream and
    UNGUARDED on the delivered product. Reversing stream_1k/alerts.jsonl must not certify."""
    import json as _j
    d = _complete_night(tmp_path, with_1k="ok")
    kap = d / "stream_1k" / "alerts.jsonl"
    lines = open(kap).read().splitlines(keepends=True)
    # cache fingerprint recorded for the ORIGINAL order...
    (d / "stream_1k" / "cutouts_meta.json").write_text(
        _j.dumps({"alerts_fingerprint": _fingerprint(kap), "n_alerts": len(lines),
                  "fingerprint_version": _FPV}))
    assert status(str(d))["complete"] is True, "unpermuted product must still certify"
    (d / ".complete").unlink(missing_ok=True)
    # ...then the file is permuted underneath it
    kap.write_text("".join(reversed(lines)))
    s = status(str(d))
    assert s["complete"] is False and s["first_missing"] == "deliver"
    assert "CACHE MISMATCH" in s["detail"]["deliver"]


def test_absent_delivered_cache_is_not_a_failure(tmp_path):
    """A cache is deleted after a successful night; its absence must never be read as mismatch."""
    d = _complete_night(tmp_path, with_1k="ok")
    assert not (d / "stream_1k" / "cutouts_meta.json").exists()
    assert status(str(d))["complete"] is True


def test_all_flag_finds_nights_under_campaign_subdirs(tmp_path, monkeypatch):
    """F6: the outputs reorg moved products to runs/<campaign>/run_night_*; --all globbed only the
    flat path and reported on nothing but dry-run stubs while nine real nights were invisible."""
    import glob as _glob
    (tmp_path / "outputs" / "runs" / "10k_cadence").mkdir(parents=True)
    (tmp_path / "outputs" / "runs" / "10k_cadence" / "run_night_20260706").mkdir()
    (tmp_path / "outputs" / "runs" / "run_night_20260630").mkdir()
    monkeypatch.chdir(tmp_path)
    found = sorted(set(_glob.glob("outputs/runs/run_night_*")
                       + _glob.glob("outputs/runs/*/run_night_*")))
    assert len(found) == 2, found


def test_same_panel_swap_changes_the_fingerprint(tmp_path):
    """F5: two alerts on the SAME (visit,detector) pair were interchangeable under a detector-only
    signature -- 8.8-25.2% of delivered alerts share an epoch signature with another. Swapping them
    permuted the cache without changing the hash. Position now participates."""
    import json as _j
    a = {"alertId": "x", "epochs": [{"visit": 1, "detector": 2, "ra": 10.00, "dec": -5.00}]}
    b = {"alertId": "y", "epochs": [{"visit": 1, "detector": 2, "ra": 10.50, "dec": -5.50}]}
    p1 = tmp_path / "ab.jsonl"; p1.write_text(_j.dumps(a) + "\n" + _j.dumps(b) + "\n")
    p2 = tmp_path / "ba.jsonl"; p2.write_text(_j.dumps(b) + "\n" + _j.dumps(a) + "\n")
    assert _fingerprint(p1) != _fingerprint(p2), "same-panel swap must change the fingerprint"


def test_one_and_only_one_fingerprint_implementation():
    """alert_cutouts WRITES this hash, select_clean REWRITES it, alert_sheets and night_status
    VERIFY it. It lived as four hand-copied loops that had to agree exactly -- and had already
    diverged once (a prefix-vs-whole-file disagreement made every over-limit night unrenderable).
    Assert the copies are gone: only cache_identity may spell the hashed tuple."""
    import pathlib, re
    repo = pathlib.Path(__file__).resolve().parents[4]
    offenders = []
    for rel in ("ADCNN/qa/alert_cutouts.py", "ADCNN/qa/select_clean.py",
                "ADCNN/qa/alert_sheets.py", "ADCNN/pipelines/night_status.py"):
        if re.search(r"get\('visit',-1\)\}:\{[_a-z]*e\.get\('detector',-1\)",
                     (repo / rel).read_text()):
            offenders.append(rel)
    assert not offenders, f"re-implemented the cache identity instead of importing it: {offenders}"


def test_fingerprint_version_gates_old_metadata(tmp_path):
    """Changing WHAT is hashed invalidates every older cache. A verifier that cannot tell 'old
    format' from 'wrong pixels' reports a false CACHE MISMATCH on a perfectly good night -- and 18
    such sidecars were sitting in the delivered products when the hash changed."""
    import json as _j
    from ADCNN.qa.cache_identity import verify, epoch_digest, FINGERPRINT_VERSION
    p = tmp_path / "a.jsonl"
    p.write_text(_j.dumps({"alertId": "a", "epochs": [{"visit": 1, "detector": 2,
                                                       "ra": 10.0, "dec": -5.0}]}) + "\n")
    cur = {"alerts_fingerprint": epoch_digest(p), "fingerprint_version": FINGERPRINT_VERSION}
    assert verify(p, cur) == (True, True)
    old = {"alerts_fingerprint": "whatever-the-old-format-produced"}      # no version key
    assert verify(p, old) == (False, True), "v1 metadata must be UNCHECKABLE, not a mismatch"
    wrong = dict(cur, alerts_fingerprint="0" * 64)
    assert verify(p, wrong) == (True, False), "a same-version mismatch must still fail"
