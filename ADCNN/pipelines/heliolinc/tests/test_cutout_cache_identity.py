"""The cutout cache is keyed by alert POSITION, so a permutation silently re-captions every image.

This shipped. `run_night` built the cache and THEN ran `rerank_alerts`, which rewrites alerts.jsonl in
place -- on 20260710 only 13 of 18,009 positions survived, and the delivered sheet_0000.png matches a
re-render from a linker-order cache to 0.13% of pixels while differing from the correct render in
76.27%. Six of nine delivered nights ran in that order.

Two independent defences are tested here: the ORDER in run_night (rank before cut, so the cache is
correct by construction) and the identity guard (which must see a permutation ANYWHERE, not just in
the first 400 rows it used to sample).
"""
import json
import re
import subprocess
import sys

import numpy as np
import pytest

from ADCNN.qa import alert_sheets


def _write(tmp_path, n, permute_from=None):
    """n alerts, each on its own (visit, detector); cache built in the ORIGINAL order."""
    alerts = [{"alertId": i, "epochs": [{"visit": 1000 + i, "detector": i % 189,
                                         "ra": 10.0 + i * 1e-3, "dec": -20.0}]} for i in range(n)]
    ai = np.arange(n)
    vv = np.array([1000 + i for i in range(n)])
    dd = np.array([i % 189 for i in range(n)])
    npz = tmp_path / "cutouts.npz"
    np.savez(npz, alert=ai, epoch=np.zeros(n, int), visit=vv, detector=dd,
             stamps=np.zeros((n, 4, 4), np.float16))
    (tmp_path / "cutouts_meta.json").write_text(json.dumps({"n_alerts": n}))
    if permute_from is not None:                       # rerank-style permutation of the TAIL only
        tail = alerts[permute_from:][::-1]
        alerts = alerts[:permute_from] + tail
    ap = tmp_path / "alerts.jsonl"
    ap.write_text("".join(json.dumps(a) + "\n" for a in alerts))
    return str(ap), str(npz)


def test_matched_cache_passes(tmp_path):
    ap, npz = _write(tmp_path, 1000)
    alert_sheets._assert_cache_matches(ap, npz, 1000)


def test_permutation_beyond_the_old_400_row_sample_is_caught(tmp_path):
    """The guard sampled `range(min(len(ai), 400))`. The cache is RANK-ORDERED, so that covered ~200
    of 20,000 alerts: permuting only ranks >=500 mis-addressed 536 of 1,037 and the guard PASSED,
    writing sheets whose every caption past rank 500 was wrong."""
    ap, npz = _write(tmp_path, 1000, permute_from=500)
    with pytest.raises(SystemExit, match="does NOT MATCH"):
        alert_sheets._assert_cache_matches(ap, npz, 1000)


def test_missing_sidecar_refuses_instead_of_skipping(tmp_path):
    ap, npz = _write(tmp_path, 50)
    (tmp_path / "cutouts_meta.json").unlink()
    with pytest.raises(SystemExit, match="no cutouts_meta.json"):
        alert_sheets._assert_cache_matches(ap, npz, 50)


def test_smaller_limit_off_a_larger_cache_is_legitimate(tmp_path):
    """A cache for the top 20,000 of a 26,253-alert file is the SHIPPED shape; an equality test made
    re-rendering that night's top-N from its own cache impossible."""
    ap, npz = _write(tmp_path, 600)
    (tmp_path / "cutouts_meta.json").write_text(json.dumps({"n_alerts": 400}))
    alert_sheets._assert_cache_matches(ap, npz, 100)           # 100 <= 400 <= 600
    with pytest.raises(SystemExit, match="STALE"):
        alert_sheets._assert_cache_matches(ap, npz, 500)       # asks past what the cache covers


def test_run_night_ranks_before_it_cuts():
    """The structural fix. If the cut precedes the rerank the cache is stale the moment it is written,
    and no guard can recover the pixels -- it can only refuse to render them."""
    src = (__import__("pathlib").Path(alert_sheets.__file__).parent.parent
           / "pipelines" / "run_night.py").read_text()
    # match the INVOCATIONS, not the prose -- run_night's docstring names alert_cutouts too
    i_rank = src.index("python -m ADCNN.qa.rerank_alerts")
    i_cut = src.index("python -m ADCNN.qa.alert_cutouts")
    assert i_rank < i_cut, ("run_night cuts cutouts before re-ranking: the cache is keyed by alert "
                            "position and rerank_alerts permutes alerts.jsonl in place")


def _fingerprint(alerts):
    import hashlib
    h = hashlib.sha256()
    for a in alerts:
        for e in (a.get("epochs") or []):
            h.update(f"{e.get('visit',-1)}:{e.get('detector',-1)};".encode())
        h.update(b"|")
    return h.hexdigest()


def _write_fp(tmp_path, n):
    alerts = [{"alertId": i, "ranking": {"pReal": 1.0 - i / n},
               "epochs": [{"visit": 1000 + i, "detector": i % 189, "ra": 10.0 + i * 1e-3,
                           "dec": -20.0}]} for i in range(n)]
    np.savez(tmp_path / "c.npz", alert=np.arange(n), epoch=np.zeros(n, int),
             visit=np.array([1000 + i for i in range(n)]), detector=np.array([i % 189 for i in range(n)]),
             stamps=np.zeros((n, 4, 4), np.float16), wide=np.zeros((n, 4, 4), np.float16),
             wide_alert=np.arange(n))
    (tmp_path / "c_meta.json").write_text(json.dumps(
        {"n_alerts": n, "alerts_fingerprint": _fingerprint(alerts)}))
    return alerts, str(tmp_path / "c.npz")


def _dump(tmp_path, alerts):
    p = tmp_path / "a.jsonl"
    p.write_text("".join(json.dumps(a) + "\n" for a in alerts))
    return str(p)


def test_fingerprint_catches_a_single_adjacent_swap(tmp_path):
    """The row-by-row guard needs the npz; the fingerprint is O(1) and covers the WHOLE sequence, so
    night_status can certify a product it currently cannot check. One swap is the smallest corruption."""
    alerts, npz = _write_fp(tmp_path, 400)
    alert_sheets._assert_cache_matches(_dump(tmp_path, alerts), npz, 400)
    sw = list(alerts); sw[300], sw[301] = sw[301], sw[300]
    with pytest.raises(SystemExit, match="fingerprint differs"):
        alert_sheets._assert_cache_matches(_dump(tmp_path, sw), npz, 400)


def test_select_clean_refreshes_the_fingerprint_it_carries_over(tmp_path):
    """The sidecar is copied from the source cache, so without a refresh the reindexed 1k product
    would fail its own check and no night could be rendered."""
    from ADCNN.qa.select_clean import select
    alerts, npz = _write_fp(tmp_path, 200)
    ap = _dump(tmp_path, alerts)
    drop = np.zeros(200, bool); drop[[3, 11, 150]] = True
    np.savez(tmp_path / "m.npz", ripple=drop)
    select(ap, str(tmp_path / "m.npz"), npz, 200,
           str(tmp_path / "o.jsonl"), str(tmp_path / "o.npz"), mode="rings")
    alert_sheets._assert_cache_matches(str(tmp_path / "o.jsonl"), str(tmp_path / "o.npz"), 197)
