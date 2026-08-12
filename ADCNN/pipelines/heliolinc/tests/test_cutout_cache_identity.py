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
