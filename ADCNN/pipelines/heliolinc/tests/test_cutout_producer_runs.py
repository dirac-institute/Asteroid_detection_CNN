"""RUN alert_cutouts.build(), do not merely import it.

This exists because a `NameError: hashlib` shipped inside `build()` while 98 tests stayed green: the
import smoke test only imports the module, and every cache-identity test hand-builds its npz with
np.savez. Nothing executed the producer. The crash landed between np.savez_compressed and the
_meta.json write, so the 2 GB npz was written and its sidecar never was -- and since every renderer
now refuses a sidecar-less cache, and run_night's reuse guard skips the cut when the npz exists, the
night wedged in a state only --force could clear (which also redoes ~45 min of linking).

Same class as the catalog.py syntax error that passed 63 tests and killed a 2 h GPU job.
"""
import json

import numpy as np
import pytest

from ADCNN.qa import alert_cutouts, alert_sheets


def _night(tmp_path, n=6):
    """A minimal but REAL invocation: alerts.jsonl + a dets CSV the panel loader can consume."""
    import pandas as pd
    alerts, rows = [], []
    for i in range(n):
        eps = [{"visit": 900 + i, "detector": 3, "ra": 10.0 + i * 1e-3, "dec": -20.0,
                "mjd": 60000.0 + j * 0.01} for j in range(2)]
        alerts.append({"alertId": i, "night": "20260706", "epochs": eps,
                       "motion": {"rate_degday": 2.0, "pa_deg": 30.0}})
        for e in eps:
            rows.append(dict(visit=e["visit"], detector=e["detector"], ra=e["ra"], dec=e["dec"],
                             mjd=e["mjd"], x=100.0, y=100.0,
                             fits_path=f"/nonexistent/diffim_{e['visit']}_{e['detector']}.fits"))
    ap = tmp_path / "alerts.jsonl"
    ap.write_text("".join(json.dumps(a) + "\n" for a in alerts))
    dp = tmp_path / "dets.csv"
    pd.DataFrame(rows).to_csv(dp, index=False)
    return str(ap), str(dp)


def _build(tmp_path, limit=None):
    ap, dp = _night(tmp_path)
    out = str(tmp_path / "cutouts.npz")
    # Every panel read fails (the paths do not exist), which is the point: build() must still complete
    # and write its sidecar. A pipeline that only works when every panel loads is not a pipeline.
    alert_cutouts.build(ap, dp, out, workers=1, limit=limit)
    return ap, out


def test_build_completes_and_writes_its_sidecar(tmp_path):
    """The regression: build() raised NameError AFTER writing the npz, leaving no sidecar."""
    ap, out = _build(tmp_path)
    import os
    assert os.path.exists(out), "npz not written"
    meta = out.replace(".npz", "_meta.json")
    assert os.path.exists(meta), "sidecar missing -- every renderer refuses this cache"
    assert json.load(open(meta)).get("alerts_fingerprint"), "no fingerprint recorded"


def test_the_cache_build_produces_passes_the_renderer_guard(tmp_path):
    """Producer and consumer must agree. They did not: the fingerprint is recorded over the --limit
    PREFIX but was verified over the WHOLE file, so any night larger than stream_top_n was
    unrenderable -- while the count branch in the same function explicitly blesses that shape."""
    ap, out = _build(tmp_path)
    alert_sheets._assert_cache_matches(ap, out, 6)


def test_a_limited_cache_still_matches_its_own_alerts_file(tmp_path):
    """run_night ALWAYS passes --limit, so this is the shipped path, not an edge case."""
    ap, out = _build(tmp_path, limit=3)
    assert json.load(open(out.replace(".npz", "_meta.json")))["n_alerts"] == 3
    alert_sheets._assert_cache_matches(ap, out, 3)


def test_a_permutation_INSIDE_the_cut_prefix_is_still_caught(tmp_path):
    """Prefix-hashing must not become a way to smuggle a permutation past the guard."""
    ap, out = _build(tmp_path, limit=3)
    al = [json.loads(l) for l in open(ap)]
    al[0], al[1] = al[1], al[0]                     # swap two alerts WITHIN the cached prefix
    open(ap, "w").write("".join(json.dumps(a) + "\n" for a in al))
    with pytest.raises(SystemExit, match="fingerprint differs"):
        alert_sheets._assert_cache_matches(ap, out, 3)


def test_select_clean_refuses_a_cache_that_does_not_match_its_alerts(tmp_path):
    """select_clean reindexes by POSITION and then stamps a FRESH fingerprint, so without validating
    its input it laundered a permuted cache into a product every downstream guard accepted -- proven
    on real 0710 pixels: 20 of 30 alerts carried the wrong alert's stamps, mean 66.66% of pixels
    differing, guard PASSED."""
    from ADCNN.qa.select_clean import select
    ap, out = _build(tmp_path)
    al = [json.loads(l) for l in open(ap)]
    open(ap, "w").write("".join(json.dumps(a) + "\n" for a in al[:1] + al[1:][::-1]))
    np.savez(tmp_path / "m.npz", ripple=np.zeros(6, bool))
    with pytest.raises(SystemExit):
        select(ap, str(tmp_path / "m.npz"), out, 6,
               str(tmp_path / "o.jsonl"), str(tmp_path / "o.npz"), mode="rings")
