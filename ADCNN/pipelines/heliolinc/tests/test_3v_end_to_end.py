"""END-TO-END: a dec-moving 3-epoch mover survives the REAL product chain.

Runs the actual CLIs — link_2visit → rerank_alerts → filter_op — on a synthetic catalogue whose one
real mover has exactly the geometry the tangent-plane shear used to kill: high RA (318 deg), motion
mostly in dec (PA 133), three epochs on the real triple cadence (0 / +2.7 / +37 min). Before the
2026-08-15 projection fix this mover's fitted motion PA came out ~39 deg instead of 133 and rule 3
rejected it; the injection campaign measured 64 of 71 TRUE triplets lost this way.

This is the test the tier never had: not a unit check of one function, but the full chain the
nightly product runs, asserting the mover arrives (a) linked as tier 3+visit, (b) carrying an
outer-pair chi2 and the geometry block, (c) ranked at the TOP by both rerank and filter_op (the
discovery tier outranks every 2-visit alert), and (d) surviving the 1k op's gates.

Noise detections are seeded deterministically so chance pairs exist and the claim competition is
real, not vacuous.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OP_FULL = os.path.join(REPO, "ADCNN/pipelines/heliolinc/op_2v_stream_fullcadence.json")
OP_1K = os.path.join(REPO, "ADCNN/pipelines/heliolinc/op_2v_stream_1k.json")
RANK_MODEL = os.path.join(REPO, "ADCNN/calibration/alert_ranking_model.json")

RATE, PA = 3.2, 132.6              # deg/day, sky PA -- the dec-heavy shear geometry
RA0, DEC0 = 317.94, -22.72
EXPT = 30.0
MJD0 = 61228.32
EPOCH_MIN = (0.0, 2.7, 37.0)       # the real 20260706 triple's cadence


def _mover_rows(rng):
    vx = RATE * np.cos(np.radians(PA))
    vy = RATE * np.sin(np.radians(PA))
    dt = EXPT / 86400.0
    rows = []
    for k, tm in enumerate(EPOCH_MIN):
        t = tm / 1440.0
        dec = DEC0 + vy * t
        cd = np.cos(np.radians(dec))
        ra = RA0 + vx * t / cd
        rows.append(dict(mjd=MJD0 + t, ra=ra, dec=dec, visit=600691 + k, detector=183,
                         ra0=ra - vx * dt / 2 / cd, dec0=dec - vy * dt / 2,
                         ra1=ra + vx * dt / 2 / cd, dec1=dec + vy * dt / 2,
                         score=0.95, len_db=20.0, length=20.0, mf_snr=8.0, mag=22.0, art_frac=0.0))
    return rows


def _noise_rows(rng, n_per_visit=30):
    """Static-sky noise with random trails: enough for real chance pairs, deterministic."""
    rows = []
    for k, tm in enumerate(EPOCH_MIN):
        t = tm / 1440.0
        for _ in range(n_per_visit):
            ra = RA0 + rng.uniform(-0.4, 0.4)
            dec = DEC0 + rng.uniform(-0.4, 0.4)
            L = rng.uniform(6.5, 18.0) * 0.2 / 3600.0          # px -> deg
            th = rng.uniform(0, np.pi)
            cd = np.cos(np.radians(dec))
            rows.append(dict(mjd=MJD0 + t, ra=ra, dec=dec, visit=600691 + k,
                             detector=int(rng.integers(1, 180)),
                             ra0=ra - L / 2 * np.cos(th) / cd, dec0=dec - L / 2 * np.sin(th),
                             ra1=ra + L / 2 * np.cos(th) / cd, dec1=dec + L / 2 * np.sin(th),
                             score=float(rng.uniform(0.72, 0.99)),
                             len_db=float(L * 3600.0 / 0.2), length=float(L * 3600.0 / 0.2),
                             mf_snr=float(rng.uniform(3.5, 9.0)),
                             mag=23.0, art_frac=0.0))
    return rows


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Run the real CLI chain once; every test inspects its outputs."""
    td = tmp_path_factory.mktemp("e2e3v")
    rng = np.random.default_rng(4242)
    dets = pd.DataFrame(_mover_rows(rng) + _noise_rows(rng))
    dets_csv = td / "dets.csv"; dets.to_csv(dets_csv, index=False)
    known = td / "known.csv"; known.write_text("ObjID,ra,dec,mjd\n")
    alerts = td / "alerts.jsonl"
    env = dict(os.environ, PYTHONPATH=REPO)

    r1 = subprocess.run(
        [sys.executable, "-m", "ADCNN.linking.link_2visit",
         "--dets", str(dets_csv), "--known", str(known),
         "--out", str(td / "tracks.csv"), "--op-point", OP_FULL,
         "--npt", "2", "--min-epochs", "2", "--seed-2v", "chord", "--train-veto",
         "--claim-order", "preal", "--rank-by", "chi2",
         "--alerts-out", str(alerts)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r1.returncode == 0, f"link failed:\n{r1.stdout[-2000:]}\n{r1.stderr[-2000:]}"

    r2 = subprocess.run(
        [sys.executable, "-m", "ADCNN.qa.rerank_alerts", "--alerts", str(alerts),
         "--model", RANK_MODEL],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, f"rerank failed:\n{r2.stdout[-1500:]}\n{r2.stderr[-1500:]}"

    # empty deep refcat: the proximity veto runs its real code path against zero stars, keeping
    # the test self-contained (no Butler-derived files) while still refusing-by-default upstream.
    refcat = td / "refcat.parquet"
    pd.DataFrame({"ra": pd.Series(dtype=float), "dec": pd.Series(dtype=float),
                  "mag": pd.Series(dtype=float)}).to_parquet(refcat)
    surv = td / "surv.jsonl"
    r3 = subprocess.run(
        [sys.executable, "-m", "ADCNN.qa.filter_op", "--alerts", str(alerts),
         "--dets", str(dets_csv), "--op", OP_1K, "--out", str(surv),
         "--refcat", str(refcat)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert r3.returncode == 0, f"filter_op failed:\n{r3.stdout[-1500:]}\n{r3.stderr[-1500:]}"

    return dict(alerts=[json.loads(l) for l in open(alerts)],
                surv=[json.loads(l) for l in open(surv)])


def _is_mover(a):
    """All epochs of the alert lie on the injected mover's track (within 1.5")."""
    vx = RATE * np.cos(np.radians(PA)); vy = RATE * np.sin(np.radians(PA))
    for e in a["epochs"]:
        t = e["mjd"] - MJD0
        dec = DEC0 + vy * t
        ra = RA0 + vx * t / np.cos(np.radians(dec))
        if np.hypot((e["ra"] - ra) * np.cos(np.radians(dec)), e["dec"] - dec) > 1.5 / 3600.0:
            return False
    return True


def test_mover_is_linked_as_3visit(chain):
    got = [a for a in chain["alerts"] if _is_mover(a)]
    assert got, "the injected mover produced no alert at all"
    assert any(a["tier"] == "3+visit" and a["nEpochs"] == 3 for a in got), \
        f"mover linked but not as 3+visit: {[(a['tier'], a['nEpochs']) for a in got]}"


def test_mover_carries_chi2_and_geometry(chain):
    a = next(x for x in chain["alerts"] if _is_mover(x) and x["tier"] == "3+visit")
    assert a["orbit"]["chi2"] is not None, "outer-pair chi2 must be populated"
    gm = a["geometry"]
    assert gm["nPoints"] == 3 and gm["linRmsArcsec"] is not None
    assert gm["trailMotionDpaMaxDeg"] < 20.0, "trail agrees with UNSHEARED motion"
    # published motion must be the true injected motion, not the sheared one
    assert abs(a["motion"]["rate_degday"] - RATE) < 0.15 * RATE
    dpa = abs(((a["motion"]["pa_deg"] % 180.0) - PA + 90) % 180 - 90)
    assert dpa < 10.0, f"published motion PA {a['motion']['pa_deg']} vs injected {PA}"


def test_rerank_puts_the_discovery_tier_first(chain):
    ranked = chain["alerts"]
    k = next(i for i, a in enumerate(ranked) if _is_mover(a) and a["tier"] == "3+visit")
    n2_before = sum(1 for a in ranked[:k] if a["tier"] == "2visit")
    assert n2_before == 0, f"{n2_before} 2-visit alerts outrank the 3+visit mover (rank {k})"


def test_filter_op_delivers_the_mover_on_top(chain):
    surv = chain["surv"]
    assert surv, "1k op delivered nothing"
    k = next((i for i, a in enumerate(surv) if _is_mover(a) and a["tier"] == "3+visit"), None)
    assert k is not None, "mover did not survive the 1k op"
    assert all(a["tier"] != "2visit" for a in surv[:k]), "a 2-visit alert outranks the mover"
