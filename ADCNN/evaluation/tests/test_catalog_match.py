"""Tests for ADCNN.evaluation.catalog_match (run directly: `python -m
ADCNN.evaluation.tests.test_catalog_match`; no pytest dependency)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ADCNN.evaluation.catalog_match import (
    match_trail_catalogs, evaluate_catalog,
    _pairwise_segment_distance, _segment_endpoints,
)


def _meas(rows):  return pd.DataFrame(rows, columns=["image_id", "x", "y", "beta", "length"])
def _truth(rows): return pd.DataFrame(rows, columns=["image_id", "x", "y", "beta", "trail_length"])


def test_geometry_segment_distance():
    # horizontal segment y=0, x in [0,10]; a parallel segment offset by 3 in y -> dist 3
    a = _segment_endpoints(_meas([[0, 5, 0, 0, 10]]), "length")          # along x at y=0
    b = _segment_endpoints(_meas([[0, 5, 3, 0, 10]]), "length")          # along x at y=3
    assert abs(_pairwise_segment_distance(a, b)[0, 0] - 3.0) < 1e-9
    # crossing segments -> distance 0
    cross = _segment_endpoints(_meas([[0, 5, 0, 90, 20]]), "length")     # vertical through (5,0)
    assert _pairwise_segment_distance(a, cross)[0, 0] == 0.0


def test_overlap_disjoint_and_counts():
    truth = _truth([[0, 100, 100, 0, 40], [0, 500, 500, 90, 40]])        # 2nd is missed
    meas = _meas([[0, 105, 103, 0, 38], [0, 2000, 2000, 0, 40]])         # 1st hits truth0, 2nd is FP
    t, m, c = match_trail_catalogs(meas, truth, tol_px=10.0)
    assert c == {"TP": 1, "FN": 1, "FP": 1}, c
    assert list(t["nn_detected"]) == [True, False]
    assert list(m["matched"]) == [True, False]


def test_tolerance_boundary():
    truth = _truth([[0, 100, 100, 0, 40]])
    meas = _meas([[0, 100, 115, 0, 40]])                                 # 15 px off-axis
    assert match_trail_catalogs(meas, truth, tol_px=20.0)[2]["TP"] == 1
    assert match_trail_catalogs(meas, truth, tol_px=10.0)[2]["TP"] == 0


def test_multiplicity_one_truth_many_detections():
    truth = _truth([[0, 100, 100, 0, 40]])
    meas = _meas([[0, 95, 100, 0, 20], [0, 110, 101, 0, 20]])            # two fragments on one trail
    t, m, c = match_trail_catalogs(meas, truth, tol_px=10.0)
    assert c == {"TP": 1, "FN": 0, "FP": 0}, c                          # truth counted once, both matched


def test_per_panel_isolation():
    truth = _truth([[0, 100, 100, 0, 40]])
    meas = _meas([[1, 100, 100, 0, 40]])                                 # same geom, different panel
    assert match_trail_catalogs(meas, truth, tol_px=20.0)[2] == {"TP": 0, "FN": 1, "FP": 1}


def test_empty_and_nan_safe():
    truth = _truth([[0, 100, 100, 0, 40]])
    assert match_trail_catalogs(_meas([]), truth, tol_px=20.0)[2] == {"TP": 0, "FN": 1, "FP": 0}
    nan_meas = _meas([[0, np.nan, 100, 0, 40]])                          # non-finite geometry never matches
    assert match_trail_catalogs(nan_meas, truth, tol_px=20.0)[2] == {"TP": 0, "FN": 1, "FP": 1}


def test_missing_column_raises():
    try:
        match_trail_catalogs(_meas([[0, 1, 1, 0, 1]]), pd.DataFrame({"image_id": [0]}), tol_px=1.0)
    except ValueError as e:
        assert "trail_length" in str(e)
    else:
        raise AssertionError("expected ValueError for missing truth column")


def test_evaluate_catalog_metrics():
    truth = _truth([[0, 100, 100, 0, 40], [1, 200, 200, 0, 40]])
    meas = _meas([[0, 102, 100, 0, 40]])                                 # hits truth on panel 0 only
    metrics, t = evaluate_catalog(meas, truth, tol_px=20.0)
    assert metrics["TP"] == 1 and metrics["FN"] == 1 and metrics["FP"] == 0
    assert abs(metrics["recall"] - 0.5) < 1e-9 and metrics["n_panels"] == 2
    assert "nn_detected" in t.columns


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  PASS {t.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED")
