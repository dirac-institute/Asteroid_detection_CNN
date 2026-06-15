"""Consolidation guards: active pipeline resolution, frozen op-points, and the leakage guard.

These lock in the consolidation contract:
  * the DEFAULT pipeline is the current detector with its OWN MF_LEN de-bias (never the v1 default),
  * the prior baseline stays selectable and carries the v1 de-bias,
  * the frozen alert/discovery op-point values are unchanged (golden values),
  * blind/test leakage is caught at the (visit,detector) exposure level.
"""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from ADCNN.config import REPO, load_pipeline
from ADCNN.pipelines.leakage_guard import assert_disjoint, LeakageError, visit_detector_pairs

HL = REPO / "ADCNN" / "pipelines" / "heliolinc"


# --------------------------------------------------------------------------- active pipeline
def test_default_pipeline_is_current_with_its_own_debias():
    p = load_pipeline()
    assert p.name == "current"
    assert (p.mf_len_offset, p.mf_len_slope) == (7.67, 0.9425), "default de-bias must be the current model's"
    assert p.seg_model.exists() and p.cnn_model.exists()
    # never the v1 generic files
    assert "segmentation_model.pt" not in p.seg_model.name


def test_legacy_pipeline_carries_v1_debias():
    p = load_pipeline("legacy_v1")
    assert p.name == "legacy_v1"
    assert (p.mf_len_offset, p.mf_len_slope) == (33.4, 0.887)
    assert p.seg_model.exists() and p.cnn_model.exists()


def test_env_pipeline_selection(monkeypatch):
    monkeypatch.setenv("ADCNN_PIPELINE", "legacy_v1")
    assert load_pipeline().name == "legacy_v1"


def test_env_mflen_override_wins(monkeypatch):
    monkeypatch.setenv("ADCNN_MF_LEN_OFFSET", "0")
    monkeypatch.setenv("ADCNN_MF_LEN_SLOPE", "1")
    p = load_pipeline("current")
    assert (p.mf_len_offset, p.mf_len_slope) == (0.0, 1.0)


def test_catalog_module_default_debias_is_current():
    # the inference engine's module-level de-bias must resolve to the current model's, not v1's
    import importlib
    import ADCNN.inference.catalog as cat
    importlib.reload(cat)
    assert (cat.MF_LEN_OFFSET, cat.MF_LEN_SLOPE) == (7.67, 0.9425)


# --------------------------------------------------------------------------- frozen op-points (golden)
def test_op_2v_alert_frozen_values():
    op = json.loads((HL / "op_2v_alert.json").read_text())
    assert op["score_min"] == 0.80
    assert op["chi2_2v_max"] == 5.0
    assert op["mfsnr_min_2v"] == 5.0
    assert (op["rate_lo_2v"], op["rate_hi_2v"]) == (1.0, 8.0)
    assert op["alerts_top_n"] == 50


def test_link_op_point_frozen_values():
    op = json.loads((HL / "link_op_point.json").read_text())
    assert op["score_min"] == 0.80
    assert op["chi2_2v_max"] == 5.0
    assert op["mfsnr_min_2v"] == 10.0  # discovery tier is STRICTER than the alert tier (deliberate)
    assert (op["rate_lo_2v"], op["rate_hi_2v"]) == (1.0, 8.0)


# --------------------------------------------------------------------------- leakage guard
def test_leakage_guard_disjoint_ok(tmp_path):
    pd.DataFrame({"visit": [1, 2, 3], "detector": [10, 11, 12]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"visit": [4, 5], "detector": [10, 11]}).to_csv(tmp_path / "blind.csv", index=False)
    assert assert_disjoint(tmp_path / "train.csv", tmp_path / "blind.csv") == (3, 2)


def test_leakage_guard_overlap_raises(tmp_path):
    pd.DataFrame({"visit": [1, 2, 3], "detector": [10, 11, 12]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"visit": [3, 5], "detector": [12, 11]}).to_csv(tmp_path / "blind.csv", index=False)  # (3,12) shared
    with pytest.raises(LeakageError):
        assert_disjoint(tmp_path / "train.csv", tmp_path / "blind.csv")


def test_leakage_guard_missing_column_raises(tmp_path):
    pd.DataFrame({"visit": [1]}).to_csv(tmp_path / "bad.csv", index=False)  # no detector column
    with pytest.raises(ValueError):
        visit_detector_pairs(tmp_path / "bad.csv")


# --------------------------------------------------------------------------- entry point wiring
def test_run_experiment_stage_table_consistent():
    from ADCNN.pipelines import run_experiment as rx
    assert set(rx.DISPATCH) == set(rx.STAGE_ORDER)
    assert rx.GPU_STAGES <= set(rx.STAGE_ORDER)
    assert "report" in rx.STAGE_ORDER and "report" not in rx.GPU_STAGES
