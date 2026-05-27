"""Smoke tests for the CNN stage-2 pipeline wiring (CPU-only, no GPU / no model files needed).

Verifies the RandomForest -> cutout-CNN refactor is internally consistent: the catalog schema
uses `score` (not `score_rf`), the config exposes `cnn_thr`, the CNN scores a candidate frame
and applies a threshold, and the trainer's focal loss + cutout extraction run end to end on a
freshly built (untrained) net. This is the leakage-safe replacement for the old RF
`validate_pipeline.py`.
"""
import numpy as np
import pandas as pd
import pytest


def test_catalog_schema_is_cnn():
    from ADCNN.inference.catalog import CATALOG_COLUMNS, InferenceConfig
    assert "score" in CATALOG_COLUMNS
    assert "score_rf" not in CATALOG_COLUMNS
    cfg = InferenceConfig()
    assert hasattr(cfg, "cnn_thr") and not hasattr(cfg, "rf_thr")


def test_no_rf_modules_remain():
    import importlib
    for mod in ("ADCNN.inference.rf_postproc", "ADCNN.inference.rf_train"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def test_inference_exports_resolve():
    import ADCNN.inference as inf
    for name in ("compute_v2_features", "label_candidates_by_injection_overlap",
                 "FEATURES_V2", "load_cnn", "apply_cnn", "build_net", "CNN_DEFAULT_THR"):
        assert getattr(inf, name) is not None


def test_apply_cnn_scores_and_thresholds():
    torch = pytest.importorskip("torch")
    from ADCNN.inference.cnn_postproc import build_net, apply_cnn, make_cutouts, CUTOUT_K
    rng = np.random.default_rng(0)
    H = W = 128
    img = rng.standard_normal((H, W)).astype(np.float32)
    prob = rng.random((H, W)).astype(np.float32)
    agg = rng.random((H, W)).astype(np.float32)
    cand = pd.DataFrame({"x_centroid": [30.0, 90.0, 64.0], "y_centroid": [40.0, 64.0, 100.0]})

    X = make_cutouts(cand, img, prob, agg)
    assert X.shape == (3, 3, CUTOUT_K, CUTOUT_K)
    assert np.isfinite(X).all() and X.min() >= -20 and X.max() <= 20

    net = build_net().eval()
    scored = apply_cnn(cand, net, img, prob, agg, device="cpu")
    assert "score" in scored.columns and len(scored) == 3
    s = scored["score"].to_numpy()
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()

    kept = apply_cnn(cand, net, img, prob, agg, thr=1.01, device="cpu")  # impossible cut
    assert len(kept) == 0


def test_focal_trainer_runs_one_panelless_fit():
    torch = pytest.importorskip("torch")
    from ADCNN.training.cnn_postproc import train_cnn, CUTOUT_K
    rng = np.random.default_rng(1)
    n = 64
    X = rng.standard_normal((n, 3, CUTOUT_K, CUTOUT_K)).astype(np.float32)
    y = (rng.random(n) < 0.4).astype(np.int8)
    panel = rng.integers(0, 5, n)
    net, info = train_cnn(X, y, panel, epochs=1, device="cpu")
    assert "n_train" in info and info["n_train"] > 0
    # the fitted net scores in [0, 1]
    with torch.no_grad():
        s = torch.sigmoid(net(torch.tensor(np.clip(X[:8], -20, 20)))).numpy()
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
