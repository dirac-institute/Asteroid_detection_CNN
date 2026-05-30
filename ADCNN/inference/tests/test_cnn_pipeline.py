"""Smoke tests for the stage-2 cutout-CNN pipeline wiring (CPU-only, no model files needed).

Verifies the canonical pipeline is internally consistent: the catalog schema uses ``score``,
the config exposes ``cnn_thr``, the lean candidate extractor + cutout CNN run end to end on a
fresh (untrained) net, and the trainer's focal loss + cutout extraction wire up cleanly.
"""
import numpy as np
import pandas as pd
import pytest


def test_catalog_schema_is_cnn():
    from ADCNN.inference.catalog import CATALOG_COLUMNS, InferenceConfig
    assert "score" in CATALOG_COLUMNS
    assert "score_rf" not in CATALOG_COLUMNS
    assert "beta_nn" not in CATALOG_COLUMNS  # legacy NN-orient diagnostic, dropped
    cfg = InferenceConfig()
    assert hasattr(cfg, "cnn_thr") and not hasattr(cfg, "rf_thr")


def test_no_rf_modules_remain():
    import importlib
    for mod in ("ADCNN.inference.rf_postproc", "ADCNN.inference.rf_train"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def test_inference_exports_resolve():
    import ADCNN.inference as inf
    for name in ("extract_panel_candidates", "label_candidates_by_injection_overlap",
                 "load_cnn", "apply_cnn", "build_net", "CNN_DEFAULT_THR",
                 "predict_panel_overlap_3ch_full"):
        assert getattr(inf, name) is not None


def test_apply_cnn_scores_and_thresholds():
    torch = pytest.importorskip("torch")
    from ADCNN.inference.cnn_postproc import build_net, apply_cnn, make_cutouts, CUTOUT_K
    rng = np.random.default_rng(0)
    H = W = 256
    img = rng.standard_normal((H, W)).astype(np.float32)
    prob = rng.random((H, W)).astype(np.float32)
    agg = rng.random((H, W)).astype(np.float32)
    cand = pd.DataFrame({"x_centroid": [60.0, 180.0, 128.0], "y_centroid": [80.0, 128.0, 200.0]})

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
    from ADCNN.training.cnn_postproc import train_cnn
    from ADCNN.inference.cnn_postproc import CUTOUT_K
    rng = np.random.default_rng(1)
    n = 64
    X = rng.standard_normal((n, 3, CUTOUT_K, CUTOUT_K)).astype(np.float32)
    y = (rng.random(n) < 0.4).astype(np.int8)
    panel = rng.integers(0, 5, n)
    state, info = train_cnn(X, y, panel, epochs=1, device="cpu", augment=False, cosine_lr=False)
    assert "n_train" in info and info["n_train"] > 0
    # the fitted net scores in [0, 1]
    from ADCNN.inference.cnn_postproc import build_net
    fresh = build_net(width=info["width"], depth=info["depth"], in_ch=info["in_ch"], k=info["k"]).eval()
    fresh.load_state_dict(state)
    with torch.no_grad():
        s = torch.sigmoid(fresh(torch.tensor(np.clip(X[:8], -20, 20)))).numpy()
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
