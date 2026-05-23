"""Inference utilities for trained models.

This package exposes the canonical two-stage diffim pipeline that produced
the promoted synthetic + real-asteroid result:

    Stage 1  NN sliding-window inference  -> predict_panel_overlap_3ch_full
    Stage 2  72-feature RandomForest rerank -> compute_v2_features
                                              -> apply_rf_v2
                                              -> materialize_label_mask_v2

Names are resolved lazily (PEP 562) so ``import ADCNN`` / ``import
ADCNN.inference`` stay cheap and never eagerly pull torch / sklearn / cv2.
The submodules remain importable directly and are unchanged; this is an
additive, backward-compatible discoverability layer:

    from ADCNN.inference import (
        predict_panel_overlap_3ch_full,        # diffim_eval
        compute_v2_features, apply_rf_v2,      # rf_postproc
        materialize_label_mask_v2, load_rf, save_rf,
        build_rf_postproc_v2, train_rf_v2, rf_score_sweep,
        RF_FEATURES_V2, DEFAULT_THR,
    )
"""

import importlib

# public name -> defining submodule (relative to this package)
_LAZY = {
    "predict_panel_overlap_3ch_full": ".diffim_eval",
    "compute_v2_features":            ".rf_postproc",
    "apply_rf_v2":                    ".rf_postproc",
    "materialize_label_mask_v2":      ".rf_postproc",
    "build_rf_postproc_v2":           ".rf_postproc",
    "train_rf_v2":                    ".rf_postproc",
    "rf_score_sweep":                 ".rf_postproc",
    "load_rf":                        ".rf_postproc",
    "save_rf":                        ".rf_postproc",
    "RF_FEATURES_V2":                 ".rf_postproc",
    "DEFAULT_THR":                    ".rf_postproc",
}

__all__ = sorted(_LAZY)


def __getattr__(name):  # PEP 562 — resolve on first access only.
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(target, __name__)
    obj = getattr(mod, name)
    globals()[name] = obj  # cache so subsequent access is a plain lookup
    return obj


def __dir__():
    return sorted(set(__all__) | set(globals()))
