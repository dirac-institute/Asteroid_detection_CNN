"""Inference utilities for the trained two-stage detector.

The canonical pipeline:

    Stage 1  NN sliding-window inference   -> predict_panel_overlap_3ch_full
    Stage 2  candidate extraction          -> extract_panel_candidates
             focal cutout-CNN FP filter    -> apply_cnn

Names are resolved lazily (PEP 562) so ``import ADCNN`` / ``import ADCNN.inference`` stay cheap
and never eagerly pull torch / cv2. Submodules remain importable directly and are unchanged;
this is an additive discoverability layer:

    from ADCNN.inference import (
        predict_panel_overlap_3ch_full,                                  # predict
        extract_panel_candidates, label_candidates_by_injection_overlap, # features
        load_cnn, apply_cnn, build_net, CNN_DEFAULT_THR,                 # cnn_postproc
    )
"""

import importlib

# public name -> defining submodule (relative to this package)
_LAZY = {
    "predict_panel_overlap_3ch_full":         ".predict",
    "extract_panel_candidates":               ".features",
    "label_candidates_by_injection_overlap":  ".features",
    "load_cnn":                               ".cnn_postproc",
    "apply_cnn":                              ".cnn_postproc",
    "build_net":                              ".cnn_postproc",
    "CNN_DEFAULT_THR":                        ".cnn_postproc",
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
