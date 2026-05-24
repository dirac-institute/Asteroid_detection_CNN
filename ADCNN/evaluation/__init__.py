"""Evaluation utilities for asteroid detection.

Catalog-based evaluation (the production path) lives in ``catalog_match`` — match a measured
detection catalog against a truth catalog and compute object-level metrics:

    from ADCNN.evaluation.catalog_match import evaluate_catalog, match_pairs

``detection`` provides the mask-based object-level confusion still used by real_eval /
fp_analysis (``objectwise_confusion``, ``combined_objectwise_confusion_separate``) and the
notebook plot helpers (``print_confusion_matrix``, ``plot_detect_hist``, ``plot_completeness_2d``).
``geometry`` holds the shared mask/component primitives. (Per-pixel inference statistics —
pixelwise confusion, pixel AUC, threshold/ROC/FROC scans, map-based parameter recovery — were
removed once evaluation moved to the catalog approach.)
"""
from .geometry import label_components, create_disk_mask, create_line_mask
from .detection import objectwise_confusion, combined_objectwise_confusion_separate
from .catalog_match import evaluate_catalog, match_pairs, match_trail_catalogs

__all__ = [
    "label_components", "create_disk_mask", "create_line_mask",
    "objectwise_confusion", "combined_objectwise_confusion_separate",
    "evaluate_catalog", "match_pairs", "match_trail_catalogs",
]
