"""Evaluation utilities for the ADCNN detector.

Catalog-based evaluation is the production path: match a measured detection catalog
against a truth catalog and compute object-level metrics.

    from ADCNN.evaluation import evaluate_catalog, match_trail_catalogs

The plot helpers (``print_confusion_matrix``, ``plot_detect_hist``, ``plot_completeness_2d``)
are the notebook visualisations used by ``Evaluation/Evaluation.ipynb`` and
``Evaluation/Evaluation_Real.ipynb``.
"""
from .geometry import label_components, create_disk_mask, create_line_mask
from .catalog_match import (evaluate_catalog, match_pairs, match_trail_catalogs,
                            stack_sigma_catalog, dedup_within_panel, dedup_cross_catalog)
from .plots import plot_detect_hist, print_confusion_matrix, plot_completeness_2d

__all__ = [
    "label_components", "create_disk_mask", "create_line_mask",
    "evaluate_catalog", "match_pairs", "match_trail_catalogs",
    "stack_sigma_catalog", "dedup_within_panel", "dedup_cross_catalog",
    "plot_detect_hist", "print_confusion_matrix", "plot_completeness_2d",
]
