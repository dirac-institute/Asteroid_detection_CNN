"""
Evaluation module for asteroid detection.

Consolidated evaluation utilities including:
- Pixel-level and object-level metrics
- ROC/AUC computation
- Detection evaluation
- Threshold scanning
"""

from .metrics import (
    masked_pixel_auc,
    resize_masks_to,
    precision,
    recall,
    f1_score,
    f2_score,
)
from .geometry import (
    label_components,
    create_disk_mask,
    create_line_mask,
)
from .detection import (
    objectwise_confusion,
    mark_detections,
    pixelwise_confusion,
)

__all__ = [
    # Metrics
    "masked_pixel_auc",
    "resize_masks_to",
    "precision",
    "recall",
    "f1_score",
    "f2_score",
    # Geometry
    "label_components",
    "create_disk_mask",
    "create_line_mask",
    # Detection
    "objectwise_confusion",
    "mark_detections",
    "pixelwise_confusion",
]

