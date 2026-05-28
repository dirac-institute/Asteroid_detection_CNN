"""Asteroid Detection CNN (ADCNN) — diffim asteroid-trail detection pipeline.

Production package for detecting asteroid trails in LSST difference images:
- ``core``       : model architectures (UNetResSEOrientHough = segmentation model, the production model)
- ``data``       : diffim dataset loading + preprocessing; ``data.dataset_creation`` builds
                   simulated (injected) and real test datasets from the Butler
- ``training``   : segmentation model training loop + EMA; ``training.cnn_postproc`` trains the stage-2 CNN
- ``inference``  : segmentation model prediction, candidate extraction, matched-filter measurement, the focal
                   cutout-CNN second-stage false-positive filter, TorchScript export
- ``evaluation`` : object detection metrics, geometry, catalog-based evaluation
- ``pipelines``  : end-to-end entry points (data production, training, inference)

The deployed model (reg2: segmentation model with lambda_orient=0 + dropout + weight-decay +
intensity-augmentation, plus the focal cutout CNN) lives in the top-level ``models/``.
"""
__version__ = "3.0.0"

from .core import UNetResSE, UNetResSEOrientHough, LineAggregator
from .data import DiffimRandomCropDataset3ch, DiffimConcatDataset, build_3channel
from .training import EMAModel

__all__ = [
    "UNetResSE", "UNetResSEOrientHough", "LineAggregator",
    "DiffimRandomCropDataset3ch", "DiffimConcatDataset", "build_3channel",
    "EMAModel",
]
