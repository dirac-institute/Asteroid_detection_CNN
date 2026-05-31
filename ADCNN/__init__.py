"""Asteroid Detection CNN (ADCNN) — diffim asteroid-trail detection pipeline.

Production package for detecting asteroid trails in LSST difference images:

- ``core``       — model architectures (``UNetResSEOrientHough``: the production segmentation model)
- ``data``       — diffim dataset loading + preprocessing; ``data.dataset_creation`` builds
                   the simulated (injected) and real test datasets from the Butler
- ``training``   — segmentation training loop + EMA; ``training.cnn_postproc`` trains the
                   stage-2 focal-loss cutout CNN
- ``inference``  — segmentation prediction, candidate extraction, matched-filter measurement,
                   the cutout-CNN false-positive filter, TorchScript export
- ``evaluation`` — catalog-based detection metrics, geometry primitives, notebook plotting
- ``pipelines``  — end-to-end entry points (dataset build, training, inference, evaluation, linking)

Deployed weights live in the top-level ``models/`` directory.
"""
__version__ = "1.0.0"

from .core import UNetResSE, UNetResSEOrientHough, LineAggregator
from .data import DiffimRandomCropDataset3ch, DiffimConcatDataset, build_3channel
from .training import EMAModel

__all__ = [
    "UNetResSE", "UNetResSEOrientHough", "LineAggregator",
    "DiffimRandomCropDataset3ch", "DiffimConcatDataset", "build_3channel",
    "EMAModel",
]
