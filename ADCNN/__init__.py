"""Asteroid Detection CNN (ADCNN) — diffim asteroid-trail detection pipeline.

Production package for detecting asteroid trails in LSST difference images:
- ``core``       : model architectures (UNetResSEOrientHough = v7, the production model)
- ``data``       : diffim dataset loading + preprocessing; ``data.dataset_creation`` builds
                   simulated (injected) and real test datasets from the Butler
- ``training``   : v7 training loop + EMA
- ``inference``  : v7 prediction, candidate extraction, matched-filter features, the
                   RandomForest second-stage post-processor, TorchScript export
- ``evaluation`` : object/pixel detection metrics, geometry, real-data evaluation
- ``pipelines``  : end-to-end entry points (data production, training, inference)

The deployed model (reg2: v7 with lambda_orient=0 + dropout + weight-decay +
intensity-augmentation, plus the neg5 RandomForest) lives in the top-level ``models/``.
"""
__version__ = "3.0.0"

from .core import UNetResSE, UNetResSEASPP, UNetResSEOrientHough, LineAggregator
from .data import DiffimRandomCropDataset3ch, DiffimConcatDataset, build_3channel
from .training import EMAModel

__all__ = [
    "UNetResSE", "UNetResSEASPP", "UNetResSEOrientHough", "LineAggregator",
    "DiffimRandomCropDataset3ch", "DiffimConcatDataset", "build_3channel",
    "EMAModel",
]
