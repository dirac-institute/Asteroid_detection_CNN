"""Core model architectures for the diffim asteroid-trail detector.

- ``UNetResSE``: the U-Net backbone with residual squeeze-excite blocks.
- ``UNetResSEOrientHough``: the production segmentation model -- ``UNetResSE`` backbone
  + per-pixel orientation head + ``LineAggregator`` (Hough-style) head.
"""
from .model import UNetResSE
from .detector import UNetResSEOrientHough, LineAggregator

__all__ = ["UNetResSE", "UNetResSEOrientHough", "LineAggregator"]
