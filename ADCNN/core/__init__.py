"""Core model architectures for the diffim asteroid-trail detector.

- ``UNetResSE`` / ``UNetResSEASPP``: the U-Net backbone with residual squeeze-excite
  blocks (the feature extractor reused by the segmentation model head).
- ``UNetResSEOrientHough`` (segmentation model): the production segmentation model — UNetResSE backbone
  + per-pixel orientation head + LineAggregator (Hough) head. This is the architecture
  behind the deployed reg2 result.
"""
from .model import UNetResSE
from .detector import UNetResSEOrientHough, LineAggregator

__all__ = ["UNetResSE", "UNetResSEOrientHough", "LineAggregator"]
