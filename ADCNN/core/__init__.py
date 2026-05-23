"""Core model architectures for the diffim asteroid-trail detector.

- ``UNetResSE`` / ``UNetResSEASPP``: the U-Net backbone with residual squeeze-excite
  blocks (the feature extractor reused by the v7 head).
- ``UNetResSEOrientHough`` (v7): the production segmentation model — UNetResSE backbone
  + per-pixel orientation head + LineAggregator (Hough) head. This is the architecture
  behind the deployed reg2 result.
"""
from .model import UNetResSE, UNetResSEASPP
from .diffim_model import UNetResSEOrientHough, LineAggregator

__all__ = ["UNetResSE", "UNetResSEASPP", "UNetResSEOrientHough", "LineAggregator"]
