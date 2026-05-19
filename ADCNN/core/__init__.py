"""
Core module containing model architecture, configuration, and losses.
"""

from .model import UNetResSE, UNetResSEASPP
# v7 diffim model — the architecture behind the promoted real/synthetic result.
from .diffim_model import UNetResSEOrientHough, LineAggregator
from .config import Config, DataConfig, LoaderConfig, ModelConfig, TrainConfig

__all__ = [
    "UNetResSE",
    "UNetResSEASPP",
    "UNetResSEOrientHough",
    "LineAggregator",
    "Config",
    "DataConfig",
    "LoaderConfig",
    "ModelConfig",
    "TrainConfig",
]

