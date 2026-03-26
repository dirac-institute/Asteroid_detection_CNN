"""
Asteroid Detection CNN - Refactored and Clean

Organized module structure:
- core: Model architecture, configuration, losses
- data: Dataset loading and preprocessing
- evaluation: Metrics, detection evaluation, threshold scanning
- training: Training utilities (EMA, phases)
- utils: Helper functions (angles, distributed, misc)
- inference: Model inference utilities
"""

__version__ = "2.0.0"

# Core exports
from .core import (
    UNetResSE,
    UNetResSEASPP,
    Config,
)

# Data exports
from .data import (
    H5TiledDataset,
)

# Evaluation exports
from .evaluation import (
    masked_pixel_auc,
    resize_masks_to,
    objectwise_confusion,
    pixelwise_confusion,
)

# Training exports
from .training import (
    EMAModel,
)

# Utils exports
from .utils import (
    deg2rad,
    rad2deg,
    set_seed,
    init_distributed,
    is_main_process,
)

__all__ = [
    # Core
    "UNetResSE",
    "UNetResSEASPP",
    "Config",
    # Data
    "H5TiledDataset",
    # Evaluation
    "masked_pixel_auc",
    "resize_masks_to",
    "objectwise_confusion",
    "pixelwise_confusion",
    # Training
    "EMAModel",
    # Utils
    "deg2rad",
    "rad2deg",
    "set_seed",
    "init_distributed",
    "is_main_process",
]

