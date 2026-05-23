"""Diffim dataset loading + normalisation.

``DiffimRandomCropDataset3ch`` (single h5) and ``DiffimConcatDataset`` (multi-h5) feed
128px 3-channel tiles (MAD-normalised diffim + local-std + DIA-mask) with sin/cos
orientation supervision. ``build_3channel`` / ``diffim_mad_sigma`` are the shared
preprocessing primitives.
"""
from .dataset import (
    DiffimRandomCropDataset3ch,
    DiffimConcatDataset,
    build_3channel,
    diffim_mad_sigma,
)

__all__ = [
    "DiffimRandomCropDataset3ch",
    "DiffimConcatDataset",
    "build_3channel",
    "diffim_mad_sigma",
]
