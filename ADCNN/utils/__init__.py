"""Utility helpers for the diffim pipeline.

  helpers.draw_one_line  — rasterise a trail segment (used by truth-mask + evaluation)
  angle_utils            — radian/degree conversions + angle normalisation
"""
from .helpers import set_seed, worker_init_fn, make_worker_init_fn, split_indices, draw_one_line
from .angle_utils import deg2rad, rad2deg, ensure_radians, normalize_angle_rad, normalize_angle_deg

__all__ = [
    "set_seed", "worker_init_fn", "make_worker_init_fn", "split_indices", "draw_one_line",
    "deg2rad", "rad2deg", "ensure_radians", "normalize_angle_rad", "normalize_angle_deg",
]
