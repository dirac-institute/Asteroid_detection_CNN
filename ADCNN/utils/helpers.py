from __future__ import annotations

import random
from typing import Tuple

import h5py
import math
import numpy as np

try:
    import cv2  # optional (only used by draw_one_line)
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def worker_init_fn(worker_id: int, base_seed: int = 1337) -> None:
    """DataLoader worker seeding hook: seeds Python/NumPy/(PyTorch) per worker as
    base_seed + worker_id, so each worker draws a distinct, reproducible stream."""
    seed = int(base_seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def split_indices(h5_path: str, val_frac: float = 0.1, seed: int = 1337) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split by panel/image index: returns (train_panel_ids, val_panel_ids) as sorted int arrays.
    """
    if not (0.0 <= float(val_frac) <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got {val_frac}")

    with h5py.File(str(h5_path), "r") as f:
        if "images" not in f:
            raise KeyError(f"H5 missing dataset 'images': {h5_path}")
        n_panels = int(f["images"].shape[0])

    idx = np.arange(n_panels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)

    split = int((1.0 - float(val_frac)) * n_panels)
    tr = np.sort(idx[:split])
    va = np.sort(idx[split:])
    return tr, va


def draw_one_line(
    mask: np.ndarray,
    origin: Tuple[float, float],
    angle_deg: float,
    length: float,
    true_value: int = 1,
    line_thickness: int = 3,
) -> np.ndarray:
    """
    Draw a thick line into `mask` (H,W) and set pixels to `true_value`.
    Uses OpenCV if available.

    Notes:
      - origin is (x, y) in pixel coordinates.
      - angle_deg: 0 deg is +x direction (OpenCV coordinates).
    """
    if cv2 is None:
        raise RuntimeError("cv2 is not available but draw_one_line() was called.")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    H, W = mask.shape
    x0, y0 = float(origin[0]), float(origin[1])

    # clip origin inside image
    x0 = float(np.clip(x0, 0.0, W - 1.0))
    y0 = float(np.clip(y0, 0.0, H - 1.0))

    ang = np.deg2rad(float(angle_deg))
    dx = float(length) * float(np.cos(ang))
    dy = float(length) * float(np.sin(ang))

    x1 = x0 - dx / 2.0
    y1 = y0 - dy / 2.0
    x2 = x0 + dx / 2.0
    y2 = y0 + dy / 2.0

    tmp = np.zeros((H, W), dtype=np.uint8)
    cv2.line(
        tmp,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        color=1,
        thickness=int(line_thickness),
        lineType=cv2.LINE_8,
    )
    mask[tmp != 0] = true_value
    return mask


def to_panel_dict(arr):
    """Normalise an ``(N, H, W)`` array (or ``{panel_id: (H, W)}`` mapping) to a
    ``{int panel_id: ndarray}`` dict. Shared by the stage-2 cutout-CNN feature extraction
    and the parameter-recovery evaluation so both index panels identically."""
    if isinstance(arr, dict):
        return {int(k): np.asarray(v) for k, v in arr.items()}
    return {int(pid): np.asarray(arr[pid]) for pid in range(len(arr))}


def trail_bbox(x, y, beta_deg, length, H, W, pad=0):
    """Axis-aligned bounding box ``(x0, x1, y0, y1)`` of a trail centred at ``(x, y)``
    with image-convention orientation ``beta_deg`` (0=+x) and full ``length``, padded by
    ``pad`` and clipped to ``[0, W] x [0, H]``. Shared ROI helper for object-confusion,
    parameter-recovery, and stage-2 candidate labelling."""
    beta_rad = math.radians(beta_deg)
    dx = abs(math.cos(beta_rad)) * length
    dy = abs(math.sin(beta_rad)) * length
    x0 = int(max(0, math.floor(x - dx - pad)))
    x1 = int(min(W, math.ceil(x + dx + pad)))
    y0 = int(max(0, math.floor(y - dy - pad)))
    y1 = int(min(H, math.ceil(y + dy + pad)))
    return x0, x1, y0, y1
