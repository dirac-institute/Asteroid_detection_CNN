"""
Geometry utilities for evaluation.

Includes:
- Component labeling (connected components with dilation)
- Mask creation (disk, line/trail)
"""

import numpy as np
import scipy.ndimage as ndi


def label_components(mask_bool: np.ndarray, pixel_gap: int = 3) -> tuple[np.ndarray, int]:
    """
    Label connected components with optional dilation to bridge small gaps.

    Args:
        mask_bool: Binary mask (H, W)
        pixel_gap: Dilation radius to bridge gaps between components

    Returns:
        labels: Integer label map (H, W), 0=background
        n: Number of components found
    """
    if pixel_gap > 1:
        grown = ndi.binary_dilation(
            mask_bool,
            structure=np.ones((2*pixel_gap+1, 2*pixel_gap+1), bool)
        )
    else:
        grown = mask_bool

    labels, n = ndi.label(grown, structure=np.ones((3, 3), bool))  # 8-connectivity
    return labels, int(n)


def create_disk_mask(shape: tuple[int, int], yc: float, xc: float, radius: float) -> np.ndarray:
    """
    Create circular mask.

    Args:
        shape: (H, W) of output mask
        yc, xc: Center coordinates
        radius: Radius in pixels

    Returns:
        Boolean mask (H, W)
    """
    H, W = shape
    y, x = np.ogrid[:H, :W]
    return (y - yc)**2 + (x - xc)**2 <= radius**2


def create_line_mask(
    shape: tuple[int, int],
    yc: float,
    xc: float,
    beta_deg: float,
    length: float,
    width: int = 1
) -> np.ndarray:
    """
    Create line/trail mask.

    Args:
        shape: (H, W) of output mask
        yc, xc: Center coordinates
        beta_deg: Angle in degrees (0° = +x direction)
        length: Length of trail in pixels
        width: Width of trail (dilation radius)

    Returns:
        Boolean mask (H, W)
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)

    L = float(length) / 2.0
    th = np.deg2rad(float(beta_deg))
    dy, dx = np.sin(th), np.cos(th)

    y0, x0 = float(yc) - L*dy, float(xc) - L*dx
    y1, x1 = float(yc) + L*dy, float(xc) + L*dx

    # Rasterize line
    steps = max(2, int(np.ceil(length * 2)))
    ys = np.clip(np.rint(np.linspace(y0, y1, steps)).astype(int), 0, H-1)
    xs = np.clip(np.rint(np.linspace(x0, x1, steps)).astype(int), 0, W-1)

    mask[ys, xs] = True

    # Optional dilation for width
    if width > 1:
        rad = max(1, int(width // 2))
        mask = ndi.binary_dilation(
            mask,
            structure=np.ones((2*rad+1, 2*rad+1), bool)
        )

    return mask

