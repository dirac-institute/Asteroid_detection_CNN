from __future__ import annotations

import numpy as np


def to_panel_dict(arr):
    """Normalise an ``(N, H, W)`` array (or ``{panel_id: (H, W)}`` mapping) to a
    ``{int panel_id: ndarray}`` dict."""
    if isinstance(arr, dict):
        return {int(k): np.asarray(v) for k, v in arr.items()}
    return {int(pid): np.asarray(arr[pid]) for pid in range(len(arr))}
