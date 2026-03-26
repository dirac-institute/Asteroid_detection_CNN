from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from ADCNN.evaluation.geometry import create_line_mask


class SoftMaskGenerator:
    """
    Lazily generate soft per-panel truth masks from CSV trail geometry.

    The soft target is built by drawing the trail centerline and applying a local
    Gaussian spread inside a compact ROI around each object. Per-panel masks are
    cached in memory, and can optionally be persisted on disk for reuse.
    """

    def __init__(
        self,
        *,
        csv_path: str,
        image_shape: tuple[int, int],
        sigma_pix: float = 2.0,
        line_width: int = 1,
        truncate: float = 4.0,
        cache_dir: Optional[str] = None,
        cache_size: int = 8,
        dtype: str = "float16",
    ):
        self.csv_path = str(csv_path)
        self.H, self.W = map(int, image_shape)
        self.sigma_pix = float(sigma_pix)
        self.line_width = max(1, int(line_width))
        self.truncate = float(truncate)
        self.cache_size = max(1, int(cache_size))
        self.cache_dtype = np.dtype(dtype)

        self.cache_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        cat = pd.read_csv(self.csv_path)
        need = {"image_id", "x", "y", "beta", "trail_length"}
        miss = need - set(cat.columns)
        if miss:
            raise ValueError(f"CSV missing required columns for soft masks: {sorted(miss)}")

        self.groups: dict[int, pd.DataFrame] = {
            int(img_id): df[["x", "y", "beta", "trail_length"]].copy()
            for img_id, df in cat.groupby("image_id", sort=False)
        }
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _cache_path(self, image_id: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"soft_mask_panel_{int(image_id):06d}.npy"

    def _load_from_disk(self, image_id: int) -> Optional[np.ndarray]:
        path = self._cache_path(image_id)
        if path is None or not path.exists():
            return None
        arr = np.load(path, mmap_mode=None)
        return np.asarray(arr, dtype=np.float32)

    def _save_to_disk(self, image_id: int, arr: np.ndarray) -> None:
        path = self._cache_path(image_id)
        if path is None:
            return
        np.save(path, np.asarray(arr, dtype=self.cache_dtype))

    def _remember(self, image_id: int, arr: np.ndarray) -> np.ndarray:
        self._cache[int(image_id)] = np.asarray(arr, dtype=np.float32)
        self._cache.move_to_end(int(image_id))
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return self._cache[int(image_id)]

    def _object_soft_roi(self, *, xc: float, yc: float, beta_deg: float, length: float) -> tuple[slice, slice, np.ndarray]:
        pad = int(np.ceil(self.truncate * self.sigma_pix)) + self.line_width + 2
        th = np.deg2rad(float(beta_deg))
        dx = abs(np.cos(th)) * float(length) / 2.0
        dy = abs(np.sin(th)) * float(length) / 2.0

        x0 = int(max(0, np.floor(float(xc) - dx - pad)))
        x1 = int(min(self.W, np.ceil(float(xc) + dx + pad + 1)))
        y0 = int(max(0, np.floor(float(yc) - dy - pad)))
        y1 = int(min(self.H, np.ceil(float(yc) + dy + pad + 1)))
        if x1 <= x0 or y1 <= y0:
            return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=np.float32)

        roi = create_line_mask(
            (y1 - y0, x1 - x0),
            yc=float(yc) - y0,
            xc=float(xc) - x0,
            beta_deg=float(beta_deg),
            length=float(length),
            width=int(self.line_width),
        ).astype(np.float32)

        soft = ndi.gaussian_filter(
            roi,
            sigma=float(self.sigma_pix),
            mode="constant",
            truncate=float(self.truncate),
        )
        peak = float(soft.max())
        if peak > 0.0:
            soft /= peak
        return slice(y0, y1), slice(x0, x1), soft.astype(np.float32, copy=False)

    def _build_panel_mask(self, image_id: int) -> np.ndarray:
        rows = self.groups.get(int(image_id))
        panel = np.zeros((self.H, self.W), dtype=np.float32)
        if rows is None or len(rows) == 0:
            return panel

        for row in rows.itertuples(index=False):
            ys, xs, soft = self._object_soft_roi(
                xc=float(row.x),
                yc=float(row.y),
                beta_deg=float(row.beta),
                length=float(row.trail_length),
            )
            if soft.size == 0:
                continue
            panel[ys, xs] = np.maximum(panel[ys, xs], soft)

        return panel

    def panel_mask(self, image_id: int) -> np.ndarray:
        image_id = int(image_id)
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
            return self._cache[image_id]

        arr = self._load_from_disk(image_id)
        if arr is None:
            arr = self._build_panel_mask(image_id)
            self._save_to_disk(image_id, arr)

        return self._remember(image_id, arr)
