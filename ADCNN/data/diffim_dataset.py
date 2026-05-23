"""v5 dataset: 3-channel input on top of v4's random-crop + orient supervision.

Channels (computed in __getitem__, no regeneration needed):
  ch0 = signed diffim MAD-normalized, clipped to ±5    (same as v4)
  ch1 = local standard deviation of ch0, window 11
            (gives per-pixel noise context — bright residuals, chip edges,
            saturation cores have high local std; flat backgrounds have low.
            The model can learn to suppress firing on high-local-std regions
            that aren't actually a thin oriented line.)
  ch2 = real_labels > 0 (binary)                       (artefact / DIA mask)

Orientation supervision and ignore mask: identical to v4.
"""
from __future__ import annotations

import math, time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def diffim_mad_sigma(arr: np.ndarray) -> float:
    """Robust noise scale of a zero-mean diffim. median(|x|) * 1.4826."""
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(good)) + 1e-8)


def _panel_orient_maps(masks_panel: np.ndarray, csv_panel) -> tuple:
    """Per-pixel sin(2β), cos(2β) for a panel, derived from its truth mask
    and the per-injection β. Each truth pixel takes the β of the nearest
    injection in (x,y)."""
    H, W = masks_panel.shape
    sin_map = np.zeros((H, W), dtype=np.float32)
    cos_map = np.zeros((H, W), dtype=np.float32)
    if len(csv_panel) == 0 or not masks_panel.any():
        return sin_map, cos_map
    ys, xs = np.nonzero(masks_panel)
    inj_xs = csv_panel["x"].to_numpy().astype(np.float32)
    inj_ys = csv_panel["y"].to_numpy().astype(np.float32)
    inj_betas = csv_panel["beta"].to_numpy().astype(np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.stack([inj_xs, inj_ys], axis=1))
        _, idx = tree.query(np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1), k=1)
    except Exception:
        idx = np.empty_like(xs)
        for k in range(xs.size):
            dd = (inj_xs - xs[k]) ** 2 + (inj_ys - ys[k]) ** 2
            idx[k] = int(np.argmin(dd))
    betas_assigned = inj_betas[idx]
    sin_vals = np.sin(np.radians(2.0 * betas_assigned)).astype(np.float32)
    cos_vals = np.cos(np.radians(2.0 * betas_assigned)).astype(np.float32)
    sin_map[ys, xs] = sin_vals
    cos_map[ys, xs] = cos_vals
    return sin_map, cos_map


def local_std_panel(arr: np.ndarray, window: int = 11) -> np.ndarray:
    """Per-pixel std over a window-by-window box centered on the pixel.

    var = E[x^2] - E[x]^2 (clipped to 0). Uses uniform_filter for speed.
    """
    from scipy.ndimage import uniform_filter
    a = arr.astype(np.float32)
    mu = uniform_filter(a, size=window, mode="nearest")
    sq = uniform_filter(a * a, size=window, mode="nearest")
    var = np.clip(sq - mu * mu, 0.0, None)
    return np.sqrt(var).astype(np.float32)


def normalize_local_std(local_std: np.ndarray) -> np.ndarray:
    """Map local_std into [0, 1]-ish for use as a NN channel.

    The diffim is already MAD-normalized (clipped to ±5), so its local std
    is bounded by ~5/sqrt(2) ≈ 3.5 for noise-only regions and much higher
    near saturated cores. We log1p-compress and clip.
    """
    return np.clip(np.log1p(local_std) / np.log1p(5.0), 0.0, 1.0).astype(np.float32)


def build_3channel(diffim_tile: np.ndarray, real_labels_tile: np.ndarray,
                   *, panel_sigma: float, clip: float = 5.0) -> np.ndarray:
    """Build the 3-channel input from one tile. Returns (3, H, W) float32."""
    z = np.clip(diffim_tile.astype(np.float32) / panel_sigma, -clip, clip)
    lstd = local_std_panel(z, window=11)
    lstd_n = normalize_local_std(lstd)
    rl = (real_labels_tile > 0).astype(np.float32)
    return np.stack([z.astype(np.float32), lstd_n, rl], axis=0)


class DiffimRandomCropDataset3ch(Dataset):
    def __init__(
        self,
        h5_path: str,
        csv_df: pd.DataFrame,
        panel_ids: Iterable[int],
        *,
        tile: int = 128,
        clip: float = 5.0,
        stats_crop: int = 1024,
        n_pos_anchors_per_epoch: int = 1800,
        n_neg_anchors_per_epoch: int = 600,
        stk_balance: float = 0.6,
        anchor_jitter: int = 48,
        orient_cache_size: int = 24,
        seed: int = 0,
        augment: bool = False,
    ):
        self.h5_path = str(h5_path)
        self.tile = int(tile)
        self.clip = float(clip)
        self.stats_crop = int(stats_crop)
        self.n_pos = int(n_pos_anchors_per_epoch)
        self.n_neg = int(n_neg_anchors_per_epoch)
        self.stk_balance = float(stk_balance)
        self.anchor_jitter = int(anchor_jitter)
        self.seed = int(seed)
        # D4 dihedral augmentation (train only): trails have no preferred
        # orientation, so flips / 90deg rotations are label-preserving once the
        # sin(2b)/cos(2b) orientation maps are transformed accordingly.
        self.augment = bool(augment)
        # intensity/noise augmentation (train only): rescale + sigma-matched noise to
        # vary effective SNR. Set via .intensity_aug after construction (trainer flag).
        self.intensity_aug = False
        self._epoch = 0

        with h5py.File(self.h5_path, "r") as f:
            shape = f["images"].shape
            self.N, self.H, self.W = int(shape[0]), int(shape[1]), int(shape[2])
        self.panel_ids = sorted(set(int(i) for i in panel_ids))

        self.csv_full = csv_df.copy()
        self.csv = csv_df[csv_df["image_id"].isin(self.panel_ids)].copy()

        # Anchor pools.
        pos_det = self.csv[self.csv["stack_detection"].astype(bool)]
        pos_miss = self.csv[~self.csv["stack_detection"].astype(bool)]
        self._pos_det_anchors = np.array(
            list(zip(pos_det["image_id"], pos_det["y"], pos_det["x"])),
            dtype=np.int64,
        ) if len(pos_det) else np.empty((0, 3), dtype=np.int64)
        self._pos_miss_anchors = np.array(
            list(zip(pos_miss["image_id"], pos_miss["y"], pos_miss["x"])),
            dtype=np.int64,
        ) if len(pos_miss) else np.empty((0, 3), dtype=np.int64)

        self._h5 = None
        self._images = None
        self._masks = None
        self._real = None

        self._sigma_cache: dict[int, float] = {}
        self._orient_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._orient_cache_size = int(orient_cache_size)

        self.regenerate_anchors(epoch=0)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=False)
            self._images = self._h5["images"]
            self._masks = self._h5["masks"]
            self._real = self._h5["real_labels"] if "real_labels" in self._h5 else None

    def _sigma_of(self, pid: int) -> float:
        if pid in self._sigma_cache:
            return self._sigma_cache[pid]
        self._ensure_open()
        s = min(self.stats_crop, self.H, self.W)
        h0 = (self.H - s) // 2
        w0 = (self.W - s) // 2
        crop = self._images[pid, h0:h0 + s, w0:w0 + s].astype(np.float32, copy=False)
        sig = diffim_mad_sigma(crop)
        self._sigma_cache[pid] = sig
        return sig

    def _orient_of(self, pid: int) -> tuple[np.ndarray, np.ndarray]:
        if pid in self._orient_cache:
            self._orient_cache.move_to_end(pid)
            return self._orient_cache[pid]
        self._ensure_open()
        msk = self._masks[pid][:]
        csv_panel = self.csv_full[self.csv_full["image_id"] == pid]
        sin_map, cos_map = _panel_orient_maps(msk, csv_panel)
        self._orient_cache[pid] = (sin_map, cos_map)
        while len(self._orient_cache) > self._orient_cache_size:
            self._orient_cache.popitem(last=False)
        return self._orient_cache[pid]

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self.regenerate_anchors(epoch=epoch)

    def regenerate_anchors(self, epoch: int):
        rng = np.random.default_rng(self.seed + 7919 * int(epoch))
        n_pos_miss = int(round(self.n_pos * self.stk_balance))
        n_pos_det = self.n_pos - n_pos_miss
        if len(self._pos_det_anchors):
            det_idx = rng.integers(0, len(self._pos_det_anchors), size=n_pos_det)
            det_chunk = self._pos_det_anchors[det_idx]
        else:
            det_chunk = np.empty((0, 3), dtype=np.int64)
        if len(self._pos_miss_anchors):
            miss_idx = rng.integers(0, len(self._pos_miss_anchors), size=n_pos_miss)
            miss_chunk = self._pos_miss_anchors[miss_idx]
        else:
            miss_chunk = np.empty((0, 3), dtype=np.int64)
        if self.n_neg > 0:
            neg_panels = rng.choice(self.panel_ids, size=self.n_neg)
            neg_y = rng.integers(self.tile // 2, self.H - self.tile // 2, size=self.n_neg)
            neg_x = rng.integers(self.tile // 2, self.W - self.tile // 2, size=self.n_neg)
            neg_chunk = np.stack([neg_panels, neg_y, neg_x], axis=1).astype(np.int64)
        else:
            neg_chunk = np.empty((0, 3), dtype=np.int64)
        anchors = np.concatenate([
            np.concatenate([det_chunk, np.full((len(det_chunk), 1), 1, dtype=np.int64)], axis=1),
            np.concatenate([miss_chunk, np.full((len(miss_chunk), 1), 2, dtype=np.int64)], axis=1),
            np.concatenate([neg_chunk, np.full((len(neg_chunk), 1), 0, dtype=np.int64)], axis=1),
        ], axis=0)
        rng.shuffle(anchors)
        self._epoch_anchors = anchors

    def __len__(self):
        return len(self._epoch_anchors)

    def __getitem__(self, idx: int):
        self._ensure_open()
        row = self._epoch_anchors[idx]
        pid, anchor_y, anchor_x, anchor_type = int(row[0]), int(row[1]), int(row[2]), int(row[3])

        sub_seed = (self._epoch * 1_000_003 + idx) & 0x7fffffff
        rng = np.random.default_rng(sub_seed)
        jy = int(rng.integers(-self.anchor_jitter, self.anchor_jitter + 1))
        jx = int(rng.integers(-self.anchor_jitter, self.anchor_jitter + 1))
        cy = anchor_y + jy
        cx = anchor_x + jx
        t = self.tile

        y0 = max(0, min(self.H - t, cy - t // 2))
        x0 = max(0, min(self.W - t, cx - t // 2))
        y1 = y0 + t
        x1 = x0 + t

        diffim_tile = self._images[pid, y0:y1, x0:x1].astype(np.float32)
        y_seg = self._masks[pid, y0:y1, x0:x1].astype(np.float32)
        rl_tile = (self._real[pid, y0:y1, x0:x1]).astype(np.int32) \
            if self._real is not None else np.zeros_like(y_seg, dtype=np.int32)
        ig = (rl_tile > 0).astype(np.float32)
        sin_map, cos_map = self._orient_of(pid)
        y_sin = sin_map[y0:y1, x0:x1].copy()
        y_cos = cos_map[y0:y1, x0:x1].copy()

        if self.augment:
            # one of the 8 D4 symmetries = rot90^k then optional left-right flip.
            # Spatial op applied identically to every map; orientation (sin2b,cos2b)
            # additionally sign-transforms: rot90^k -> *(-1)^k (both); flip_lr -> sin*=-1.
            k = int(rng.integers(0, 4)); flip = bool(rng.integers(0, 2))

            def _d4(a):
                a = np.rot90(a, k)
                if flip:
                    a = a[:, ::-1]
                return np.ascontiguousarray(a)

            diffim_tile = _d4(diffim_tile)
            y_seg = _d4(y_seg)
            rl_tile = _d4(rl_tile)
            sgn = (-1.0) ** k
            y_sin = _d4(y_sin) * sgn * (-1.0 if flip else 1.0)
            y_cos = _d4(y_cos) * sgn
            ig = (rl_tile > 0).astype(np.float32)

        sig = self._sigma_of(pid)
        if self.intensity_aug:
            # vary effective SNR/background (data-like augmentation): rescale the
            # diffim and add sigma-matched noise BEFORE the panel_sigma normalization,
            # so a trail's apparent SNR shifts -> teaches the faint regime. Labels
            # unchanged (a dimmer/noisier trail is still the same trail).
            diffim_tile = diffim_tile * float(rng.uniform(0.65, 1.5))
            diffim_tile = diffim_tile + rng.normal(
                0.0, float(rng.uniform(0.0, 0.6)) * sig, size=diffim_tile.shape
            ).astype(np.float32)
        x_chans = build_3channel(diffim_tile, rl_tile, panel_sigma=sig, clip=self.clip)

        meta = {
            "panel_id": pid, "anchor_y": cy, "anchor_x": cx,
            "type": anchor_type, "y0": y0, "x0": x0,
        }
        return (
            torch.from_numpy(x_chans),                # (3, T, T)
            torch.from_numpy(y_seg[None, ...]),
            torch.from_numpy(ig[None, ...]),
            torch.from_numpy(y_sin[None, ...]),
            torch.from_numpy(y_cos[None, ...]),
            meta,
        )


def collate_v5(batch):
    xs = torch.stack([b[0] for b in batch], 0)        # (B, 3, T, T)
    ys = torch.stack([b[1] for b in batch], 0)
    igs = torch.stack([b[2] for b in batch], 0)
    sins = torch.stack([b[3] for b in batch], 0)
    coss = torch.stack([b[4] for b in batch], 0)
    metas = [b[5] for b in batch]
    return xs, ys, igs, sins, coss, metas


class DiffimConcatDataset(Dataset):
    """Train on MULTIPLE full-panel h5 datasets at once WITHOUT merging files (no
    storage for a merged copy). Composes one validated DiffimRandomCropDataset3ch per
    (h5,csv) source -- each keeps its own panels/anchors/sigma/orient logic UNCHANGED
    (statistics identical) -- and concatenates them. Each source contributes
    n_pos/n_sources + n_neg/n_sources anchors per epoch so the combined epoch matches
    the requested totals. This is how we use the existing realistic h5s + new ones
    together for 'much more data', losslessly.
    """
    def __init__(self, sources, *, tile=128, clip=5.0,
                 n_pos_anchors_per_epoch=3000, n_neg_anchors_per_epoch=900,
                 stk_balance=0.6, anchor_jitter=48, seed=0, augment=False,
                 intensity_aug=False):
        # Weight each source PROPORTIONAL to its panel count so every panel is sampled
        # at the same rate -> identical to one combined h5. (Equal-per-source would
        # over-sample a smaller shard's panels, e.g. a 685-panel shard vs 1100.)
        dfs = [(h5, pd.read_csv(csv) if isinstance(csv, str) else csv) for h5, csv in sources]
        npans = [int(df["image_id"].nunique()) for _, df in dfs]
        ntot = max(1, sum(npans))
        self.subs = []
        for i, ((h5, df), npan) in enumerate(zip(dfs, npans)):
            frac = npan / ntot
            sub = DiffimRandomCropDataset3ch(
                h5, df, panel_ids=sorted(df["image_id"].unique()), tile=tile, clip=clip,
                n_pos_anchors_per_epoch=max(1, round(n_pos_anchors_per_epoch * frac)),
                n_neg_anchors_per_epoch=max(1, round(n_neg_anchors_per_epoch * frac)),
                stk_balance=stk_balance, anchor_jitter=anchor_jitter,
                seed=seed + 13 * i, augment=augment)
            sub.intensity_aug = bool(intensity_aug)
            self.subs.append(sub)
        self._reindex()

    def _reindex(self):
        self._cum = np.cumsum([0] + [len(s) for s in self.subs])

    def set_epoch(self, e):
        for s in self.subs:
            s.set_epoch(e)
        self._reindex()

    def __len__(self):
        return int(self._cum[-1])

    def __getitem__(self, i):
        src = int(np.searchsorted(self._cum, i, side="right") - 1)
        return self.subs[src][i - self._cum[src]]
