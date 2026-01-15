#!/usr/bin/env python3
"""
Idea 8: Combine FP-masking + LSST-missed loss + Focal Tversky

Implements:
1) Loss masking (Idea 7):
   - Ignore pixels where real_labels > 0 (exclude from ALL loss terms).
2) LSST-missed weighting (Idea 2 style, but foreground-only):
   - Build a per-BASE-tile flag from train.csv rows with stack_detection==0.
   - If a tile is flagged "missed", multiply weights on POSITIVE (foreground) pixels
     by `--missed-weight` (default 2.0). Background stays weight 1.0.
3) Loss:
   - Focal Tversky only (alpha, gamma configurable), supports:
       * per-pixel valid mask (from real_labels)
       * per-pixel weights (foreground-only missed weighting)
4) Logging:
   - Each epoch prints:
       * active loss pixels (sum of valid pixels used by loss)
       * masked-out percentage (fraction of pixels excluded by real_labels)
5) Output:
   - Saves best model to checkpoints/Experiments/Last/idea8.pt (by VAL AUC, matching baseline logic)

Assumptions:
- train.h5 contains: images, masks, and optionally real_labels (shape [N,H,W])
- train.csv contains at least: image_id, x, y, trail_length, stack_detection
  (This matches your injection CSV style used in idea2; we mark tiles touched by the object bbox.)
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# -------------------------
# Path setup (repo imports)
# -------------------------
def add_repo_to_syspath(repo_root: str):
    import sys
    repo = Path(repo_root).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(repo / "ADCNN") not in sys.path:
        sys.path.insert(0, str(repo / "ADCNN"))


def import_project():
    from ADCNN.data.h5tiles import H5TiledDataset
    from ADCNN.models.unet_res_se import UNetResSEASPP
    from ADCNN.utils.utils import set_seed, split_indices
    from metrics import roc_auc_ddp
    from phases import (
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
    )
    from thresholds import pick_thr_with_floor, resize_masks_to
    from utils.dist_utils import init_distributed, is_main_process

    return (
        H5TiledDataset,
        UNetResSEASPP,
        set_seed,
        split_indices,
        roc_auc_ddp,
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
        pick_thr_with_floor,
        resize_masks_to,
        init_distributed,
        is_main_process,
    )


# -------------------------
# Helpers: DDP reduce for scalars
# -------------------------
def ddp_sum_float(x: float, device: torch.device) -> float:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([x], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    return float(x)


# -------------------------
# Dataset helpers
# -------------------------
def filter_tiles_by_panels(base_ds, panel_id_set: set[int]) -> np.ndarray:
    kept = [k for k, (pid, _r, _c) in enumerate(base_ds.indices) if int(pid) in panel_id_set]
    return np.asarray(kept, dtype=np.int64)


def tiles_touched_by_bbox(xc: float, yc: float, R: float, H: int, W: int, tile: int) -> Tuple[int, int, int, int]:
    """
    Returns inclusive tile-index bbox (r0..r1, c0..c1) for a circle bbox around (x,y) with radius R.
    Coordinates:
      - CSV uses x (column), y (row) in pixel coords.
      - base_ds.indices uses (panel_id, r, c) for tiles.
    """
    Hb = int(np.ceil(H / tile))
    Wb = int(np.ceil(W / tile))

    x0 = max(0.0, xc - R)
    x1 = min(float(W - 1), xc + R)
    y0 = max(0.0, yc - R)
    y1 = min(float(H - 1), yc + R)

    c0 = int(np.floor(x0 / tile))
    c1 = int(np.floor(x1 / tile))
    r0 = int(np.floor(y0 / tile))
    r1 = int(np.floor(y1 / tile))

    c0 = max(0, min(Wb - 1, c0))
    c1 = max(0, min(Wb - 1, c1))
    r0 = max(0, min(Hb - 1, r0))
    r1 = max(0, min(Hb - 1, r1))
    return r0, r1, c0, c1


def build_missed_tile_mask_from_csv(
    train_csv: str,
    base_ds,
    *,
    tile: int,
    margin_pix: float = 0.0,
    stack_col: str = "stack_detection",
) -> np.ndarray:
    """
    Returns hard_mask_base: bool array of length len(base_ds), True for tiles touched by
    an injection with stack_detection == 0.

    Expected CSV columns (idea2-style):
      image_id, x, y, trail_length, stack_detection
    """
    cat = pd.read_csv(train_csv)
    need = {"image_id", "x", "y", "trail_length", stack_col}
    miss = need - set(cat.columns)
    if miss:
        raise ValueError(f"train.csv missing required columns: {sorted(miss)}")

    # Only LSST-missed rows
    miss_df = cat[cat[stack_col].astype(int) == 0].copy()
    if len(miss_df) == 0:
        return np.zeros(len(base_ds), dtype=bool)

    # Need panel H/W
    with h5py.File(base_ds.h5_path, "r") as f:
        H = int(f["images"].shape[1])
        W = int(f["images"].shape[2])

    Hb = int(np.ceil(H / tile))
    Wb = int(np.ceil(W / tile))

    # Build (panel_id -> set((r,c))) touched by missed injections
    hard_rc_by_panel: Dict[int, set[Tuple[int, int]]] = {}

    for _, row in miss_df.iterrows():
        pid = int(row["image_id"])
        x = float(row["x"])
        y = float(row["y"])
        L = float(row["trail_length"])
        R = (L / 2.0) + float(margin_pix)

        r0, r1, c0, c1 = tiles_touched_by_bbox(x, y, R, H, W, tile)
        r0 = max(0, min(Hb - 1, r0))
        r1 = max(0, min(Hb - 1, r1))
        c0 = max(0, min(Wb - 1, c0))
        c1 = max(0, min(Wb - 1, c1))

        s = hard_rc_by_panel.setdefault(pid, set())
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                s.add((r, c))

    base_map = {(i, r, c): k for k, (i, r, c) in enumerate(base_ds.indices)}
    hard_mask = np.zeros(len(base_ds), dtype=bool)

    for pid, rcset in hard_rc_by_panel.items():
        for (r, c) in rcset:
            k = base_map.get((pid, r, c))
            if k is not None:
                hard_mask[k] = True

    return hard_mask


class TileSubsetWithRealAndMissed(Dataset):
    """
    Subset of base tiled dataset by BASE tile indices.
    Returns (x, y, real, missed_flag) where:
      - real is tile from train_h5[real_labels_key] if exists, else zeros
      - missed_flag indicates whether this BASE tile is flagged missed by LSST (from CSV-derived mask)
    """
    def __init__(
        self,
        base,
        base_tile_indices: np.ndarray,
        *,
        train_h5: str,
        real_labels_key: str,
        missed_mask_base: np.ndarray,  # len(base_ds) bool
    ):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
        self.train_h5 = str(train_h5)
        self.real_labels_key = str(real_labels_key)

        missed_mask_base = np.asarray(missed_mask_base, dtype=bool)
        if missed_mask_base.shape[0] != len(base):
            raise ValueError("missed_mask_base must have length len(base_ds)")
        self.missed_mask_base = missed_mask_base

        # Lazily opened per worker/process
        self._h5 = None
        self._has_real = None

    def __len__(self):
        return int(self.base_tile_indices.size)

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.train_h5, "r")
            self._has_real = (self.real_labels_key in self._h5)

    def __getitem__(self, i: int):
        base_idx = int(self.base_tile_indices[int(i)])
        x, y = self.base[base_idx]

        missed_flag = bool(self.missed_mask_base[base_idx])

        self._ensure_h5()
        if not self._has_real:
            real = torch.zeros_like(y) if torch.is_tensor(y) else np.zeros_like(y)
            return x, y, real, missed_flag

        panel_i, r, c = self.base.indices[base_idx]
        panel_i = int(panel_i)
        r = int(r)
        c = int(c)
        t = int(self.base.tile)

        rl = self._h5[self.real_labels_key]  # [N,H,W]
        H, W = int(rl.shape[1]), int(rl.shape[2])

        r0, c0 = r * t, c * t
        r1, c1 = min(r0 + t, H), min(c0 + t, W)
        real_np = rl[panel_i, r0:r1, c0:c1]

        # pad to match y shape (edge tiles)
        if torch.is_tensor(y):
            real = torch.as_tensor(real_np, dtype=y.dtype)
            if real.shape != y.shape:
                out = torch.zeros_like(y)
                hh = min(out.shape[-2], real.shape[-2])
                ww = min(out.shape[-1], real.shape[-1])
                out[..., :hh, :ww] = real[..., :hh, :ww]
                real = out
        else:
            real = real_np.astype(y.dtype, copy=False)
            if real.shape != y.shape:
                out = np.zeros_like(y)
                hh = min(out.shape[-2], real.shape[-2])
                ww = min(out.shape[-1], real.shape[-1])
                out[..., :hh, :ww] = real[..., :hh, :ww]
                real = out

        return x, y, real, missed_flag


class TileSubset(Dataset):
    def __init__(self, base, base_tile_indices: np.ndarray):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)

    def __len__(self):
        return int(self.base_tile_indices.size)

    def __getitem__(self, i: int):
        return self.base[int(self.base_tile_indices[int(i)])]


# -------------------------
# Loss: masked + weighted focal tversky
# -------------------------
def valid_mask_from_real(real: torch.Tensor) -> torch.Tensor:
    # real > 0 => ignore ; valid = 1 for pixels we keep
    if real.dtype != torch.bool:
        real = real > 0.5
    return (~real).to(dtype=torch.float32)


def focal_tversky_masked_weighted(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    logits, targets: [B,1,H,W]
    valid, weights:  [B,1,H,W] float32, where valid is 0/1
    Computes TP/FP/FN only over valid pixels, scaled by weights.
    """
    p = torch.sigmoid(logits)
    t = targets.clamp(0, 1)

    w = (valid * weights).to(dtype=p.dtype)

    TP = (w * p * t).sum(dim=(1, 2, 3))
    FP = (w * p * (1.0 - t)).sum(dim=(1, 2, 3))
    FN = (w * (1.0 - p) * t).sum(dim=(1, 2, 3))

    tv = (TP + eps) / (TP + alpha * FP + (1.0 - alpha) * FN + eps)
    loss = torch.pow(1.0 - tv, gamma)
    return loss.mean()


# -------------------------
# Trainer (Idea7-like schedule, but FT loss everywhere)
# -------------------------
class TrainerIdea8:
    def __init__(self, *, init_distributed, is_main_process, device: torch.device, use_amp: bool = True):
        self.init_distributed = init_distributed
        self.is_main_process = is_main_process
        self.device = device
        self.amp = bool(use_amp)

    def _set_loader_epoch(self, loader, seed: int):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(int(seed))

    def train_full_probe(
        self,
        model,
        *,
        train_loader,
        val_loader,
        seed: int,
        init_head_prior: float,
        # warmup/head/tail/long (keep baseline defaults unless overridden)
        warmup_epochs: int,
        warmup_batches: int,
        warmup_lr: float,
        head_epochs: int,
        head_batches: int,
        head_lr: float,
        tail_epochs: int,
        tail_batches: int,
        tail_lr: float,
        max_epochs: int,
        val_every: int,
        base_lrs: Tuple[float, float, float],
        weight_decay: float,
        # thr picking
        thr_beta: float,
        thr_pos_rate_early: Tuple[float, float],
        thr_pos_rate_late: Tuple[float, float],
        # focal tversky params
        alpha: float,
        gamma: float,
        missed_weight: float,
        # io
        save_best_to: str,
        quick_eval_train_batches: int = 6,
        quick_eval_val_batches: int = 12,
        long_batches: int = 0,
        verbose: int = 2,
    ):
        (
            _H5TiledDataset,
            _UNetResSEASPP,
            _set_seed,
            _split_indices,
            roc_auc_ddp,
            _unfreeze_if_exists,
            apply_phase,
            freeze_all,
            make_opt_sched,
            maybe_init_head_bias_to_prior,
            pick_thr_with_floor,
            resize_masks_to,
            _init_distributed,
            _is_main_process,
        ) = import_project()

        is_dist, rank, local_rank, world_size = self.init_distributed()
        raw_model = model.module if isinstance(model, DDP) else model

        # AMP scaler
        scaler = amp.GradScaler("cuda", enabled=self.amp)

        # ---- Warmup ----
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=warmup_lr, weight_decay=0.0)

        if hasattr(raw_model, "head"):
            maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 100 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            # logging accumulators
            valid_sum = 0.0
            pix_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb, missed_flag = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                missed_flag_t = torch.as_tensor(missed_flag, device=self.device)
                missed_flag_t = missed_flag_t.to(dtype=torch.float32).view(-1, 1, 1, 1)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)

                    valid = valid_mask_from_real(rb_r)  # [B,1,H,W] float
                    pos = (yb_r > 0.5).to(dtype=valid.dtype)
                    weights = torch.ones_like(valid)
                    if missed_weight != 1.0:
                        weights = weights + (float(missed_weight) - 1.0) * pos * missed_flag_t

                    loss = focal_tversky_masked_weighted(
                        logits, yb_r, valid, weights, alpha=float(alpha), gamma=float(gamma)
                    )

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)

                valid_sum += float(valid.sum().item())
                pix_sum += float(valid.numel())

                if b >= warmup_batches:
                    break

            if verbose >= 2 and self.is_main_process():
                active = ddp_sum_float(valid_sum, self.device)
                total = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))
                print(
                    f"[WARMUP] ep{ep} loss {loss_sum / max(seen, 1):.4f} | "
                    f"active_pix={active:.0f} masked_out={masked_pct:.2f}%"
                )

        # ---- Head ----
        freeze_all(raw_model)
        for g in ["head", "u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=1e-4)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 200 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            valid_sum = 0.0
            pix_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb, missed_flag = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                missed_flag_t = torch.as_tensor(missed_flag, device=self.device).to(dtype=torch.float32).view(-1, 1, 1, 1)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)

                    valid = valid_mask_from_real(rb_r)
                    pos = (yb_r > 0.5).to(dtype=valid.dtype)
                    weights = torch.ones_like(valid)
                    if missed_weight != 1.0:
                        weights = weights + (float(missed_weight) - 1.0) * pos * missed_flag_t

                    loss = focal_tversky_masked_weighted(
                        logits, yb_r, valid, weights, alpha=float(alpha), gamma=float(gamma)
                    )

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                valid_sum += float(valid.sum().item())
                pix_sum += float(valid.numel())

                if b >= head_batches:
                    break

            if verbose >= 2 and self.is_main_process():
                active = ddp_sum_float(valid_sum, self.device)
                total = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))
                print(
                    f"[HEAD] ep{ep} loss {loss_sum / max(seen, 1):.4f} | "
                    f"active_pix={active:.0f} masked_out={masked_pct:.2f}%"
                )

        # ---- Tail probe ----
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)

        # quick threshold initialization uses val loader (unchanged baseline strategy)
        metric_thr = 0.5

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 300 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            valid_sum = 0.0
            pix_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb, missed_flag = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                missed_flag_t = torch.as_tensor(missed_flag, device=self.device).to(dtype=torch.float32).view(-1, 1, 1, 1)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)

                    valid = valid_mask_from_real(rb_r)
                    pos = (yb_r > 0.5).to(dtype=valid.dtype)
                    weights = torch.ones_like(valid)
                    if missed_weight != 1.0:
                        weights = weights + (float(missed_weight) - 1.0) * pos * missed_flag_t

                    loss = focal_tversky_masked_weighted(
                        logits, yb_r, valid, weights, alpha=float(alpha), gamma=float(gamma)
                    )

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                valid_sum += float(valid.sum().item())
                pix_sum += float(valid.numel())

                if b >= tail_batches:
                    break

            # quick threshold pick
            pr_min, pr_max = thr_pos_rate_early
            metric_thr, _, aux = pick_thr_with_floor(
                model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                min_pos_rate=pr_min, max_pos_rate=pr_max
            )
            metric_thr = float(metric_thr)

            if verbose >= 2 and self.is_main_process():
                active = ddp_sum_float(valid_sum, self.device)
                total = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))
                print(
                    f"[TAIL] ep{ep} loss {loss_sum / max(seen, 1):.4f} | thr={metric_thr:.3f} | "
                    f"active_pix={active:.0f} masked_out={masked_pct:.2f}% | pos_rate≈{aux.get('pos_rate', float('nan')):.3f}"
                )

        # ---- Long training (baseline phase schedule) ----
        best = {"auc": -1.0, "state": None, "thr": float(metric_thr), "ep": 0, "P": 0.0, "R": 0.0, "F": 0.0}

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 1000 + ep)

            _ = apply_phase(raw_model, ep)
            opt, sched = make_opt_sched(raw_model, ep, base_lrs, weight_decay)

            model.train()
            seen = 0
            loss_sum = 0.0
            t0 = time.time()

            valid_sum = 0.0
            pix_sum = 0.0

            for i, batch in enumerate(train_loader, 1):
                xb, yb, rb, missed_flag = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                missed_flag_t = torch.as_tensor(missed_flag, device=self.device).to(dtype=torch.float32).view(-1, 1, 1, 1)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)

                    valid = valid_mask_from_real(rb_r)
                    pos = (yb_r > 0.5).to(dtype=valid.dtype)
                    weights = torch.ones_like(valid)
                    if missed_weight != 1.0:
                        weights = weights + (float(missed_weight) - 1.0) * pos * missed_flag_t

                    loss = focal_tversky_masked_weighted(
                        logits, yb_r, valid, weights, alpha=float(alpha), gamma=float(gamma)
                    )

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step(i)

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)

                valid_sum += float(valid.sum().item())
                pix_sum += float(valid.numel())

                if (long_batches > 0) and (i >= long_batches):
                    break

            train_loss = loss_sum / max(seen, 1)

            if self.is_main_process() and verbose >= 1:
                active = ddp_sum_float(valid_sum, self.device)
                total = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))
                print(
                    f"[EP{ep:02d}] loss {train_loss:.4f} | thr={best['thr']:.3f} | "
                    f"active_pix={active:.0f} masked_out={masked_pct:.2f}% | {time.time() - t0:.1f}s"
                )

            if (ep % val_every == 0) or (ep <= 3):
                # threshold pick (baseline)
                pr_min, pr_max = (thr_pos_rate_early if ep < 26 else thr_pos_rate_late)
                thr, (_VP, _VR, _VF), aux = pick_thr_with_floor(
                    model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                thr = float(thr)
                auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=12)

                if self.is_main_process():
                    print(
                        f"[VAL ep{ep}] AUC {auc:.3f} | thr={thr:.3f} | pos_rate≈{aux.get('pos_rate', float('nan')):.3f}"
                    )

                # keep baseline criterion: best by AUC
                if auc > best["auc"]:
                    best = {
                        "auc": float(auc),
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": thr,
                        "ep": int(ep),
                        "P": 0.0,
                        "R": 0.0,
                        "F": 0.0,
                    }
                    if self.is_main_process():
                        Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"state": best["state"], "thr": best["thr"], "ep": best["ep"], "auc": best["auc"]},
                            save_best_to,
                        )

        # load best back
        if best["state"] is not None:
            raw_model.load_state_dict(best["state"])

        return model, float(best["thr"]), best


# -------------------------
# CLI / main
# -------------------------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".", help="Repo root (contains ADCNN/)")
    ap.add_argument("--train-h5", type=str, default="./DATA/train.h5")
    ap.add_argument("--train-csv", type=str, default="./DATA/train.csv")
    ap.add_argument("--real-labels-key", type=str, default="real_labels")

    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--pin-memory", action="store_true", default=True)

    # focal tversky
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--missed-weight", type=float, default=2.0)

    # missed-tiling parameters
    ap.add_argument("--margin-pix", type=float, default=0.0, help="Extra radius margin (pixels) for touched tiles")

    # training (inline defaults = Config defaults)
    ap.add_argument("--init-head-prior", type=float, default=0.70)

    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--warmup-batches", type=int, default=800)
    ap.add_argument("--warmup-lr", type=float, default=2e-4)

    ap.add_argument("--head-epochs", type=int, default=10)
    ap.add_argument("--head-batches", type=int, default=2000)
    ap.add_argument("--head-lr", type=float, default=3e-5)

    ap.add_argument("--tail-epochs", type=int, default=6)
    ap.add_argument("--tail-batches", type=int, default=2500)
    ap.add_argument("--tail-lr", type=float, default=1.5e-4)

    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--base-lrs", type=float, nargs=3, default=[3e-4, 2e-4, 1e-4])
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--thr-beta", type=float, default=1.0)
    ap.add_argument("--thr-pos-rate-early", type=float, nargs=2, default=[0.03, 0.10])
    ap.add_argument("--thr-pos-rate-late", type=float, nargs=2, default=[0.08, 0.12])

    ap.add_argument("--save-best-to", type=str, default="checkpoints/Experiments/Last/idea8.pt")
    ap.add_argument("--verbose", type=int, default=2)
    return ap.parse_args()


def main():
    args = cli()
    add_repo_to_syspath(args.repo_root)

    (
        H5TiledDataset,
        UNetResSEASPP,
        set_seed,
        split_indices,
        roc_auc_ddp,
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
        pick_thr_with_floor,
        resize_masks_to,
        init_distributed,
        is_main_process,
    ) = import_project()

    is_dist, rank, local_rank, world_size = init_distributed()
    import builtins
    if not is_main_process():
        builtins.print = lambda *a, **k: None

    set_seed(int(args.seed))

    base_ds = H5TiledDataset(args.train_h5, tile=int(args.tile), k_sigma=5.0)

    idx_tr, idx_va = split_indices(args.train_h5, val_frac=float(args.val_frac), seed=int(args.seed))
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))

    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    # Build missed-mask on BASE tiles from CSV
    missed_mask_base = build_missed_tile_mask_from_csv(
        args.train_csv,
        base_ds,
        tile=int(args.tile),
        margin_pix=float(args.margin_pix),
        stack_col="stack_detection",
    )

    train_ds = TileSubsetWithRealAndMissed(
        base_ds,
        tiles_tr,
        train_h5=args.train_h5,
        real_labels_key=args.real_labels_key,
        missed_mask_base=missed_mask_base,
    )
    val_ds = TileSubset(base_ds, tiles_va)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    if is_main_process():
        print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")
        try:
            with h5py.File(args.train_h5, "r") as f:
                print(f"real_labels present: {args.real_labels_key in f}")
        except Exception as e:
            print(f"WARNING: could not inspect H5 for real_labels: {e}")

        print(f"Missed BASE tiles (from CSV): {int(missed_mask_base.sum())} / {len(missed_mask_base)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    trainer = TrainerIdea8(init_distributed=init_distributed, is_main_process=is_main_process, device=device, use_amp=True)

    model, thr, summary = trainer.train_full_probe(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed=int(args.seed),
        init_head_prior=float(args.init_head_prior),

        warmup_epochs=int(args.warmup_epochs),
        warmup_batches=int(args.warmup_batches),
        warmup_lr=float(args.warmup_lr),

        head_epochs=int(args.head_epochs),
        head_batches=int(args.head_batches),
        head_lr=float(args.head_lr),

        tail_epochs=int(args.tail_epochs),
        tail_batches=int(args.tail_batches),
        tail_lr=float(args.tail_lr),

        max_epochs=int(args.max_epochs),
        val_every=int(args.val_every),
        base_lrs=(float(args.base_lrs[0]), float(args.base_lrs[1]), float(args.base_lrs[2])),
        weight_decay=float(args.weight_decay),

        thr_beta=float(args.thr_beta),
        thr_pos_rate_early=(float(args.thr_pos_rate_early[0]), float(args.thr_pos_rate_early[1])),
        thr_pos_rate_late=(float(args.thr_pos_rate_late[0]), float(args.thr_pos_rate_late[1])),

        alpha=float(args.alpha),
        gamma=float(args.gamma),
        missed_weight=float(args.missed_weight),

        save_best_to=str(args.save_best_to),
        verbose=int(args.verbose),
    )

    if is_main_process():
        print("Final threshold:", thr)
        print("Summary:", summary)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
