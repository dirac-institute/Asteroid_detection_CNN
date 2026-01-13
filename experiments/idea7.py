#!/usr/bin/env python3
"""
Idea 7 — Ignore pixels marked in `real_labels` when computing loss.

- Same training procedure as baseline (warmup/head/tail/long + phase schedule).
- ONLY change: loss is computed on VALID pixels where real_labels == 0.
- If `real_labels` dataset is missing in train.h5, falls back to normal training.

Assumptions:
- train_h5 contains datasets: images, masks, and optionally real_labels
- real_labels is same shape as masks [N,H,W], with 1 meaning "ignore" (masked out).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    from ADCNN.config import Config
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
        Config,
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
# Dataset helpers
# -------------------------
def filter_tiles_by_panels(base_ds, panel_id_set: set[int]) -> np.ndarray:
    kept = [k for k, (pid, _r, _c) in enumerate(base_ds.indices) if int(pid) in panel_id_set]
    return np.asarray(kept, dtype=np.int64)


class TileSubsetWithReal(Dataset):
    """
    Subset of base tiled dataset by BASE tile indices.
    Returns (x, y, real) where:
      - y is truth mask tile (same as base)
      - real is tile from train_h5[real_labels_key] if exists, else zeros
    """
    def __init__(self, base, base_tile_indices: np.ndarray, train_h5: str, real_labels_key: str):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
        self.train_h5 = str(train_h5)
        self.real_labels_key = str(real_labels_key)

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
        x, y = self.base[base_idx]  # returns tile tensors/arrays already

        # figure out tile bbox in the ORIGINAL panel image to slice real_labels
        # base_ds.indices maps BASE tile index -> (panel_id, r, c)
        self._ensure_h5()
        if not self._has_real:
            # return zeros mask (no ignoring)
            if torch.is_tensor(y):
                real = torch.zeros_like(y)
            else:
                real = np.zeros_like(y)
            return x, y, real

        panel_i, r, c = self.base.indices[base_idx]
        panel_i = int(panel_i); r = int(r); c = int(c)
        t = int(self.base.tile)

        # Need full H/W to slice
        # real_labels stored as [N,H,W]
        rl = self._h5[self.real_labels_key]
        H, W = int(rl.shape[1]), int(rl.shape[2])

        r0, c0 = r * t, c * t
        r1, c1 = min(r0 + t, H), min(c0 + t, W)
        real_np = rl[panel_i, r0:r1, c0:c1]

        # If edge tiles are smaller than tile size, pad to match y shape if needed
        # (H5TiledDataset typically pads/crops; we match y's shape)
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

        return x, y, real


class TileSubset(Dataset):
    def __init__(self, base, base_tile_indices: np.ndarray):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)

    def __len__(self):
        return int(self.base_tile_indices.size)

    def __getitem__(self, i: int):
        return self.base[int(self.base_tile_indices[int(i)])]


# -------------------------
# Masked loss (replicates losses.py behavior, but ignores real_labels==1)
# -------------------------
def _valid_mask_from_real(real: torch.Tensor) -> torch.Tensor:
    # real==1 => ignore; valid = ~real
    if real.dtype != torch.bool:
        real = real > 0.5
    return (~real).to(dtype=torch.float32)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, pos_weight: float) -> torch.Tensor:
    posw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_map = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=posw, reduction="none")
    # valid is float32 [B,1,H,W] or broadcastable
    num = (loss_map * valid).sum()
    den = valid.sum().clamp_min(1.0)
    return num / den


def masked_soft_iou(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = targets.clamp(0, 1)

    # apply valid mask
    p = p * valid
    t = t * valid

    inter = (p * t).sum(dim=(1, 2, 3))
    union = (p + t - p * t).sum(dim=(1, 2, 3)) + eps
    iou = inter / union
    return (1.0 - iou).mean()


def sobel_edge(x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def masked_edge_l1(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = targets.clamp(0, 1)

    ep = sobel_edge(p, kx, ky)
    et = sobel_edge(t, kx, ky)

    loss_map = (ep - et).abs()
    num = (loss_map * valid).sum()
    den = valid.sum().clamp_min(1.0)
    return num / den


def masked_aftl(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
                alpha: float = 0.45, beta: float = 0.55, gamma: float = 1.3, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    t = targets.clamp(0, 1)

    # valid mask in sums
    p = p * valid
    t = t * valid

    p = p.view(p.size(0), -1)
    t = t.view(t.size(0), -1)

    TP = (p * t).sum(1)
    FP = ((1.0 - t) * p).sum(1)
    FN = (t * (1.0 - p)).sum(1)

    ti = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    loss = torch.pow(1.0 - ti, gamma)
    return loss.mean()


def make_masked_loss_for_epoch(ep: int, device: torch.device):
    """
    Mirror losses.make_loss_for_epoch() schedule, but masked.
    """
    # kx/ky like losses.py
    kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).to(device)
    ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).to(device)

    if ep <= 10:
        params = {"lambda_bce": 0.6, "pos_weight": 8.0, "lambda_edge": 0.00, "aftl": False}
        w = {"w_core": 1.0, "w_aftl": 0.0}
    elif ep <= 25:
        params = {"lambda_bce": 0.6, "pos_weight": 8.0, "lambda_edge": 0.00, "aftl": True}
        w = {"w_core": 0.85, "w_aftl": 0.15}
    else:
        params = {"lambda_bce": 0.8, "pos_weight": 8.0, "lambda_edge": 0.03, "aftl": True}
        w = {"w_core": 0.85, "w_aftl": 0.15}

    return params, w, kx, ky


def masked_blended_loss(params, w, logits, targets, valid, kx, ky):
    # core = lambda_bce*BCE + (1-lambda_bce)*IoU + lambda_edge*Edge
    lam_bce = float(params["lambda_bce"])
    lam_edge = float(params["lambda_edge"])
    posw = float(params["pos_weight"])

    loss_bce = masked_bce_with_logits(logits, targets, valid, pos_weight=posw)
    loss_iou = masked_soft_iou(logits, targets, valid)

    core = lam_bce * loss_bce + (1.0 - lam_bce) * loss_iou
    if lam_edge > 0:
        core = core + lam_edge * masked_edge_l1(logits, targets, valid, kx=kx, ky=ky)

    loss = w["w_core"] * core

    if params.get("aftl", False) and w.get("w_aftl", 0) > 0:
        loss = loss + w["w_aftl"] * masked_aftl(logits, targets, valid)

    return loss


# -------------------------
# Eval helper (unchanged; uses y only)
# -------------------------
@torch.no_grad()
def pix_eval(model, resize_masks_to, loader, thr: float, max_batches: int):
    model.eval()
    dev = next(model.parameters()).device

    tp = 0.0
    fp = 0.0
    fn = 0.0
    posm = []
    negm = []

    for bi, batch in enumerate(loader, 1):
        xb, yb = batch[0], batch[1]
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)

        logits = model(xb)
        yb_r = resize_masks_to(logits, yb)
        p = torch.sigmoid(logits)

        if (yb_r > 0.5).any():
            posm.append(float(p[yb_r > 0.5].mean()))
        negm.append(float(p[yb_r <= 0.5].mean()))

        pv = p.reshape(-1)
        tv = yb_r.reshape(-1)
        pred = (pv >= thr).float()

        tp += float((pred * tv).sum())
        fp += float((pred * (1.0 - tv)).sum())
        fn += float(((1.0 - pred) * tv).sum())

        if bi >= max_batches:
            break

    if dist.is_available() and dist.is_initialized():
        vec = torch.tensor([tp, fp, fn], device=dev, dtype=torch.float32)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        tp, fp, fn = map(float, vec.tolist())

    P = tp / max(tp + fp, 1.0)
    R = tp / max(tp + fn, 1.0)
    F1 = 2 * P * R / max(P + R, 1e-8)

    return {
        "P": P,
        "R": R,
        "F": F1,
        "pos_mean": float(sum(posm) / max(len(posm), 1)),
        "neg_mean": float(sum(negm) / max(len(negm), 1)),
    }


# -------------------------
# Trainer: baseline procedure, masked loss everywhere
# -------------------------
class TrainerIdea7:
    def __init__(self, init_distributed, is_main_process, *, device=None, use_amp=True):
        self.init_distributed = init_distributed
        self.is_main_process = is_main_process
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(use_amp)

    def _set_loader_epoch(self, loader, epoch: int):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

    def train_full_probe(
        self,
        model,
        train_loader,
        val_loader,
        *,
        # baseline knobs (same signature as your train_full_probe calls)
        seed=1337,
        init_head_prior=0.70,
        warmup_epochs=1,
        warmup_batches=800,
        warmup_lr=2e-4,
        warmup_pos_weight=40.0,
        head_epochs=2,
        head_batches=2000,
        head_lr=3e-5,
        head_pos_weight=5.0,
        tail_epochs=2,
        tail_batches=2500,
        tail_lr=1.5e-4,
        tail_pos_weight=2.0,
        max_epochs=60,
        long_batches=0,
        val_every=3,
        base_lrs=(3e-4, 2e-4, 1e-4),
        weight_decay=1e-4,
        thr_beta=1.0,
        thr_pos_rate_early=(0.03, 0.10),
        thr_pos_rate_late=(0.08, 0.12),
        save_best_to=None,
        save_last_to=None,
        verbose: int = 2,
        quick_eval_train_batches: int = 6,
        quick_eval_val_batches: int = 12,
        thr0: float = 0.5,
        # injected funcs
        resize_masks_to=None,
        pick_thr_with_floor=None,
        roc_auc_ddp=None,
        maybe_init_head_bias_to_prior=None,
        apply_phase=None,
        freeze_all=None,
        _unfreeze_if_exists=None,
        make_opt_sched=None,
    ):
        assert resize_masks_to is not None
        assert pick_thr_with_floor is not None
        assert roc_auc_ddp is not None

        is_dist, rank, local_rank, world_size = self.init_distributed()
        scaler = amp.GradScaler("cuda", enabled=self.amp)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = model.to(self.device)
        if is_dist:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        raw_model = model.module if isinstance(model, DDP) else model

        maybe_init_head_bias_to_prior(raw_model, init_head_prior)

        # -------- Warmup (masked BCE) --------
        freeze_all(raw_model)
        for p in raw_model.parameters():
            p.requires_grad = True

        opt = torch.optim.Adam(raw_model.parameters(), lr=warmup_lr, weight_decay=0.0)

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = _valid_mask_from_real(rb_r)
                    loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight=float(warmup_pos_weight))

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                if b >= warmup_batches:
                    break

            if self.is_main_process() and verbose >= 2:
                stats = pix_eval(model, resize_masks_to, train_loader, thr=float(thr0), max_batches=quick_eval_train_batches)
                print(f"[WARMUP] ep{ep} loss {loss_sum / max(seen, 1):.4f} | F1 {stats['F']:.3f}")

        metric_thr = float(thr0)

        # -------- Head-only (masked BCE) --------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=0.0)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, warmup_epochs + 100 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = _valid_mask_from_real(rb_r)
                    loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight=float(head_pos_weight))

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                if b >= head_batches:
                    break

            if self.is_main_process() and verbose >= 2:
                stats = pix_eval(model, resize_masks_to, train_loader, thr=metric_thr, max_batches=quick_eval_train_batches)
                print(f"[HEAD] ep{ep} loss {loss_sum / max(seen, 1):.4f} | F1 {stats['F']:.3f}")

        # -------- Tail probe (masked schedule loss) --------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, warmup_epochs + 200 + head_epochs + ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            params, w, kx, ky = make_masked_loss_for_epoch(ep=1, device=self.device)  # tail: stable regime

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = _valid_mask_from_real(rb_r)

                    # tail uses tail_pos_weight, keep same spirit as baseline
                    params_local = dict(params)
                    params_local["pos_weight"] = float(tail_pos_weight)
                    loss = masked_blended_loss(params_local, w, logits, yb_r, valid, kx, ky)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                if b >= tail_batches:
                    break

            # quick threshold pick
            pr_min, pr_max = thr_pos_rate_early
            metric_thr, _, aux = pick_thr_with_floor(
                model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                min_pos_rate=pr_min, max_pos_rate=pr_max
            )
            metric_thr = float(metric_thr)

            if self.is_main_process() and verbose >= 2:
                val_stats = pix_eval(model, resize_masks_to, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                print(f"[TAIL] ep{ep} loss {loss_sum / max(seen,1):.4f} | val F1 {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux.get('pos_rate', float('nan')):.3f}")

        # -------- Long training (masked schedule loss) --------
        best = {"auc": -1.0, "state": None, "thr": float(metric_thr), "ep": 0, "P": 0.0, "R": 0.0, "F": 0.0}

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, warmup_epochs + 300 + head_epochs + tail_epochs + ep)

            _ = apply_phase(raw_model, ep)
            opt, sched = make_opt_sched(raw_model, ep, base_lrs, weight_decay)

            params, w, kx, ky = make_masked_loss_for_epoch(ep=ep, device=self.device)

            model.train()
            seen = 0
            loss_sum = 0.0
            t0 = time.time()

            for i, batch in enumerate(train_loader, 1):
                xb, yb, rb = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = _valid_mask_from_real(rb_r)
                    loss = masked_blended_loss(params, w, logits, yb_r, valid, kx, ky)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step(i)

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)

                if (long_batches > 0) and (i >= long_batches):
                    break

            train_loss = loss_sum / max(seen, 1)

            if self.is_main_process() and verbose >= 2:
                tr_stats = pix_eval(model, resize_masks_to, train_loader, thr=float(metric_thr), max_batches=quick_eval_train_batches)
                print(
                    f"[EP{ep:02d}] loss {train_loss:.4f} | "
                    f"train P {tr_stats['P']:.3f} R {tr_stats['R']:.3f} F {tr_stats['F']:.3f} "
                    f"| {time.time() - t0:.1f}s"
                )

            do_val = (ep % int(val_every) == 0) or (ep <= 1) or (ep == max_epochs)
            if do_val:
                pr_min, pr_max = (thr_pos_rate_early if ep < 26 else thr_pos_rate_late)
                thr, _, aux = pick_thr_with_floor(
                    model,
                    val_loader,
                    max_batches=120,
                    n_bins=256,
                    beta=thr_beta,
                    min_pos_rate=pr_min,
                    max_pos_rate=pr_max,
                )
                metric_thr = float(thr)

                val_stats = pix_eval(model, resize_masks_to, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)

                if self.is_main_process() and verbose >= 1:
                    print(
                        f"[VAL ep{ep}] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} "
                        f"F {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux.get('pos_rate', float('nan')):.3f}"
                    )

                if float(auc) > best["auc"]:
                    best = {
                        "auc": float(auc),
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": float(metric_thr),
                        "ep": int(ep),
                        "P": float(val_stats["P"]),
                        "R": float(val_stats["R"]),
                        "F": float(val_stats["F"]),
                    }
                    if self.is_main_process() and save_best_to:
                        torch.save(best, save_best_to)

            if self.is_main_process() and save_last_to:
                torch.save(best, save_last_to)

        if best["state"] is not None:
            raw_model.load_state_dict(best["state"], strict=True)

        summary = {
            "best_auc": float(best["auc"]),
            "best_ep": int(best["ep"]),
            "best_P": float(best["P"]),
            "best_R": float(best["R"]),
            "best_F": float(best["F"]),
            "final_thr": float(best["thr"]),
        }
        if self.is_main_process():
            print("=== DONE ===")
            print("Best summary:", summary)

        return raw_model, best["thr"], summary


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Idea7: ignore real_labels in loss.")
    ap.add_argument("--repo-root", type=str, default="../")
    ap.add_argument("--train-h5", type=str, default="../DATA/test.h5")
    ap.add_argument("--test-h5", type=str, default="../DATA/test.h5")  # kept for consistency (unused here)
    ap.add_argument("--train-csv", type=str, default="../DATA/train.csv")  # kept for consistency (unused here)

    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.1)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true", default=False)

    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--long-batches", type=int, default=0)

    ap.add_argument("--real-labels-key", type=str, default="real_labels",
                    help="HDF5 dataset name for ignore mask (1=ignore).")

    ap.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    ap.add_argument("--tag", type=str, default="idea7")
    ap.add_argument("--verbose", type=int, default=3)
    return ap.parse_args()


def main():
    args = parse_args()
    add_repo_to_syspath(args.repo_root)

    (
        Config,
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

    # initialize DDP
    is_dist, rank, local_rank, world_size = init_distributed()

    cfg = Config()
    cfg.train.max_epochs = int(args.max_epochs)
    cfg.train.val_every = int(args.val_every) if args.val_every is not None else int(args.max_epochs)

    set_seed(int(args.seed))

    base_ds = H5TiledDataset(args.train_h5, tile=int(args.tile), k_sigma=5.0)

    idx_tr, idx_va = split_indices(args.train_h5, val_frac=float(args.val_frac), seed=int(args.seed))
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))

    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    train_ds = TileSubsetWithReal(base_ds, tiles_tr, train_h5=args.train_h5, real_labels_key=args.real_labels_key)
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
        # quick check if real_labels exists
        try:
            with h5py.File(args.train_h5, "r") as f:
                print(f"real_labels present: {args.real_labels_key in f}")
        except Exception as e:
            print(f"WARNING: could not inspect H5 for real_labels: {e}")

    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    trainer = TrainerIdea7(init_distributed=init_distributed, is_main_process=is_main_process, device=device, use_amp=True)

    model, thr, summary = trainer.train_full_probe(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed=int(args.seed),
        init_head_prior=cfg.train.init_head_prior,

        warmup_epochs=cfg.train.warmup_epochs,
        warmup_batches=cfg.train.warmup_batches,
        warmup_lr=cfg.train.warmup_lr,
        warmup_pos_weight=cfg.train.warmup_pos_weight,

        head_epochs=cfg.train.head_epochs,
        head_batches=cfg.train.head_batches,
        head_lr=cfg.train.head_lr,
        head_pos_weight=cfg.train.head_pos_weight,

        tail_epochs=cfg.train.tail_epochs,
        tail_batches=cfg.train.tail_batches,
        tail_lr=cfg.train.tail_lr,
        tail_pos_weight=cfg.train.tail_pos_weight,

        max_epochs=int(args.max_epochs),
        long_batches=int(args.long_batches),
        val_every=int(args.val_every),

        base_lrs=cfg.train.base_lrs,
        weight_decay=cfg.train.weight_decay,

        thr_beta=cfg.train.thr_beta,
        thr_pos_rate_early=cfg.train.thr_pos_rate_early,
        thr_pos_rate_late=cfg.train.thr_pos_rate_late,

        save_best_to=best_path,
        save_last_to=last_path,
        verbose=int(args.verbose),

        # inject project functions
        resize_masks_to=resize_masks_to,
        pick_thr_with_floor=pick_thr_with_floor,
        roc_auc_ddp=roc_auc_ddp,
        maybe_init_head_bias_to_prior=maybe_init_head_bias_to_prior,
        apply_phase=apply_phase,
        freeze_all=freeze_all,
        _unfreeze_if_exists=_unfreeze_if_exists,
        make_opt_sched=make_opt_sched,
    )

    if is_main_process():
        print("Final thr:", float(thr))
        print("Summary:", summary)
        print("Best ckpt:", best_path)
        print("Last ckpt:", last_path)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
