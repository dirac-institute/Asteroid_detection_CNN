#!/usr/bin/env python3
"""
Idea 6 — Line coherence bias (auxiliary head)

Auxiliary-head option: add a head predicting local trail orientation/direction.

Implementation:
- Model: UNetResSEASPP backbone + two heads:
    * head:     segmentation logits (1 ch)
    * aux_head: orientation field (2 ch) predicting (cos(2θ), sin(2θ)) in [-1,1] via tanh
- Targets: derive θ from GT mask using blurred-mask Sobel gradients:
    θ_tangent = atan2(gy, gx) + pi/2
    target = (cos(2θ), sin(2θ))
- Loss: standard seg loss + ori_weight * MSE over valid pixels (GT positive, optionally dilated),
        with stability gating on gradient magnitude.

Goal:
- Reuse same training schedule as baseline (warmup/head/tail/long)
- Change ONLY the model + add auxiliary loss during all phases (or optionally only LONG)
- Keep budget comparable to other Ideas.

Standalone script (does NOT modify ADCNN/train.py).
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# -------------------------
# DDP init safety wrapper
# -------------------------
def init_distributed_once(init_fn):
    def _wrapped():
        if dist.is_available() and dist.is_initialized():
            import os
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return True, rank, local_rank, world_size
        return init_fn()
    return _wrapped


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
    # keep same imports as idea5-style scripts
    from ADCNN.data.h5tiles import H5TiledDataset
    from ADCNN.config import Config
    from ADCNN.utils.utils import set_seed, split_indices
    from thresholds import pick_thr_with_floor, resize_masks_to
    from metrics import roc_auc_ddp
    from utils.dist_utils import init_distributed, is_main_process
    from phases import (
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
    )
    return (
        Config,
        H5TiledDataset,
        set_seed,
        split_indices,
        pick_thr_with_floor,
        resize_masks_to,
        roc_auc_ddp,
        init_distributed,
        is_main_process,
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
    )


# -------------------------
# Data helpers
# -------------------------
def filter_tiles_by_panels(base_ds, panel_ids_set: set[int]) -> np.ndarray:
    out = []
    for ti, (pid, _r, _c) in enumerate(base_ds.indices):
        if int(pid) in panel_ids_set:
            out.append(ti)
    return np.asarray(out, dtype=np.int64)


class TileSubset(Dataset):
    def __init__(self, base, tile_indices: np.ndarray):
        self.base = base
        self.tile_indices = np.asarray(tile_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.tile_indices.size)

    def __getitem__(self, i: int):
        return self.base[int(self.tile_indices[int(i)])]


# -------------------------
# Orientation target builder (from GT mask)
# -------------------------
@torch.no_grad()
def orientation_target_from_mask(
    y: torch.Tensor,
    *,
    blur_ksize: int = 7,
    blur_sigma: float = 1.5,
    dilate: int = 3,
    grad_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y: float tensor [B,1,H,W] in {0,1} (or [B,1,h,w] after resize)
    Returns:
      ori_tgt:  [B,2,H,W] with (cos2, sin2)
      ori_mask: [B,1,H,W] boolean mask where loss is applied
    """
    assert y.ndim == 4 and y.size(1) == 1
    dev = y.device
    yt = y.float()

    # Optional dilation to include nearby pixels around the trail centerline.
    if dilate and dilate > 1:
        k = int(dilate)
        yt_d = F.max_pool2d(yt, kernel_size=k, stride=1, padding=k // 2)
        pos_mask = (yt_d > 0.5)
    else:
        pos_mask = (yt > 0.5)

    # Gaussian blur via separable 1D kernel (cheap, no torchvision dependency).
    if blur_ksize and blur_ksize >= 3:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        r = k // 2
        xs = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        g = torch.exp(-(xs ** 2) / (2.0 * float(blur_sigma) ** 2))
        g = (g / g.sum()).view(1, 1, 1, k)  # horizontal
        yt_blur = F.conv2d(yt, g, padding=(0, r))
        gT = g.transpose(-1, -2)  # vertical
        yt_blur = F.conv2d(yt_blur, gT, padding=(r, 0))
    else:
        yt_blur = yt

    # Sobel filters
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=dev, dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=dev, dtype=torch.float32).view(1, 1, 3, 3)

    gx = F.conv2d(yt_blur, kx, padding=1)
    gy = F.conv2d(yt_blur, ky, padding=1)

    gmag = torch.sqrt(gx * gx + gy * gy)
    # valid where positive and gradient is meaningful
    ori_mask = pos_mask & (gmag > float(grad_eps))

    # Gradient direction is normal to line; tangent is +90deg
    theta = torch.atan2(gy, gx) + (math.pi / 2.0)

    # Use doubled-angle representation to remove 180° ambiguity:
    cos2 = torch.cos(2.0 * theta)
    sin2 = torch.sin(2.0 * theta)
    ori_tgt = torch.cat([cos2, sin2], dim=1)  # [B,2,H,W]
    return ori_tgt, ori_mask.float()


# -------------------------
# Model: UNetResSEASPP + aux orientation head
# (based on your current UNetResSEASPP code)
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


def _norm(c, groups=8):
    g = min(groups, c) if c % groups == 0 else 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, act=nn.SiLU, se=True):
        super().__init__()
        p = k // 2
        self.proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)
        self.bn1 = _norm(c_in)
        self.c1 = nn.Conv2d(c_in, c_out, k, padding=p, bias=False)
        self.bn2 = _norm(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, k, padding=p, bias=False)
        self.act = act()
        self.se = SEBlock(c_out) if se else nn.Identity()

    def forward(self, x):
        h = self.act(self.bn1(x))
        h = self.c1(h)
        h = self.act(self.bn2(h))
        h = self.c2(h)
        h = self.se(h)
        return h + self.proj(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.rb = ResBlock(c_in, c_out)

    def forward(self, x):
        return self.rb(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in, 2, stride=2)
        self.rb1 = ResBlock(c_in + c_skip, c_out)
        self.rb2 = ResBlock(c_out, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh or dw:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
        x = torch.cat([x, skip], 1)
        x = self.rb1(x)
        x = self.rb2(x)
        return x


class ASPP(nn.Module):
    def __init__(self, c, r=(1, 6, 12, 18)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c // 4, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(c // 4),
                nn.SiLU(True),
            )
            for d in r
        ])
        self.project = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.blocks], 1))


class UNetResSEASPP_AuxOri(nn.Module):
    """
    Returns (seg_logits, ori_pred) where:
      seg_logits: [B,1,H,W]
      ori_pred:   [B,2,H,W] (tanh range)
    """
    def __init__(self, in_ch=1, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(w[0]),
            nn.SiLU(True),
            ResBlock(w[0], w[0]),
        )
        self.d1 = Down(w[0], w[1])
        self.d2 = Down(w[1], w[2])
        self.d3 = Down(w[2], w[3])
        self.d4 = Down(w[3], w[4])

        self.aspp = ASPP(w[4])

        self.u1 = Up(w[4], w[3], w[3])
        self.u2 = Up(w[3], w[2], w[2])
        self.u3 = Up(w[2], w[1], w[1])
        self.u4 = Up(w[1], w[0], w[0])

        self.head = nn.Conv2d(w[0], 1, 1)     # segmentation
        self.aux_head = nn.Conv2d(w[0], 2, 1) # orientation (cos2, sin2)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        b = self.d4(s3)
        b = self.aspp(b)

        x = self.u1(b, s3)
        x = self.u2(x, s2)
        x = self.u3(x, s1)
        x = self.u4(x, s0)

        seg = self.head(x)
        ori = torch.tanh(self.aux_head(x))
        return seg, ori

class SegOnlyWrapper(nn.Module):
    """Expose only segmentation logits for code paths expecting model(x)->logits Tensor."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
# -------------------------
# Small eval for printing (seg only)
# -------------------------
@torch.no_grad()
def pix_eval(model, resize_masks_to, loader, thr=0.2, max_batches=12):
    model.eval()
    dev = next(model.parameters()).device
    tp = fp = fn = 0.0
    for i, batch in enumerate(loader, 1):
        xb, yb = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)

        out = model(xb)
        seg_logits = out[0] if isinstance(out, (tuple, list)) else out

        yb_r = resize_masks_to(seg_logits, yb)
        p = torch.sigmoid(seg_logits)

        pv, tv = p.reshape(-1), yb_r.reshape(-1)
        pred = (pv >= thr).float()
        tp += float((pred * tv).sum())
        fp += float((pred * (1 - tv)).sum())
        fn += float(((1 - pred) * tv).sum())
        if i >= max_batches:
            break

    # DDP sum
    if dist.is_available() and dist.is_initialized():
        vec = torch.tensor([tp, fp, fn], device=dev, dtype=torch.float32)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        tp, fp, fn = map(float, vec.tolist())

    P = tp / max(tp + fp, 1.0)
    R = tp / max(tp + fn, 1.0)
    F1 = 2 * P * R / max(P + R, 1e-8)
    return {"P": P, "R": R, "F": F1}


def bce_with_logits_posw(logits, targets, pos_weight: float):
    t = targets.clamp(0, 1)
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)


def ori_mse_loss(ori_pred, ori_tgt, ori_mask):
    """
    ori_pred: [B,2,H,W] in [-1,1]
    ori_tgt:  [B,2,H,W]
    ori_mask: [B,1,H,W] in {0,1}
    """
    m = ori_mask
    if m.sum() < 1.0:
        # no positives in batch: return 0 (do not destabilize)
        return ori_pred.new_tensor(0.0)
    diff2 = (ori_pred - ori_tgt) ** 2
    diff2 = diff2 * m  # broadcast over channels
    return diff2.sum() / (m.sum() * ori_pred.size(1))


# -------------------------
# Trainer (same schedule, adds aux loss)
# -------------------------
class TrainerIdea6:
    def __init__(self, device=None, amp_enabled=True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(amp_enabled)

    def init_distributed(self):
        # imported from utils.dist_utils in repo
        from utils.dist_utils import init_distributed
        return init_distributed()

    def is_main_process(self):
        from utils.dist_utils import is_main_process
        return is_main_process()

    def _set_loader_epoch(self, loader, epoch):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(int(epoch))

    def train_full_probe(
        self,
        model,
        train_loader,
        val_loader,
        *,
        resize_masks_to,
        pick_thr_with_floor,
        roc_auc_ddp,
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
        # Idea6 knobs
        ori_weight: float = 0.10,
        ori_only_long: bool = False,
        ori_blur_ksize: int = 7,
        ori_blur_sigma: float = 1.5,
        ori_dilate: int = 3,
        ori_grad_eps: float = 1e-3,
        # baseline schedule knobs
        seed=1337,
        init_head_prior=0.70,
        warmup_epochs=1, warmup_batches=800, warmup_lr=2e-4, warmup_pos_weight=40.0,
        head_epochs=2, head_batches=2000, head_lr=3e-5, head_pos_weight=5.0,
        tail_epochs=2, tail_batches=2500, tail_lr=1.5e-4, tail_pos_weight=2.0,
        max_epochs=60, val_every=3, base_lrs=(3e-4, 2e-4, 1e-4), weight_decay=1e-4,
        thr_beta=1.0, thr_pos_rate_early=(0.03, 0.10), thr_pos_rate_late=(0.08, 0.12),
        save_best_to="ckpt_best.pt", save_last_to="ckpt_last.pt",
        quick_eval_train_batches=6, quick_eval_val_batches=12,
        long_batches: int = 0,
        verbose: int = 2,
    ):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        scaler = amp.GradScaler(enabled=self.amp)

        is_dist, rank, local_rank, world_size = self.init_distributed()

        raw_model = model.to(self.device)
        if is_dist:
            model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        ddp_model = model
        raw_model = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
        seg_model = SegOnlyWrapper(ddp_model)

        # init seg head prior only
        maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        def ensure_aux_trainable_when_head_is():
            if hasattr(raw_model, "aux_head"):
                for p in raw_model.aux_head.parameters():
                    p.requires_grad = True

        # ---------- Warmup ----------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        ensure_aux_trainable_when_head_is()

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=warmup_lr, weight_decay=1e-4)

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, 10_000 + ep)
            ddp_model.train()
            seen, loss_sum = 0, 0.0
            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    seg_logits, ori_pred = ddp_model(xb)
                    yb_r = resize_masks_to(seg_logits, yb)

                    loss_seg = bce_with_logits_posw(seg_logits, yb_r, pos_weight=float(warmup_pos_weight))

                    loss = loss_seg
                    if (not ori_only_long) and (ori_weight > 0):
                        ori_tgt, ori_mask = orientation_target_from_mask(
                            yb_r,
                            blur_ksize=int(ori_blur_ksize),
                            blur_sigma=float(ori_blur_sigma),
                            dilate=int(ori_dilate),
                            grad_eps=float(ori_grad_eps),
                        )
                        loss_ori = ori_mse_loss(ori_pred, ori_tgt, ori_mask)
                        loss = loss + float(ori_weight) * loss_ori

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
                stats = pix_eval(seg_model, resize_masks_to, train_loader, thr=0.2, max_batches=quick_eval_train_batches)
                print(f"[WARMUP] ep{ep} loss {loss_sum / max(seen,1):.4f} | F1 {stats['F']:.3f}")

        # Threshold pick
        thr0, *_ = pick_thr_with_floor(
            seg_model, val_loader, max_batches=200, n_bins=256, beta=2.0,
            min_pos_rate=thr_pos_rate_early[0], max_pos_rate=thr_pos_rate_early[1]
        )
        thr0 = float(thr0)
        if self.is_main_process() and verbose >= 2:
            val_stats = pix_eval(seg_model, resize_masks_to, val_loader, thr=thr0, max_batches=quick_eval_val_batches)
            auc = roc_auc_ddp(seg_model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)
            print(f"[WARMUP VALIDATION] AUC {auc:.3f} | F1 {val_stats['F']:.3f} | thr={thr0:.3f}")

        # ---------- Head ----------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        ensure_aux_trainable_when_head_is()

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=1e-4)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, 20_000 + ep)
            ddp_model.train()
            seen, loss_sum = 0, 0.0
            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    seg_logits, ori_pred = ddp_model(xb)
                    yb_r = resize_masks_to(seg_logits, yb)

                    loss_seg = bce_with_logits_posw(seg_logits, yb_r, pos_weight=float(head_pos_weight))
                    loss = loss_seg

                    if (not ori_only_long) and (ori_weight > 0):
                        ori_tgt, ori_mask = orientation_target_from_mask(
                            yb_r,
                            blur_ksize=int(ori_blur_ksize),
                            blur_sigma=float(ori_blur_sigma),
                            dilate=int(ori_dilate),
                            grad_eps=float(ori_grad_eps),
                        )
                        loss_ori = ori_mse_loss(ori_pred, ori_tgt, ori_mask)
                        loss = loss + float(ori_weight) * loss_ori

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
                stats = pix_eval(seg_model, resize_masks_to, train_loader, thr=thr0, max_batches=quick_eval_train_batches)
                print(f"[HEAD] ep{ep} loss {loss_sum / max(seen,1):.4f} | F1 {stats['F']:.3f}")

        # ---------- Tail probe ----------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        ensure_aux_trainable_when_head_is()
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, 30_000 + ep)
            ddp_model.train()
            seen, loss_sum = 0, 0.0
            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    seg_logits, ori_pred = ddp_model(xb)
                    yb_r = resize_masks_to(seg_logits, yb)

                    loss_seg = bce_with_logits_posw(seg_logits, yb_r, pos_weight=float(tail_pos_weight))
                    loss = loss_seg

                    if (not ori_only_long) and (ori_weight > 0):
                        ori_tgt, ori_mask = orientation_target_from_mask(
                            yb_r,
                            blur_ksize=int(ori_blur_ksize),
                            blur_sigma=float(ori_blur_sigma),
                            dilate=int(ori_dilate),
                            grad_eps=float(ori_grad_eps),
                        )
                        loss_ori = ori_mse_loss(ori_pred, ori_tgt, ori_mask)
                        loss = loss + float(ori_weight) * loss_ori

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

            # Update threshold on val (same pattern as idea5)
            pr_min, pr_max = thr_pos_rate_early
            thr0, _, aux = pick_thr_with_floor(
                seg_model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                min_pos_rate=pr_min, max_pos_rate=pr_max
            )
            thr0 = float(thr0)

            if self.is_main_process() and verbose >= 2:
                val_stats = pix_eval(seg_model, resize_masks_to, val_loader, thr=thr0, max_batches=quick_eval_val_batches)
                print(f"[TAIL] ep{ep} loss {loss_sum / max(seen,1):.4f} | val F1 {val_stats['F']:.3f} | thr={thr0:.3f} | pos_rate≈{aux.get('pos_rate', float('nan')):.3f}")

        # ---------- Long ----------
        best = {"auc": -1.0, "state": None, "thr": float(thr0), "ep": 0, "F": 0.0}

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, 40_000 + ep)
            _ = apply_phase(raw_model, ep)
            ensure_aux_trainable_when_head_is()  # keep aux head aligned with head training

            opt, sched = make_opt_sched(raw_model, ep, tuple(float(x) for x in base_lrs), float(weight_decay))

            ddp_model.train()
            seen, loss_sum = 0, 0.0
            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    seg_logits, ori_pred = ddp_model(xb)
                    yb_r = resize_masks_to(seg_logits, yb)

                    # baseline seg loss: BCEWithLogits (same as other scripts)
                    loss_seg = bce_with_logits_posw(seg_logits, yb_r, pos_weight=float(tail_pos_weight))
                    loss = loss_seg

                    # orientation loss (optionally only LONG)
                    if (ori_weight > 0):
                        ori_tgt, ori_mask = orientation_target_from_mask(
                            yb_r,
                            blur_ksize=int(ori_blur_ksize),
                            blur_sigma=float(ori_blur_sigma),
                            dilate=int(ori_dilate),
                            grad_eps=float(ori_grad_eps),
                        )
                        loss_ori = ori_mse_loss(ori_pred, ori_tgt, ori_mask)
                        loss = loss + float(ori_weight) * loss_ori

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step(ep - 1 + b / max(len(train_loader), 1))

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                if long_batches and b >= int(long_batches):
                    break

            if (ep % int(val_every)) == 0:
                pr_min, pr_max = thr_pos_rate_late if ep >= 15 else thr_pos_rate_early
                metric_thr, _, aux = pick_thr_with_floor(
                    seg_model, val_loader, max_batches=160, n_bins=256, beta=thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                metric_thr = float(metric_thr)

                val_stats = pix_eval(seg_model, resize_masks_to, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                auc = roc_auc_ddp(seg_model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)

                if self.is_main_process() and verbose >= 2:
                    print(
                        f"[LONG] ep{ep:03d} loss {loss_sum/max(seen,1):.4f} | "
                        f"AUC {auc:.3f} | val F1 {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux.get('pos_rate', float('nan')):.3f}"
                    )

                if auc > best["auc"]:
                    best = {
                        "auc": float(auc),
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": float(metric_thr),
                        "ep": int(ep),
                        "F": float(val_stats["F"]),
                    }
                    if self.is_main_process() and save_best_to:
                        torch.save(best, save_best_to)

            if self.is_main_process() and save_last_to:
                torch.save(best, save_last_to)

        if best["state"] is not None:
            raw_model.load_state_dict(best["state"], strict=True)
        return raw_model, float(best["thr"]), best


# -------------------------
# CLI / main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Idea6: UNet + auxiliary orientation head (line coherence bias).")
    ap.add_argument("--repo-root", type=str, default="../")
    ap.add_argument("--train-h5", type=str, default="/home/karlo/train.h5")
    ap.add_argument("--train-csv", type=str, default="../DATA/train.csv")
    ap.add_argument("--test-h5", type=str, default="../DATA/test.h5")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.1)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true", default=False)

    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--long-batches", type=int, default=0)

    # Idea6 knobs
    ap.add_argument("--ori-weight", type=float, default=0.10)
    ap.add_argument("--ori-only-long", action="store_true", default=False)
    ap.add_argument("--ori-blur-ksize", type=int, default=7)
    ap.add_argument("--ori-blur-sigma", type=float, default=1.5)
    ap.add_argument("--ori-dilate", type=int, default=3)
    ap.add_argument("--ori-grad-eps", type=float, default=1e-3)

    ap.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    ap.add_argument("--tag", type=str, default="idea6")
    ap.add_argument("--verbose", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    add_repo_to_syspath(args.repo_root)

    (
        Config,
        H5TiledDataset,
        set_seed,
        split_indices,
        pick_thr_with_floor,
        resize_masks_to,
        roc_auc_ddp,
        init_distributed,
        is_main_process,
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
    ) = import_project()

    init_distributed = init_distributed_once(init_distributed)

    # Init once so samplers see ranks
    is_dist, rank, local_rank, world_size = init_distributed()

    cfg = Config()
    set_seed(int(args.seed))

    base_tr = H5TiledDataset(args.train_h5, tile=int(args.tile), k_sigma=5.0)

    idx_tr, idx_va = split_indices(args.train_h5, val_frac=float(args.val_frac), seed=int(args.seed))
    tiles_tr = filter_tiles_by_panels(base_tr, set(map(int, idx_tr.tolist())))
    tiles_va = filter_tiles_by_panels(base_tr, set(map(int, idx_va.tolist())))

    train_ds = TileSubset(base_tr, tiles_tr)
    val_ds = TileSubset(base_tr, tiles_va)

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

    # Model: Idea6 dual-head
    model = UNetResSEASPP_AuxOri(in_ch=1, widths=(32, 64, 128, 256, 512))

    # Output paths
    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    trainer = TrainerIdea6()
    #cfg.train.warmup_epochs = cfg.train.head_epochs = cfg.train.tail_epochs = 1

    raw_model, thr, best = trainer.train_full_probe(
        model,
        train_loader,
        val_loader,
        resize_masks_to=resize_masks_to,
        pick_thr_with_floor=pick_thr_with_floor,
        roc_auc_ddp=roc_auc_ddp,
        _unfreeze_if_exists=_unfreeze_if_exists,
        apply_phase=apply_phase,
        freeze_all=freeze_all,
        make_opt_sched=make_opt_sched,
        maybe_init_head_bias_to_prior=maybe_init_head_bias_to_prior,
        ori_weight=float(args.ori_weight),
        ori_only_long=bool(args.ori_only_long),
        ori_blur_ksize=int(args.ori_blur_ksize),
        ori_blur_sigma=float(args.ori_blur_sigma),
        ori_dilate=int(args.ori_dilate),
        ori_grad_eps=float(args.ori_grad_eps),
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
    )

    if is_main_process():
        print(f"[DONE] best ep={best['ep']} auc={best['auc']:.3f} F={best['F']:.3f} thr={best['thr']:.3f}")
        print(f"saved: {best_path}")
        print(f"saved: {last_path}")


if __name__ == "__main__":
    main()
