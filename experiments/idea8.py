from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
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
    from thresholds import resize_masks_to
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
        resize_masks_to,
        init_distributed,
        is_main_process,
    )


# -------------------------
# DDP reduce helpers
# -------------------------
def ddp_sum_float(x: float, device: torch.device) -> float:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([x], device=device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    return float(x)


def ddp_sum_i64(x: int, device: torch.device) -> int:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([int(x)], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())
    return int(x)


# -------------------------
# Dataset helpers
# -------------------------
def filter_tiles_by_panels(base_ds, panel_id_set: set[int]) -> np.ndarray:
    kept = [k for k, (pid, _r, _c) in enumerate(base_ds.indices) if int(pid) in panel_id_set]
    return np.asarray(kept, dtype=np.int64)


def tiles_touched_by_bbox(
    xc: float, yc: float, R: float, H: int, W: int, tile: int
) -> Tuple[int, int, int, int]:
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
    cat = pd.read_csv(train_csv)
    need = {"image_id", "x", "y", "trail_length", stack_col}
    miss = need - set(cat.columns)
    if miss:
        raise ValueError(f"train.csv missing required columns: {sorted(miss)}")

    miss_df = cat[cat[stack_col].astype(int) == 0].copy()
    if len(miss_df) == 0:
        return np.zeros(len(base_ds), dtype=bool)

    with h5py.File(base_ds.h5_path, "r") as f:
        H = int(f["images"].shape[1])
        W = int(f["images"].shape[2])

    Hb = int(np.ceil(H / tile))
    Wb = int(np.ceil(W / tile))

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
    Returns (x, y, real, missed_flag)
    - real: tile of real_labels (or zeros if missing)
    - missed_flag: bool per tile from CSV-derived mask in BASE indexing
    """
    def __init__(
        self,
        base,
        base_tile_indices: np.ndarray,
        *,
        train_h5: str,
        real_labels_key: str,
        missed_mask_base: np.ndarray,
    ):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
        self.train_h5 = str(train_h5)
        self.real_labels_key = str(real_labels_key)

        missed_mask_base = np.asarray(missed_mask_base, dtype=bool)
        if missed_mask_base.shape[0] != len(base):
            raise ValueError("missed_mask_base must have length len(base_ds)")
        self.missed_mask_base = missed_mask_base

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
        panel_i = int(panel_i); r = int(r); c = int(c)
        t = int(self.base.tile)

        rl = self._h5[self.real_labels_key]  # [N,H,W]
        H, W = int(rl.shape[1]), int(rl.shape[2])

        r0, c0 = r * t, c * t
        r1, c1 = min(r0 + t, H), min(c0 + t, W)
        real_np = rl[panel_i, r0:r1, c0:c1]

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


# -------------------------
# Idea 1: DDP-safe stratified sampler (missed oversampling)
# -------------------------
class DistributedStratifiedMissedSampler(Sampler[int]):
    """
    Builds an epoch index list with a controlled missed fraction,
    then shards across DDP ranks.

    dataset indices are LOCAL to the dataset passed to DataLoader (0..len(ds)-1).
    """
    def __init__(
        self,
        *,
        dataset_size: int,
        miss_ids: np.ndarray,
        norm_ids: np.ndarray,
        num_replicas: int,
        rank: int,
        seed: int,
        missed_frac: float,
        epoch_size: Optional[int] = None,
    ):
        self.dataset_size = int(dataset_size)
        self.miss_ids = np.asarray(miss_ids, dtype=np.int64)
        self.norm_ids = np.asarray(norm_ids, dtype=np.int64)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.missed_frac = float(missed_frac)

        if epoch_size is None:
            self.epoch_size = int(math.ceil(self.dataset_size / max(self.num_replicas, 1)))
        else:
            self.epoch_size = int(epoch_size)

        if self.miss_ids.size == 0 or self.norm_ids.size == 0:
            self.missed_frac = 0.0

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        g = np.random.default_rng(self.seed + 10_000 * self._epoch + self.rank)

        total = self.epoch_size * self.num_replicas

        if self.missed_frac <= 0.0:
            ids = np.arange(self.dataset_size, dtype=np.int64)
            g.shuffle(ids)
            if ids.size < total:
                reps = int(math.ceil(total / ids.size))
                ids = np.tile(ids, reps)
            ids = ids[:total]
        else:
            nm = int(round(total * self.missed_frac))
            nn = total - nm

            miss = g.choice(self.miss_ids, size=nm, replace=True)
            norm = g.choice(self.norm_ids, size=nn, replace=True)

            ids = np.concatenate([miss, norm], axis=0)
            g.shuffle(ids)

        shard = ids[self.rank: ids.size: self.num_replicas]
        shard = shard[: self.epoch_size]
        return iter(shard.tolist())


# -------------------------
# Masks / metrics / losses
# -------------------------
def valid_mask_from_real(real: torch.Tensor) -> torch.Tensor:
    if real.dtype != torch.bool:
        real = real > 0.5
    return (~real).to(dtype=torch.float32)


@torch.no_grad()
def update_confusion_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    thr: float = 0.5,
) -> Tuple[int, int, int]:
    p = torch.sigmoid(logits.float())
    pred = p >= float(thr)
    gt = targets > 0.5
    v = valid > 0.5

    tp = int((pred & gt & v).sum().item())
    fp = int((pred & (~gt) & v).sum().item())
    fn = int(((~pred) & gt & v).sum().item())
    return tp, fp, fn


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float) -> float:
    tp = float(tp); fp = float(fp); fn = float(fn)
    prec = tp / max(tp + fp, 1.0)
    rec  = tp / max(tp + fn, 1.0)
    b2 = float(beta) ** 2
    denom = (b2 * prec + rec)
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * prec * rec / denom


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    t = targets.float().clamp(0.0, 1.0)
    posw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_map = F.binary_cross_entropy_with_logits(logits, t, pos_weight=posw, reduction="none")
    num = (loss_map * valid).sum()
    den = valid.sum().clamp_min(1.0)
    return num / den


def focal_tversky_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits_f = logits.float()
    t = targets.float().clamp(0.0, 1.0)
    v = valid.float()

    p = torch.sigmoid(logits_f).clamp(eps, 1.0 - eps)
    w = v

    TP = (w * p * t).sum(dim=(1, 2, 3))
    FP = (w * p * (1.0 - t)).sum(dim=(1, 2, 3))
    FN = (w * (1.0 - p) * t).sum(dim=(1, 2, 3))

    denom = TP + float(alpha) * FP + (1.0 - float(alpha)) * FN + eps
    tv = (TP + eps) / denom
    tv = tv.clamp(0.0, 1.0)

    loss = torch.pow(1.0 - tv, float(gamma))
    return loss.mean()


def ramp_lambda(ep: int, *, kind: str, e0: int, e1: int, k: float) -> float:
    if e1 <= e0:
        return 1.0 if ep >= e0 else 0.0
    if kind == "linear":
        x = (ep - e0) / float(e1 - e0)
        return float(np.clip(x, 0.0, 1.0))
    if kind == "sigmoid":
        x = np.clip((ep - e0) / float(e1 - e0), 0.0, 1.0)
        z = (x * 2.0 - 1.0) * float(k)
        return float(1.0 / (1.0 + np.exp(-z)))
    raise ValueError(kind)


# -------------------------
# Trainer
# -------------------------
class TrainerIdea8:
    def __init__(self, *, init_distributed, is_main_process, device: torch.device, use_amp: bool = True):
        self.init_distributed = init_distributed
        self.is_main_process = is_main_process
        self.device = device
        self.amp = bool(use_amp)

    def _set_loader_epoch(self, loader, epoch: int):
        s = getattr(loader, "sampler", None)
        if s is not None and hasattr(s, "set_epoch"):
            s.set_epoch(int(epoch))

    @torch.no_grad()
    def eval_val_f1f2_auc(
        self,
        model,
        loader,
        *,
        thr: float,
        max_batches: int,
        roc_auc_ddp,
        resize_masks_to,
    ) -> Tuple[float, float, float]:
        model.eval()
        tp_sum = fp_sum = fn_sum = 0

        for i, batch in enumerate(loader, 1):
            xb, yb, rb, _missed = batch
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            rb = rb.to(self.device, non_blocking=True)

            logits = model(xb)
            yb_r = resize_masks_to(logits, yb)
            rb_r = resize_masks_to(logits, rb)
            valid = valid_mask_from_real(rb_r)

            tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=thr)
            tp_sum += tp; fp_sum += fp; fn_sum += fn

            if max_batches > 0 and i >= max_batches:
                break

        tp_all = ddp_sum_i64(tp_sum, self.device)
        fp_all = ddp_sum_i64(fp_sum, self.device)
        fn_all = ddp_sum_i64(fn_sum, self.device)

        f1 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
        f2 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

        auc = float(roc_auc_ddp(model, loader, n_bins=256, max_batches=12))
        return f1, f2, auc

    def train_full_probe(
        self,
        model,
        *,
        train_loader,
        val_loader,
        seed: int,
        init_head_prior: float,
        # warmup/head/tail
        warmup_epochs: int,
        warmup_batches: int,
        warmup_lr: float,
        warmup_pos_weight: float,
        head_epochs: int,
        head_batches: int,
        head_lr: float,
        head_pos_weight: float,
        tail_epochs: int,
        tail_batches: int,
        tail_lr: float,
        tail_pos_weight: float,
        # long
        max_epochs: int,
        val_every: int,
        base_lrs: Tuple[float, float, float],
        weight_decay: float,
        # idea3 schedule knobs
        ramp_kind: str,
        ramp_start_epoch: int,
        ramp_end_epoch: int,
        sigmoid_k: float,
        bce_pos_weight_long: float,
        ft_alpha: float,
        ft_gamma: float,
        # fixed threshold + metrics
        fixed_thr: float,
        val_metric_batches: int,
        # io
        save_best_to: str,
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
            resize_masks_to,
            _init_distributed,
            _is_main_process,
        ) = import_project()

        FIXED_THR = float(fixed_thr)

        is_dist, rank, local_rank, world_size = self.init_distributed()
        if is_dist and not isinstance(model, DDP):
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        raw_model = model.module if isinstance(model, DDP) else model

        scaler = amp.GradScaler("cuda", enabled=self.amp)
        maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        # ---- Warmup: masked BCE ----
        freeze_all(raw_model)
        for p in raw_model.parameters():
            p.requires_grad = True

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=warmup_lr, weight_decay=0.0)

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 100 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            valid_sum = 0.0
            pix_sum = 0.0
            tp_sum = fp_sum = fn_sum = 0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb, _missed = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = valid_mask_from_real(rb_r)
                    loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight=float(warmup_pos_weight))

                tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=FIXED_THR)
                tp_sum += tp; fp_sum += fp; fn_sum += fn

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
                total  = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))

                tp_all = ddp_sum_i64(tp_sum, self.device)
                fp_all = ddp_sum_i64(fp_sum, self.device)
                fn_all = ddp_sum_i64(fn_sum, self.device)
                f1 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
                f2 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

                print(
                    f"[WARMUP] ep{ep} loss {loss_sum / max(seen,1):.4f} | thr={FIXED_THR:.3f} | "
                    f"F1={f1:.4f} F2={f2:.4f} | active_pix={active:.0f} masked_out={masked_pct:.2f}%"
                )

        # ---- Head: masked BCE ----
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=1e-4)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 200 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            valid_sum = 0.0
            pix_sum = 0.0
            tp_sum = fp_sum = fn_sum = 0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb, _missed = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = valid_mask_from_real(rb_r)
                    loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight=float(head_pos_weight))

                tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=FIXED_THR)
                tp_sum += tp; fp_sum += fp; fn_sum += fn

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
                total  = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))

                tp_all = ddp_sum_i64(tp_sum, self.device)
                fp_all = ddp_sum_i64(fp_sum, self.device)
                fn_all = ddp_sum_i64(fn_sum, self.device)
                f1 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
                f2 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

                print(
                    f"[HEAD] ep{ep} loss {loss_sum / max(seen,1):.4f} | thr={FIXED_THR:.3f} | "
                    f"F1={f1:.4f} F2={f2:.4f} | active_pix={active:.0f} masked_out={masked_pct:.2f}%"
                )

        # ---- Tail: masked BCE (no threshold search) ----
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 300 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            valid_sum = 0.0
            pix_sum = 0.0
            tp_sum = fp_sum = fn_sum = 0

            for b, batch in enumerate(train_loader, 1):
                xb, yb, rb, _missed = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = valid_mask_from_real(rb_r)
                    loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight=float(tail_pos_weight))

                tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=FIXED_THR)
                tp_sum += tp; fp_sum += fp; fn_sum += fn

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

            if verbose >= 2 and self.is_main_process():
                active = ddp_sum_float(valid_sum, self.device)
                total  = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))

                tp_all = ddp_sum_i64(tp_sum, self.device)
                fp_all = ddp_sum_i64(fp_sum, self.device)
                fn_all = ddp_sum_i64(fn_sum, self.device)
                f1 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
                f2 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

                print(
                    f"[TAIL] ep{ep} loss {loss_sum / max(seen,1):.4f} | thr={FIXED_THR:.3f} | "
                    f"F1={f1:.4f} F2={f2:.4f} | active_pix={active:.0f} masked_out={masked_pct:.2f}%"
                )

        # ---- Long: Idea 3 schedule (masked BCE -> masked FT) ----
        best = {"auc": -1.0, "state": None, "thr": FIXED_THR, "ep": 0, "f2": 0.0}

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, seed + 1000 + ep)

            _ = apply_phase(raw_model, ep)
            opt, sched = make_opt_sched(raw_model, ep, tuple(float(x) for x in base_lrs), float(weight_decay))

            lam = ramp_lambda(
                ep,
                kind=str(ramp_kind),
                e0=int(ramp_start_epoch),
                e1=int(ramp_end_epoch),
                k=float(sigmoid_k),
            )

            model.train()
            seen = 0
            loss_sum = 0.0
            t0 = time.time()

            valid_sum = 0.0
            pix_sum = 0.0
            tp_sum = fp_sum = fn_sum = 0

            for i, batch in enumerate(train_loader, 1):
                xb, yb, rb, _missed = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = valid_mask_from_real(rb_r)

                    loss_bce = masked_bce_with_logits(logits, yb_r, valid, pos_weight=float(bce_pos_weight_long))
                    if lam <= 0.0:
                        loss = loss_bce
                    else:
                        loss_ft = focal_tversky_masked(
                            logits, yb_r, valid, alpha=float(ft_alpha), gamma=float(ft_gamma)
                        )
                        loss = (1.0 - lam) * loss_bce + lam * loss_ft

                if not torch.isfinite(loss):
                    raise RuntimeError(f"NaN/Inf loss at ep={ep} iter={i}: {float(loss.item())}")

                tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=FIXED_THR)
                tp_sum += tp; fp_sum += fp; fn_sum += fn

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

            # train metrics
            tp_all = ddp_sum_i64(tp_sum, self.device)
            fp_all = ddp_sum_i64(fp_sum, self.device)
            fn_all = ddp_sum_i64(fn_sum, self.device)
            f1_tr = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
            f2_tr = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

            if self.is_main_process() and verbose >= 1:
                active = ddp_sum_float(valid_sum, self.device)
                total  = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))

                print(
                    f"[EP{ep:02d}] lam={lam:.3f} loss {train_loss:.4f} | thr={FIXED_THR:.3f} | "
                    f"F1={f1_tr:.4f} F2={f2_tr:.4f} | active_pix={active:.0f} masked_out={masked_pct:.2f}% | {time.time()-t0:.1f}s"
                )

            if (ep % val_every == 0) or (ep <= 3):
                f1v, f2v, auc = self.eval_val_f1f2_auc(
                    model,
                    val_loader,
                    thr=FIXED_THR,
                    max_batches=int(val_metric_batches),
                    roc_auc_ddp=roc_auc_ddp,
                    resize_masks_to=resize_masks_to,
                )

                if self.is_main_process():
                    print(f"[VAL ep{ep}] AUC {auc:.3f} | thr={FIXED_THR:.3f} | F1={f1v:.4f} F2={f2v:.4f}")

                # keep baseline criterion: best by AUC (thr fixed)
                if float(auc) > best["auc"]:
                    best = {
                        "auc": float(auc),
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": FIXED_THR,
                        "ep": int(ep),
                        "f2": float(f2v),
                    }
                    if self.is_main_process():
                        Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"state": best["state"], "thr": best["thr"], "ep": best["ep"], "auc": best["auc"], "f2": best["f2"]},
                            save_best_to,
                        )

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

    # Idea 1 oversampling
    ap.add_argument("--missed-frac", type=float, default=0.35,
                    help="Target fraction of LSST-missed tiles in sampled stream (0..1).")
    ap.add_argument("--epoch-size", type=int, default=0,
                    help="Samples PER RANK per epoch for sampler (0 -> ceil(N/world_size)).")

    # missed-tiling bbox margin
    ap.add_argument("--margin-pix", type=float, default=0.0)

    # fixed threshold
    ap.add_argument("--fixed-thr", type=float, default=0.5)

    # training defaults
    ap.add_argument("--init-head-prior", type=float, default=0.70)

    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--warmup-batches", type=int, default=800)
    ap.add_argument("--warmup-lr", type=float, default=2e-4)
    ap.add_argument("--warmup-pos-weight", type=float, default=40.0)

    ap.add_argument("--head-epochs", type=int, default=10)
    ap.add_argument("--head-batches", type=int, default=2000)
    ap.add_argument("--head-lr", type=float, default=3e-5)
    ap.add_argument("--head-pos-weight", type=float, default=5.0)

    ap.add_argument("--tail-epochs", type=int, default=6)
    ap.add_argument("--tail-batches", type=int, default=2500)
    ap.add_argument("--tail-lr", type=float, default=1.5e-4)
    ap.add_argument("--tail-pos-weight", type=float, default=2.0)

    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--base-lrs", type=float, nargs=3, default=[3e-4, 2e-4, 1e-4])
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--long-batches", type=int, default=0)

    # Idea 3 schedule (LONG only)
    ap.add_argument("--ramp-kind", type=str, default="linear", choices=["linear", "sigmoid"])
    ap.add_argument("--ramp-start-epoch", type=int, default=11)
    ap.add_argument("--ramp-end-epoch", type=int, default=40)
    ap.add_argument("--sigmoid-k", type=float, default=8.0)

    ap.add_argument("--bce-pos-weight-long", type=float, default=8.0)
    ap.add_argument("--ft-alpha", type=float, default=0.45)
    ap.add_argument("--ft-gamma", type=float, default=1.3)

    # metrics on val
    ap.add_argument("--val-metric-batches", type=int, default=60,
                    help="How many val batches to use for F1/F2 (0 = all).")

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
    # val also includes real_labels for masking metrics
    val_ds = TileSubsetWithRealAndMissed(
        base_ds,
        tiles_va,
        train_h5=args.train_h5,
        real_labels_key=args.real_labels_key,
        missed_mask_base=missed_mask_base,  # unused for val metrics, but needed for shape
    )

    # Build missed/non-missed lists in TRAIN dataset local indexing (0..len(train_ds)-1)
    missed_local: List[int] = []
    normal_local: List[int] = []
    base_indices = train_ds.base_tile_indices
    for j in range(len(train_ds)):
        bidx = int(base_indices[j])
        (missed_local if bool(missed_mask_base[bidx]) else normal_local).append(j)

    missed_local = np.asarray(missed_local, dtype=np.int64)
    normal_local = np.asarray(normal_local, dtype=np.int64)

    if is_main_process():
        print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")
        try:
            with h5py.File(args.train_h5, "r") as f:
                print(f"real_labels present: {args.real_labels_key in f}")
        except Exception as e:
            print(f"WARNING: could not inspect H5 for real_labels: {e}")
        print(f"Missed BASE tiles (from CSV): {int(missed_mask_base.sum())} / {len(missed_mask_base)}")
        print(f"Missed TRAIN tiles: {missed_local.size} / {len(train_ds)}")
        print(f"Sampler missed_frac target: {float(args.missed_frac):.3f}")

    # Sampler: Idea 1 oversampling
    epoch_size = None if int(args.epoch_size) <= 0 else int(args.epoch_size)
    if is_dist:
        train_sampler = DistributedStratifiedMissedSampler(
            dataset_size=len(train_ds),
            miss_ids=missed_local,
            norm_ids=normal_local,
            num_replicas=world_size,
            rank=rank,
            seed=int(args.seed),
            missed_frac=float(args.missed_frac),
            epoch_size=epoch_size,
        )
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = DistributedStratifiedMissedSampler(
            dataset_size=len(train_ds),
            miss_ids=missed_local,
            norm_ids=normal_local,
            num_replicas=1,
            rank=0,
            seed=int(args.seed),
            missed_frac=float(args.missed_frac),
            epoch_size=epoch_size,
        )
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
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
        warmup_pos_weight=float(args.warmup_pos_weight),

        head_epochs=int(args.head_epochs),
        head_batches=int(args.head_batches),
        head_lr=float(args.head_lr),
        head_pos_weight=float(args.head_pos_weight),

        tail_epochs=int(args.tail_epochs),
        tail_batches=int(args.tail_batches),
        tail_lr=float(args.tail_lr),
        tail_pos_weight=float(args.tail_pos_weight),

        max_epochs=int(args.max_epochs),
        val_every=int(args.val_every),
        base_lrs=(float(args.base_lrs[0]), float(args.base_lrs[1]), float(args.base_lrs[2])),
        weight_decay=float(args.weight_decay),

        ramp_kind=str(args.ramp_kind),
        ramp_start_epoch=int(args.ramp_start_epoch),
        ramp_end_epoch=int(args.ramp_end_epoch),
        sigmoid_k=float(args.sigmoid_k),

        bce_pos_weight_long=float(args.bce_pos_weight_long),
        ft_alpha=float(args.ft_alpha),
        ft_gamma=float(args.ft_gamma),

        fixed_thr=float(args.fixed_thr),
        val_metric_batches=int(args.val_metric_batches),

        save_best_to=str(args.save_best_to),
        long_batches=int(args.long_batches),
        verbose=int(args.verbose),
    )

    if is_main_process():
        print("Final threshold (fixed):", thr)
        print("Summary:", summary)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
