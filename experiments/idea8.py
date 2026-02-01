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

# Based on your existing idea8.py, with the key change:
#   - build 3 buckets in TRAIN: (missed, detected, background)
#   - sample each epoch with fixed mixture fractions (DDP-safe)
# Everything else (masking real_labels in loss, warmup/head/tail/long, ramp) is preserved.
#
# Source baseline: your uploaded idea8.py :contentReference[oaicite:0]{index=0}


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


class XYOnlyLoader:
    """Wrap a DataLoader that yields (x,y,...) and expose it as (x,y) only."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield batch[0], batch[1]


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


def build_tile_mask_from_csv(
    train_csv: str,
    base_ds,
    *,
    tile: int,
    margin_pix: float = 0.0,
    stack_col: str = "stack_detection",
    stack_value: Optional[int] = None,
) -> np.ndarray:
    """
    Returns BASE-length boolean mask over tiles in base_ds.

    stack_value:
      - None: use all rows (any injection)
      - 0: only rows where stack_col==0 (LSST missed)
      - 1: only rows where stack_col==1 (LSST detected)
    """
    cat = pd.read_csv(train_csv)
    need = {"image_id", "x", "y", "trail_length"}
    if stack_value is not None:
        need.add(stack_col)
    miss = need - set(cat.columns)
    if miss:
        raise ValueError(f"train.csv missing required columns: {sorted(miss)}")

    if stack_value is None:
        df = cat
    else:
        df = cat[cat[stack_col].astype(int) == int(stack_value)]

    if len(df) == 0:
        return np.zeros(len(base_ds), dtype=bool)

    with h5py.File(base_ds.h5_path, "r") as f:
        H = int(f["images"].shape[1])
        W = int(f["images"].shape[2])

    Hb = int(np.ceil(H / tile))
    Wb = int(np.ceil(W / tile))

    rc_by_panel: Dict[int, set[Tuple[int, int]]] = {}

    for _, row in df.iterrows():
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

        s = rc_by_panel.setdefault(pid, set())
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                s.add((r, c))

    base_map = {(i, r, c): k for k, (i, r, c) in enumerate(base_ds.indices)}
    mask = np.zeros(len(base_ds), dtype=bool)

    for pid, rcset in rc_by_panel.items():
        for (r, c) in rcset:
            k = base_map.get((pid, r, c))
            if k is not None:
                mask[k] = True

    return mask


class TileSubsetWithRealAndFlags(Dataset):
    """
    Returns (x, y, real, missed_flag, detected_flag)
    - real: tile of real_labels (or zeros if missing)
    - missed_flag/detected_flag are BASE-derived masks (in base_ds indexing)
    """

    def __init__(
        self,
        base,
        base_tile_indices: np.ndarray,
        *,
        train_h5: str,
        real_labels_key: str,
        missed_mask_base: np.ndarray,
        detected_mask_base: np.ndarray,
    ):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
        self.train_h5 = str(train_h5)
        self.real_labels_key = str(real_labels_key)

        missed_mask_base = np.asarray(missed_mask_base, dtype=bool)
        detected_mask_base = np.asarray(detected_mask_base, dtype=bool)
        if missed_mask_base.shape[0] != len(base):
            raise ValueError("missed_mask_base must have length len(base_ds)")
        if detected_mask_base.shape[0] != len(base):
            raise ValueError("detected_mask_base must have length len(base_ds)")
        self.missed_mask_base = missed_mask_base
        self.detected_mask_base = detected_mask_base

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
        detected_flag = bool(self.detected_mask_base[base_idx])

        self._ensure_h5()
        if not self._has_real:
            real = torch.zeros_like(y) if torch.is_tensor(y) else np.zeros_like(y)
            return x, y, real, missed_flag, detected_flag

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

        return x, y, real, missed_flag, detected_flag


# -------------------------
# DDP-safe mixture sampler (missed + detected + background)
# -------------------------
class DistributedMixtureSampler(Sampler[int]):
    """
    Builds an epoch index list with controlled mixture fractions across 3 buckets,
    then shards across DDP ranks.

    dataset indices are LOCAL to the dataset passed to DataLoader (0..len(ds)-1).
    """

    def __init__(
        self,
        *,
        dataset_size: int,
        missed_ids: np.ndarray,
        detected_ids: np.ndarray,
        background_ids: np.ndarray,
        num_replicas: int,
        rank: int,
        seed: int,
        frac_missed: float,
        frac_detected: float,
        frac_background: float,
        epoch_size: Optional[int] = None,
    ):
        self.dataset_size = int(dataset_size)
        self.missed_ids = np.asarray(missed_ids, dtype=np.int64)
        self.detected_ids = np.asarray(detected_ids, dtype=np.int64)
        self.background_ids = np.asarray(background_ids, dtype=np.int64)

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)

        # normalize fractions robustly
        f = np.asarray([frac_missed, frac_detected, frac_background], dtype=float)
        f = np.clip(f, 0.0, None)
        s = float(f.sum())
        if s <= 0:
            f = np.asarray([0.0, 0.0, 1.0], dtype=float)
        else:
            f = f / s
        self.frac_missed = float(f[0])
        self.frac_detected = float(f[1])
        self.frac_background = float(f[2])

        if epoch_size is None:
            self.epoch_size = int(math.ceil(self.dataset_size / max(self.num_replicas, 1)))
        else:
            self.epoch_size = int(epoch_size)

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        g = np.random.default_rng(self.seed + 10_000 * self._epoch + self.rank)

        total = self.epoch_size * self.num_replicas

        # if any bucket is empty, spill its share to background
        fm, fd, fb = self.frac_missed, self.frac_detected, self.frac_background
        if self.missed_ids.size == 0 and fm > 0:
            fb += fm
            fm = 0.0
        if self.detected_ids.size == 0 and fd > 0:
            fb += fd
            fd = 0.0
        if self.background_ids.size == 0 and fb > 0:
            # last resort: use all indices
            self.background_ids = np.arange(self.dataset_size, dtype=np.int64)

        # re-normalize after spill
        s = fm + fd + fb
        if s <= 0:
            fm, fd, fb = 0.0, 0.0, 1.0
        else:
            fm, fd, fb = fm / s, fd / s, fb / s

        nm = int(round(total * fm))
        nd = int(round(total * fd))
        nb = total - nm - nd

        ids_m = g.choice(self.missed_ids, size=nm, replace=True) if nm > 0 else np.empty((0,), np.int64)
        ids_d = g.choice(self.detected_ids, size=nd, replace=True) if nd > 0 else np.empty((0,), np.int64)
        ids_b = g.choice(self.background_ids, size=nb, replace=True) if nb > 0 else np.empty((0,), np.int64)

        ids = np.concatenate([ids_m, ids_d, ids_b], axis=0)
        g.shuffle(ids)

        shard = ids[self.rank : ids.size : self.num_replicas]
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
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
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
            xb, yb, rb, _missed, _detected = batch
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            rb = rb.to(self.device, non_blocking=True)

            logits = model(xb)
            yb_r = resize_masks_to(logits, yb)
            rb_r = resize_masks_to(logits, rb)
            valid = valid_mask_from_real(rb_r)

            tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=thr)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

            if max_batches > 0 and i >= max_batches:
                break

        tp_all = ddp_sum_i64(tp_sum, self.device)
        fp_all = ddp_sum_i64(fp_sum, self.device)
        fn_all = ddp_sum_i64(fn_sum, self.device)

        f1 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
        f2 = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

        auc = float(roc_auc_ddp(model, XYOnlyLoader(loader), n_bins=256, max_batches=12))
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
        # schedule knobs
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
        save_last_to: str,
        best_metric: str,
        long_batches: int = 0,
        verbose: int = 2,
        resume_epoch: int = None,
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
        use_ddp = bool(is_dist and world_size > 1)

        if use_ddp and not isinstance(model, DDP):
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
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
        if resume_epoch is None:
            for ep in range(1, warmup_epochs + 1):
                self._set_loader_epoch(train_loader, seed + 100 + ep)
                model.train()
                seen = 0
                loss_sum = 0.0
                valid_sum = 0.0
                pix_sum = 0.0
                tp_sum = fp_sum = fn_sum = 0

                for b, batch in enumerate(train_loader, 1):
                    xb, yb, rb, _missed, _detected = batch
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
                    tp_sum += tp
                    fp_sum += fp
                    fn_sum += fn

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

                if verbose >= 2 and self.is_main_process() and warmup_epochs > 0:
                    active = ddp_sum_float(valid_sum, self.device)
                    total = ddp_sum_float(pix_sum, self.device)
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
                    xb, yb, rb, _missed, _detected = batch
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
                    tp_sum += tp
                    fp_sum += fp
                    fn_sum += fn

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

                if verbose >= 2 and self.is_main_process() and head_epochs > 0:
                    active = ddp_sum_float(valid_sum, self.device)
                    total = ddp_sum_float(pix_sum, self.device)
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

            # ---- Tail: masked BCE ----
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
                    xb, yb, rb, _missed, _detected = batch
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
                    tp_sum += tp
                    fp_sum += fp
                    fn_sum += fn

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

                if verbose >= 2 and self.is_main_process() and tail_epochs > 0:
                    active = ddp_sum_float(valid_sum, self.device)
                    total = ddp_sum_float(pix_sum, self.device)
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
        if resume_epoch is not None:
            start_epoch = int(resume_epoch) + 1
        else:
            start_epoch = 1
        # ---- Long: masked BCE -> masked FT ----
        best = {"auc": -1.0, "state": None, "thr": FIXED_THR, "ep": 0, "f2": -1e9}

        for ep in range(start_epoch, max_epochs + 1):
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
                xb, yb, rb, _missed, _detected = batch
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
                        loss_ft = focal_tversky_masked(logits, yb_r, valid, alpha=float(ft_alpha), gamma=float(ft_gamma))
                        loss = (1.0 - lam) * loss_bce + lam * loss_ft

                if not torch.isfinite(loss):
                    raise RuntimeError(f"NaN/Inf loss at ep={ep} iter={i}: {float(loss.item())}")

                tp, fp, fn = update_confusion_from_logits(logits, yb_r, valid, thr=FIXED_THR)
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

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

            tp_all = ddp_sum_i64(tp_sum, self.device)
            fp_all = ddp_sum_i64(fp_sum, self.device)
            fn_all = ddp_sum_i64(fn_sum, self.device)
            f1_tr = fbeta_from_counts(tp_all, fp_all, fn_all, beta=1.0)
            f2_tr = fbeta_from_counts(tp_all, fp_all, fn_all, beta=2.0)

            if self.is_main_process() and verbose >= 1:
                active = ddp_sum_float(valid_sum, self.device)
                total = ddp_sum_float(pix_sum, self.device)
                masked_pct = 100.0 * (1.0 - (active / max(total, 1.0)))

                print(
                    f"[EP{ep:02d}] lam={lam:.3f} loss {train_loss:.4f} | thr={FIXED_THR:.3f} | "
                    f"F1={f1_tr:.4f} F2={f2_tr:.4f} | active_pix={active:.0f} masked_out={masked_pct:.2f}% | {time.time()-t0:.1f}s"
                )

                if save_last_to:
                    Path(save_last_to).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": raw_model.state_dict(), "thr": FIXED_THR, "ep": int(ep)}, save_last_to)

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

                auc_v = float(auc) if np.isfinite(float(auc)) else -1e9
                f2_v = float(f2v) if np.isfinite(float(f2v)) else -1e9

                score = auc_v if best_metric == "auc" else f2_v
                best_score = best["auc"] if best_metric == "auc" else best.get("f2", -1e9)

                if score > best_score:
                    best = {
                        "auc": auc_v,
                        "f2": f2_v,
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": FIXED_THR,
                        "ep": int(ep),
                    }
                    if self.is_main_process() and save_best_to:
                        Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "state": best["state"],
                                "thr": best["thr"],
                                "ep": best["ep"],
                                "auc": best["auc"],
                                "f2": best["f2"],
                                "best_metric": best_metric,
                            },
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
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    # Mixture fractions (train stream)
    ap.add_argument("--frac-missed", type=float, default=0.60)
    ap.add_argument("--frac-detected", type=float, default=0.25)
    ap.add_argument("--frac-background", type=float, default=0.15)
    ap.add_argument("--epoch-size", type=int, default=0, help="Samples PER RANK per epoch (0 -> ceil(N/world_size)).")

    # bbox margin for mapping injections -> tiles
    ap.add_argument("--margin-pix", type=float, default=0.0)
    ap.add_argument("--stack-col", type=str, default="stack_detection")

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

    ap.add_argument("--ramp-kind", type=str, default="linear", choices=["linear", "sigmoid"])
    ap.add_argument("--ramp-start-epoch", type=int, default=11)
    ap.add_argument("--ramp-end-epoch", type=int, default=40)
    ap.add_argument("--sigmoid-k", type=float, default=8.0)

    ap.add_argument("--bce-pos-weight-long", type=float, default=8.0)
    ap.add_argument("--ft-alpha", type=float, default=0.45)
    ap.add_argument("--ft-gamma", type=float, default=1.3)

    ap.add_argument("--val-metric-batches", type=int, default=60, help="0 = all")

    ap.add_argument("--save-best-to", type=str, default="../checkpoints/Experiments/Best/idea8.pt")
    ap.add_argument("--save-last-to", type=str, default="../checkpoints/Experiments/Last/idea8.pt")
    ap.add_argument("--best-metric", type=str, default="auc", choices=["auc", "f2"])
    ap.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="If set, skip warmup/head/tail and start long training at this epoch. "
             "Loads weights from --save-last-to by default."
    )

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

    # Build BASE masks from CSV bboxes:
    touched_any_base = build_tile_mask_from_csv(
        args.train_csv,
        base_ds,
        tile=int(args.tile),
        margin_pix=float(args.margin_pix),
        stack_col=str(args.stack_col),
        stack_value=None,
    )
    missed_base = build_tile_mask_from_csv(
        args.train_csv,
        base_ds,
        tile=int(args.tile),
        margin_pix=float(args.margin_pix),
        stack_col=str(args.stack_col),
        stack_value=0,
    )
    detected_base = build_tile_mask_from_csv(
        args.train_csv,
        base_ds,
        tile=int(args.tile),
        margin_pix=float(args.margin_pix),
        stack_col=str(args.stack_col),
        stack_value=1,
    )

    train_ds = TileSubsetWithRealAndFlags(
        base_ds,
        tiles_tr,
        train_h5=args.train_h5,
        real_labels_key=args.real_labels_key,
        missed_mask_base=missed_base,
        detected_mask_base=detected_base,
    )
    val_ds = TileSubsetWithRealAndFlags(
        base_ds,
        tiles_va,
        train_h5=args.train_h5,
        real_labels_key=args.real_labels_key,
        missed_mask_base=missed_base,
        detected_mask_base=detected_base,
    )

    # Build TRAIN-local bucket ids.
    # Background = tiles in train split NOT touched by any injection bbox.
    missed_local: List[int] = []
    detected_local: List[int] = []
    background_local: List[int] = []

    base_indices = train_ds.base_tile_indices
    for j in range(len(train_ds)):
        bidx = int(base_indices[j])
        if bool(missed_base[bidx]):
            missed_local.append(j)
        elif bool(detected_base[bidx]):
            detected_local.append(j)
        else:
            # If it is touched_any but not in missed/detected due to CSV quirks, treat as detected-like.
            if bool(touched_any_base[bidx]):
                detected_local.append(j)
            else:
                background_local.append(j)

    missed_local = np.asarray(missed_local, dtype=np.int64)
    detected_local = np.asarray(detected_local, dtype=np.int64)
    background_local = np.asarray(background_local, dtype=np.int64)

    if is_main_process():
        print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")
        try:
            with h5py.File(args.train_h5, "r") as f:
                print(f"real_labels present: {args.real_labels_key in f}")
        except Exception as e:
            print(f"WARNING: could not inspect H5 for real_labels: {e}")
        print(f"TRAIN bucket sizes: missed={missed_local.size} detected={detected_local.size} background={background_local.size}")
        fm, fd, fb = float(args.frac_missed), float(args.frac_detected), float(args.frac_background)
        s = fm + fd + fb
        if s > 0:
            fm, fd, fb = fm / s, fd / s, fb / s
        print(f"TRAIN mixture target: missed={fm:.3f} detected={fd:.3f} background={fb:.3f}")

    # Samplers
    epoch_size = None if int(args.epoch_size) <= 0 else int(args.epoch_size)
    use_ddp = bool(is_dist and world_size > 1)

    if use_ddp:
        train_sampler = DistributedMixtureSampler(
            dataset_size=len(train_ds),
            missed_ids=missed_local,
            detected_ids=detected_local,
            background_ids=background_local,
            num_replicas=world_size,
            rank=rank,
            seed=int(args.seed),
            frac_missed=float(args.frac_missed),
            frac_detected=float(args.frac_detected),
            frac_background=float(args.frac_background),
            epoch_size=epoch_size,
        )
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = DistributedMixtureSampler(
            dataset_size=len(train_ds),
            missed_ids=missed_local,
            detected_ids=detected_local,
            background_ids=background_local,
            num_replicas=1,
            rank=0,
            seed=int(args.seed),
            frac_missed=float(args.frac_missed),
            frac_detected=float(args.frac_detected),
            frac_background=float(args.frac_background),
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
    if args.resume_epoch is not None:
        if is_main_process():
            print(f"Resuming from epoch {args.resume_epoch} using weights from {args.save_last_to}")
        checkpoint = torch.load(args.save_last_to, map_location="cpu")
        model = UNetResSEASPP(in_ch=1, out_ch=1, widths=(48, 96, 192, 384, 768)).to(device)
        model.load_state_dict(checkpoint["state"])
    else:
        model = UNetResSEASPP(in_ch=1, out_ch=1, widths=(48, 96, 192, 384, 768)).to(device)

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
        save_last_to=str(args.save_last_to),
        best_metric=str(args.best_metric),
        long_batches=int(args.long_batches),
        verbose=int(args.verbose),
        resume_epoch = args.resume_epoch,

    )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
