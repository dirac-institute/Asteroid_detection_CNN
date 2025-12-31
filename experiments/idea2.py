#!/usr/bin/env python3
"""
Idea 2 — Baseline trainer + LONG-loss bias toward LSST-missed examples (hard tiles)

Goal:
- Reuse the same training process as the default Trainer.train_full_probe (warmup/head/tail/long)
- Change ONLY the loss in the LONG loop by adding an extra penalty on POSITIVE pixels in HARD tiles
- Keep training budget comparable across Ideas (same phases, same schedulers, same eval cadence)

This is a standalone script (does NOT modify train.py).
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
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
    from losses import BCEIoUEdge, blended_loss, make_loss_for_epoch
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
        BCEIoUEdge,
        blended_loss,
        make_loss_for_epoch,
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
# Config
# -------------------------
@dataclass
class Cfg:
    repo_root: str
    train_h5: str
    train_csv: str
    test_h5: str

    tile: int = 128
    seed: int = 1337

    # dataloader
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    # baseline-ish defaults (you will pass exact values in train_full_probe call)
    pos_weight: float = 8.0

    # hard-tile construction from CSV
    margin_pix: float = 8.0
    min_pos_pix: int = 1

    # Idea2 weighting (LONG only)
    hard_pos_boost: float = 4.0
    ramp_kind: str = "linear"  # "linear" or "sigmoid"
    ramp_start_epoch: int = 1
    ramp_end_epoch: int = 60
    sigmoid_k: float = 8.0

    # outputs
    save_best_to: str = "ckpt_best_idea2.pt"
    save_last_to: str = "ckpt_last_idea2.pt"


# -------------------------
# Hard tiles from CSV + truth filter
# -------------------------
def tiles_touched_by_bbox(x: float, y: float, R: float, H: int, W: int, tile: int) -> Tuple[int, int, int, int]:
    x0 = max(0.0, x - R)
    x1 = min(float(W - 1), x + R)
    y0 = max(0.0, y - R)
    y1 = min(float(H - 1), y + R)

    c0 = int(math.floor(x0 / tile))
    c1 = int(math.floor(x1 / tile))
    r0 = int(math.floor(y0 / tile))
    r1 = int(math.floor(y1 / tile))
    return r0, r1, c0, c1


def build_hard_mask_base_from_csv(train_csv: str, base_ds, *, margin_pix: float) -> np.ndarray:
    """
    Boolean hard mask in BASE tile index space (len = len(base_ds)).
    Hard if touched by any missed injection bbox (stack_detection == 0).
    """
    tile = int(base_ds.tile)
    N, H, W = int(base_ds.N), int(base_ds.H), int(base_ds.W)
    Hb = math.ceil(H / tile)
    Wb = math.ceil(W / tile)

    df = pd.read_csv(train_csv)
    for col in ["image_id", "x", "y", "trail_length", "stack_detection"]:
        if col not in df.columns:
            raise ValueError(f"train_csv missing column '{col}'")

    hard_rc_by_panel: Dict[int, set] = {}
    missed = df[df["stack_detection"] == 0]

    for _, row in missed.iterrows():
        pid = int(row["image_id"])
        if pid < 0 or pid >= N:
            continue

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


def filter_tiles_by_truth(train_h5: str, base_ds, base_tile_indices: np.ndarray, *, min_pos_pix: int) -> np.ndarray:
    """
    Keep only those BASE tile indices whose truth mask has >= min_pos_pix positives.
    """
    base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
    kept: list[int] = []

    with h5py.File(train_h5, "r") as f:
        Y = f["masks"]  # [N,H,W]
        H, W = Y.shape[1:]
        t = int(base_ds.tile)

        for idx in base_tile_indices:
            i, r, c = base_ds.indices[int(idx)]
            r0, c0 = r * t, c * t
            r1, c1 = min(r0 + t, H), min(c0 + t, W)
            y = Y[i, r0:r1, c0:c1]
            if int(np.count_nonzero(y)) >= int(min_pos_pix):
                kept.append(int(idx))

    return np.asarray(kept, dtype=np.int64)


def filter_tiles_by_panels(base_ds, panel_id_set: set[int]) -> np.ndarray:
    kept = [k for k, (pid, _r, _c) in enumerate(base_ds.indices) if int(pid) in panel_id_set]
    return np.asarray(kept, dtype=np.int64)


class TileSubsetWithId(Dataset):
    """
    Subset of base tiled dataset by BASE tile indices.
    Returns (x, y, tid) where tid is DATASET INDEX (0..len-1) for this subset.
    That makes hard_mask_train[tid] valid.
    """

    def __init__(self, base, base_tile_indices: np.ndarray):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)

    def __len__(self):
        return len(self.base_tile_indices)

    def __getitem__(self, i: int):
        x, y = self.base[int(self.base_tile_indices[int(i)])]
        return x, y, int(i)


class TileSubset(Dataset):
    def __init__(self, base, base_tile_indices: np.ndarray):
        self.base = base
        self.base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)

    def __len__(self):
        return len(self.base_tile_indices)

    def __getitem__(self, i: int):
        return self.base[int(self.base_tile_indices[int(i)])]


# -------------------------
# Eval helpers
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
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch

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
# Trainer: baseline + Idea2 tweak in LONG only
# -------------------------
class Trainer:
    def __init__(self, init_distributed, is_main_process, *, device=None, use_amp=True):
        self.init_distributed = init_distributed
        self.is_main_process = is_main_process
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(use_amp)

    def _set_loader_epoch(self, loader, epoch: int):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
            return
        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
            return

    def train_full_probe(
            self,
            model,
            train_loader,
            val_loader,
            *,
            hard_mask_train: Optional[np.ndarray],
            hard_pos_boost: float = 4.0,
            ramp_kind: str = "linear",
            ramp_start_epoch: int = 1,
            ramp_end_epoch: int = 60,
            sigmoid_k: float = 8.0,
            pos_weight_long: float = 8.0,
            # baseline knobs (pass same values as your baseline experiments)
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
            save_best_to="ckpt_best.pt",
            save_last_to="ckpt_last.pt",
            quick_eval_train_batches=6,
            quick_eval_val_batches=12,
            verbose=2,
            # injected from import_project()
            resize_masks_to=None,
            pick_thr_with_floor=None,
            roc_auc_ddp=None,
            maybe_init_head_bias_to_prior=None,
            apply_phase=None,
            freeze_all=None,
            _unfreeze_if_exists=None,
            make_loss_for_epoch=None,
            blended_loss=None,
            BCEIoUEdge=None,
            make_opt_sched=None,
    ):
        assert resize_masks_to is not None
        assert pick_thr_with_floor is not None
        assert roc_auc_ddp is not None

        is_dist, rank, local_rank, world_size = self.init_distributed()
        scaler = amp.GradScaler("cuda", enabled=self.amp)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if hard_mask_train is not None:
            hard_mask_train = np.asarray(hard_mask_train, dtype=bool)
            assert hard_mask_train.ndim == 1

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

        # -------- Warmup (BCE) --------
        freeze_all(raw_model)
        for p in raw_model.parameters():
            p.requires_grad = True

        posw = torch.tensor(warmup_pos_weight, device=self.device)
        opt = torch.optim.Adam(raw_model.parameters(), lr=warmup_lr, weight_decay=0.0)

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0], batch[1]
                xb, yb = xb.to(self.device), yb.to(self.device)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=posw)

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

            stats = pix_eval(model, resize_masks_to, train_loader, thr=0.5, max_batches=quick_eval_train_batches)
            if self.is_main_process() and verbose >= 2:
                print(
                    f"[WARMUP] ep{ep} loss {loss_sum / max(seen, 1):.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        # initial thr
        thr0 = 0.5
        val_stats = pix_eval(model, resize_masks_to, val_loader, thr=float(thr0), max_batches=quick_eval_val_batches)
        auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)
        if self.is_main_process() and verbose >= 2:
            print(
                f"[WARMUP VALIDATION] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f} | thr={float(thr0):.3f}")

        # -------- Head-only (BCE) --------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True

        head_posw = torch.tensor(head_pos_weight, device=self.device)
        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=0.0)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, warmup_epochs + 1 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0], batch[1]
                xb, yb = xb.to(self.device), yb.to(self.device)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=head_posw)

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

            stats = pix_eval(model, resize_masks_to, train_loader, thr=float(thr0),
                             max_batches=quick_eval_train_batches)
            if self.is_main_process() and verbose >= 2:
                print(
                    f"[HEAD] ep{ep} loss {loss_sum / max(seen, 1):.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        # -------- Tail probe (gentle) --------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        core_probe = BCEIoUEdge(lambda_bce=0.9, pos_weight=tail_pos_weight, lambda_edge=0.0).to(self.device)
        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, warmup_epochs + 1 + head_epochs + 1 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                xb, yb = batch[0], batch[1]
                xb, yb = xb.to(self.device), yb.to(self.device)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = core_probe(logits, yb_r)

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

        # -------- Long training (baseline + Idea2 tweak) --------
        best = {"auc": -1.0, "state": None, "thr": float(thr0), "ep": 0, "P": 0.0, "R": 0.0, "F": 0.0}
        metric_thr = float(thr0)
        val_every = int(val_every)
        posw_long = torch.tensor(float(pos_weight_long), device=self.device)

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, warmup_epochs + 1 + head_epochs + 1 + tail_epochs + 1 + ep)

            _ = apply_phase(raw_model, ep)
            core, aftl, w = make_loss_for_epoch(ep, self.device)
            opt, sched = make_opt_sched(raw_model, ep, base_lrs, weight_decay)

            lam = ramp_lambda(ep, kind=ramp_kind, e0=ramp_start_epoch, e1=ramp_end_epoch, k=sigmoid_k)

            model.train()
            seen = 0
            loss_sum = 0.0
            t0 = time.time()

            for i, batch in enumerate(train_loader, 1):
                # require (xb, yb, tid) for Idea2
                xb, yb, tid = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)

                    # baseline blended loss (unchanged schedule)
                    loss_base = blended_loss(core, aftl, w, logits, yb_r)

                    # Idea2 extra term: boost POS pixels in hard tiles
                    if (hard_mask_train is None) or (lam <= 0.0) or (hard_pos_boost <= 1.0):
                        loss = loss_base
                    else:
                        tid_np = tid.detach().cpu().numpy().astype(np.int64)
                        hard = torch.from_numpy(hard_mask_train[tid_np].astype(np.float32)).to(self.device)  # [B]
                        hard = hard.view(-1, 1, 1, 1)

                        pos = (yb_r > 0.5).to(logits.dtype)  # [B,1,H,W]
                        bce_map = F.binary_cross_entropy_with_logits(
                            logits, yb_r, pos_weight=posw_long, reduction="none"
                        )

                        # weight only positive pixels in hard tiles
                        W = 1.0 + (hard_pos_boost - 1.0) * hard * pos
                        extra = (bce_map * (W - 1.0)).mean()

                        loss = loss_base + lam * extra

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
            tr_stats = pix_eval(model, resize_masks_to, train_loader, thr=metric_thr,
                                max_batches=quick_eval_train_batches)

            if self.is_main_process() and verbose >= 2:
                print(
                    f"[EP{ep:02d}] lam={lam:.3f} loss {train_loss:.4f} | "
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

                val_stats = pix_eval(model, resize_masks_to, val_loader, thr=metric_thr,
                                     max_batches=quick_eval_val_batches)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default="../")
    ap.add_argument("--train-h5", type=str, default="/home/karlo/train.h5")
    ap.add_argument("--train-csv", type=str, default="../DATA/train.csv")
    ap.add_argument("--test-h5", type=str, default="../DATA/test.h5")

    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.1)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--pos-weight", type=float, default=8.0)

    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--long-batches", type=int, default=0)

    ap.add_argument("--margin-pix", type=float, default=8.0)
    ap.add_argument("--min-pos-pix", type=int, default=1)

    ap.add_argument("--hard-pos-boost", type=float, default=4.0)
    ap.add_argument("--ramp-kind", type=str, default="linear", choices=["linear", "sigmoid"])
    ap.add_argument("--ramp-start-epoch", type=int, default=1)
    ap.add_argument("--ramp-end-epoch", type=int, default=60)
    ap.add_argument("--sigmoid-k", type=float, default=8.0)

    ap.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    ap.add_argument("--tag", type=str, default="idea2")
    ap.add_argument("--verbose", type=int, default=3)
    return ap.parse_args()


def main():
    args = parse_args()
    add_repo_to_syspath(args.repo_root)

    (Config,
     H5TiledDataset,
     UNetResSEASPP,
     set_seed,
     split_indices,
     BCEIoUEdge,
     blended_loss,
     make_loss_for_epoch,
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

    cfg = Config()

    cfg.train.max_epochs = args.max_epochs
    cfg.train.val_every = args.val_every if args.val_every is not None else args.max_epochs

    # output paths
    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    cfg.tile = args.tile,
    cfg.batch_size = args.batch_size,
    cfg.num_workers = args.num_workers,
    cfg.pos_weight = args.pos_weight,
    cfg.margin_pix = args.margin_pix,
    cfg.min_pos_pix = args.min_pos_pix,
    cfg.hard_pos_boost = args.hard_pos_boost,
    cfg.ramp_kind = args.ramp_kind,
    cfg.ramp_start_epoch = args.ramp_start_epoch,
    cfg.ramp_end_epoch = args.ramp_end_epoch,
    cfg.sigmoid_k = args.sigmoid_k,
    cfg.save_best_to = best_path,
    cfg.save_last_to = last_path,

    set_seed(args.seed)

    base_ds = H5TiledDataset(args.train_h5, tile=args.tile, k_sigma=5.0)

    idx_tr, idx_va = split_indices(args.train_h5, val_frac=float(args.val_frac), seed=args.seed)
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))

    tiles_tr_base = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va_base = filter_tiles_by_panels(base_ds, idx_va_set)

    # hard tiles
    hard_mask_base = build_hard_mask_base_from_csv(args.train_csv, base_ds, margin_pix=args.margin_pix)
    hard_candidates_base = np.flatnonzero(hard_mask_base)

    tr_set = set(map(int, tiles_tr_base.tolist()))
    hard_candidates_base = np.asarray([i for i in hard_candidates_base if int(i) in tr_set], dtype=np.int64)
    hard_base_kept = filter_tiles_by_truth(args.train_h5, base_ds, hard_candidates_base, min_pos_pix=args.min_pos_pix)

    # datasets
    train_ds = TileSubsetWithId(base_ds, tiles_tr_base)  # returns (x,y,tid)
    val_ds = TileSubset(base_ds, tiles_va_base)

    # map hard base -> train tid
    base_to_train_tid = {int(b): int(tid) for tid, b in enumerate(train_ds.base_tile_indices)}
    hard_train_tid = np.asarray([base_to_train_tid[int(b)] for b in hard_base_kept if int(b) in base_to_train_tid],
                                dtype=np.int64)

    hard_mask_train = np.zeros(len(train_ds), dtype=bool)
    hard_mask_train[hard_train_tid] = True

    if is_main_process():
        print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")
        print(f"Hard tiles after truth filter: {len(hard_train_tid)} | hard_frac={hard_mask_train.mean():.6f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = UNetResSEASPP(in_ch=1, out_ch=1)

    trainer = Trainer(
        init_distributed=init_distributed,
        is_main_process=is_main_process,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_amp=True,
    )

    trainer.train_full_probe(
        model, train_loader=train_loader, val_loader=val_loader,
        seed=cfg.train.seed,
        init_head_prior=cfg.train.init_head_prior,
        warmup_epochs=cfg.train.warmup_epochs, warmup_batches=cfg.train.warmup_batches,
        warmup_lr=cfg.train.warmup_lr, warmup_pos_weight=cfg.train.warmup_pos_weight,
        head_epochs=cfg.train.head_epochs, head_batches=cfg.train.head_batches,
        head_lr=cfg.train.head_lr, head_pos_weight=cfg.train.head_pos_weight,
        tail_epochs=cfg.train.tail_epochs, tail_batches=cfg.train.tail_batches,
        tail_lr=cfg.train.tail_lr, tail_pos_weight=cfg.train.tail_pos_weight,
        max_epochs=cfg.train.max_epochs, val_every=cfg.train.val_every,
        base_lrs=cfg.train.base_lrs, weight_decay=cfg.train.weight_decay,
        thr_beta=cfg.train.thr_beta, long_batches=cfg.train.long_batches,
        thr_pos_rate_early=cfg.train.thr_pos_rate_early, thr_pos_rate_late=cfg.train.thr_pos_rate_late,
        save_best_to=best_path, save_last_to=last_path,
        verbose=args.verbose,

        # --- Idea2 additions (explicit kwargs) ---
        hard_mask_train=hard_mask_train,
        hard_pos_boost=args.hard_pos_boost,
        ramp_kind=args.ramp_kind,
        ramp_start_epoch=args.ramp_start_epoch,
        ramp_end_epoch=args.ramp_end_epoch,
        sigmoid_k=args.sigmoid_k,
        pos_weight_long=args.pos_weight,  # Use args.pos_weight instead of cfg.train.pos_weight

        # inject project fns (keeps this file independent of global imports)
        resize_masks_to=resize_masks_to,
        pick_thr_with_floor=pick_thr_with_floor,
        roc_auc_ddp=roc_auc_ddp,
        maybe_init_head_bias_to_prior=maybe_init_head_bias_to_prior,
        apply_phase=apply_phase,
        freeze_all=freeze_all,
        _unfreeze_if_exists=_unfreeze_if_exists,
        make_loss_for_epoch=make_loss_for_epoch,
        blended_loss=blended_loss,
        BCEIoUEdge=BCEIoUEdge,
        make_opt_sched=make_opt_sched,
    )


if __name__ == "__main__":
    main()
