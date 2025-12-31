#!/usr/bin/env python3
"""
Idea 3 — Two-phase loss schedule (BCE → Focal Tversky)

Phase 1: train with BCE (optionally BCE+Dice) for stability.
Phase 2: gradually blend in Focal Tversky:
    L = (1 - λ) * L_BCE + λ * L_FocalTversky
with λ ramped from 0→1 over selected epochs. Optionally reduce LR at start of Phase 2.

Standalone script (does NOT modify ADCNN/train.py), following the same principle/style
as idea1.py / idea2.py (custom Trainer with minimal deltas). :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
# Path setup (repo imports)
# -------------------------

def init_distributed_once(init_fn):
    """
    Wrap ADCNN.utils.dist_utils.init_distributed() so it's safe to call multiple times.
    Returns a function with same signature.
    """
    def _wrapped():
        # If already initialized, just report current state.
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            # local_rank from env (torchrun)
            local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
            return True, rank, local_rank, world_size
        return init_fn()
    return _wrapped


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
    from phases import (
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
    )
    from thresholds import pick_thr_with_floor, resize_masks_to
    from metrics import roc_auc_ddp
    from utils.dist_utils import init_distributed, is_main_process

    return (
        Config,
        H5TiledDataset,
        UNetResSEASPP,
        set_seed,
        split_indices,
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
        pick_thr_with_floor,
        resize_masks_to,
        roc_auc_ddp,
        init_distributed,
        is_main_process,
    )


# -------------------------
# Small dataset helpers (same idea as idea1/idea2)
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
# Losses for Idea3
# -------------------------
class BCELoss(nn.Module):
    """BCEWithLogits(pos_weight) over full map."""
    def __init__(self, pos_weight: float = 8.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.clamp(0, 1)
        posw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, t, pos_weight=posw)


class SoftDiceLoss(nn.Module):
    """Soft Dice on sigmoid(logits)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        t = targets.clamp(0, 1)
        dims = tuple(range(1, p.ndim))
        inter = (p * t).sum(dims)
        denom = (p + t).sum(dims) + self.eps
        dice = (2.0 * inter + self.eps) / denom
        return (1.0 - dice).mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky:
      TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
      L  = (1 - TI)^gamma
    """
    def __init__(self, alpha: float = 0.45, beta: float = 0.55, gamma: float = 1.3, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        t = targets.clamp(0, 1)

        # flatten per sample
        p = p.view(p.size(0), -1)
        t = t.view(t.size(0), -1)

        TP = (p * t).sum(1)
        FP = ((1.0 - t) * p).sum(1)
        FN = (t * (1.0 - p)).sum(1)

        ti = (TP + self.eps) / (TP + self.alpha * FP + self.beta * FN + self.eps)
        loss = torch.pow(1.0 - ti, self.gamma)
        return loss.mean()


def ramp_lambda(ep: int, *, kind: str, e0: int, e1: int, k: float) -> float:
    """
    Same ramp helper pattern as idea2. :contentReference[oaicite:1]{index=1}
    """
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
# Eval helpers (same as idea2 pattern)
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

    for bi, (xb, yb) in enumerate(loader, 1):
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
# Trainer (baseline process + Idea3 loss in LONG loop)
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

    def train_full_probe(
        self,
        model,
        train_loader,
        val_loader,
        *,
        # Idea3 knobs
        phase1_epochs: int = 10,
        ramp_kind: str = "linear",
        ramp_start_epoch: int = 11,
        ramp_end_epoch: int = 40,
        sigmoid_k: float = 8.0,
        bce_pos_weight_long: float = 8.0,
        bce_dice_mix: float = 0.0,  # 0 -> pure BCE; e.g. 0.2 -> 0.8*BCE + 0.2*Dice
        ft_alpha: float = 0.45,
        ft_beta: float = 0.55,
        ft_gamma: float = 1.3,
        phase2_lr_mult: float = 1.0,  # e.g. 0.5 to reduce LR once FT blending begins

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
        thr_pos_rate_late=(0.01, 0.05),
        save_best_to=None,
        save_last_to=None,
        verbose: int = 2,
        quick_eval_train_batches: int = 25,
        quick_eval_val_batches: int = 60,
        thr0: float = 0.5,
    ):
        torch.manual_seed(seed)
        scaler = amp.GradScaler(enabled=self.amp)

        # DDP wrap
        is_dist, rank, local_rank, world_size = self.init_distributed()
        raw_model = model
        if is_dist:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True,
                        gradient_as_bucket_view=True)

        # ---------- Warmup ----------
        raw_model.train()
        from phases import freeze_all, maybe_init_head_bias_to_prior  # local import for clarity
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True

        maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        core_warm = BCELoss(pos_weight=float(warmup_pos_weight)).to(self.device)
        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=warmup_lr, weight_decay=1e-4)

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, 10_000 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            for b, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    # no resize helper in warmup; keep consistent with later:
                    from thresholds import resize_masks_to
                    yb_r = resize_masks_to(logits, yb)
                    loss = core_warm(logits, yb_r)

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
                print(f"[WARMUP] ep{ep} loss {loss_sum / max(seen, 1):.4f}")

        # ---------- Head ----------
        from phases import _unfreeze_if_exists
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        core_head = BCELoss(pos_weight=float(head_pos_weight)).to(self.device)
        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=1e-4)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, 20_000 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            for b, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    from thresholds import resize_masks_to
                    yb_r = resize_masks_to(logits, yb)
                    loss = core_head(logits, yb_r)

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
                stats = pix_eval(model, resize_masks_to, train_loader, thr=float(thr0), max_batches=quick_eval_train_batches)
                print(f"[HEAD] ep{ep} loss {loss_sum / max(seen, 1):.4f} | F1 {stats['F']:.3f}")

        # ---------- Tail probe ----------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        core_tail = BCELoss(pos_weight=float(tail_pos_weight)).to(self.device)
        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)

        from thresholds import resize_masks_to, pick_thr_with_floor
        metric_thr = float(thr0)

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, 30_000 + ep)
            model.train()
            seen = 0
            loss_sum = 0.0
            for b, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = core_tail(logits, yb_r)

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
                print(f"[TAIL] ep{ep} loss {loss_sum / max(seen,1):.4f} | val F1 {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux['pos_rate']:.3f}")

        # ---------- Long training (Idea3 loss schedule) ----------
        bce = BCELoss(pos_weight=float(bce_pos_weight_long)).to(self.device)
        dice = SoftDiceLoss().to(self.device) if bce_dice_mix > 0 else None
        ft = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma).to(self.device)

        best = {"auc": -1.0, "state": None, "thr": float(metric_thr), "ep": 0, "P": 0.0, "R": 0.0, "F": 0.0}

        from phases import apply_phase, make_opt_sched
        from metrics import roc_auc_ddp

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, 40_000 + ep)

            # phase-wise unfreezing/lr schedule still comes from phases.py :contentReference[oaicite:2]{index=2}
            _ = apply_phase(raw_model, ep)

            # optional LR reduction starting when we begin blending in FT
            lrs = tuple(float(x) for x in base_lrs)
            if ep >= int(ramp_start_epoch) and float(phase2_lr_mult) != 1.0:
                lrs = tuple(float(phase2_lr_mult) * x for x in lrs)

            opt, sched = make_opt_sched(raw_model, ep, lrs, weight_decay)

            # λ=0 for Phase1; ramp λ during Phase2
            if ep <= int(phase1_epochs):
                lam = 0.0
            else:
                lam = ramp_lambda(ep, kind=ramp_kind, e0=int(ramp_start_epoch), e1=int(ramp_end_epoch), k=float(sigmoid_k))

            model.train()
            seen = 0
            loss_sum = 0.0
            t0 = time.time()

            for i, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)

                    loss_bce = bce(logits, yb_r)
                    if dice is not None:
                        loss_bce = (1.0 - float(bce_dice_mix)) * loss_bce + float(bce_dice_mix) * dice(logits, yb_r)

                    if lam <= 0.0:
                        loss = loss_bce
                    else:
                        loss_ft = ft(logits, yb_r)
                        loss = (1.0 - lam) * loss_bce + lam * loss_ft

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
            tr_stats = pix_eval(model, resize_masks_to, train_loader, thr=float(metric_thr), max_batches=quick_eval_train_batches)

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
                    model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                metric_thr = float(thr)

                val_stats = pix_eval(model, resize_masks_to, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)

                if self.is_main_process():
                    print(f"[VAL ep{ep}] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux['pos_rate']:.3f}")

                if auc > best["auc"]:
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
                        torch.save({"state": best["state"], "thr": best["thr"], "ep": best["ep"],
                                    "P": best["P"], "R": best["R"], "F": best["F"], "auc": best["auc"]},
                                   save_best_to)

            if self.is_main_process() and save_last_to:
                torch.save({"state": best["state"], "thr": best["thr"], "ep": best["ep"],
                            "P": best["P"], "R": best["R"], "F": best["F"], "auc": best["auc"]},
                           save_last_to)

        # load best back
        if best["state"] is not None:
            raw_model.load_state_dict(best["state"], strict=True)

        return raw_model, float(best["thr"]), best


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Idea3: BCE -> (blended) Focal Tversky loss schedule.")
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

    # Idea3 knobs
    ap.add_argument("--phase1-epochs", type=int, default=10)
    ap.add_argument("--bce-pos-weight-long", type=float, default=8.0)
    ap.add_argument("--bce-dice-mix", type=float, default=0.0)  # 0..1

    ap.add_argument("--ft-alpha", type=float, default=0.45)
    ap.add_argument("--ft-beta", type=float, default=0.55)
    ap.add_argument("--ft-gamma", type=float, default=1.3)

    ap.add_argument("--ramp-kind", type=str, default="linear", choices=["linear", "sigmoid"])
    ap.add_argument("--ramp-start-epoch", type=int, default=11)
    ap.add_argument("--ramp-end-epoch", type=int, default=40)
    ap.add_argument("--sigmoid-k", type=float, default=8.0)

    ap.add_argument("--phase2-lr-mult", type=float, default=1.0, help="e.g. 0.5 to halve LR when FT starts blending")

    ap.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    ap.add_argument("--tag", type=str, default="idea3")
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
        _unfreeze_if_exists,
        apply_phase,
        freeze_all,
        make_opt_sched,
        maybe_init_head_bias_to_prior,
        pick_thr_with_floor,
        resize_masks_to,
        roc_auc_ddp,
        init_distributed,
        is_main_process,
    ) = import_project()

    # wrap init_distributed so it won't init twice
    init_distributed = init_distributed_once(init_distributed)

    # initialize once here (so samplers can be created)
    is_dist, rank, local_rank, world_size = init_distributed()

    cfg = Config()
    cfg.train.max_epochs = int(args.max_epochs)
    cfg.train.val_every = int(args.val_every) if args.val_every is not None else int(args.max_epochs)

    set_seed(int(args.seed))

    # base dataset
    base_ds = H5TiledDataset(args.train_h5, tile=int(args.tile), k_sigma=5.0)

    # panel split -> tile subset
    idx_tr, idx_va = split_indices(args.train_h5, val_frac=float(args.val_frac), seed=int(args.seed))
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))

    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    train_ds = TileSubset(base_ds, tiles_tr)
    val_ds = TileSubset(base_ds, tiles_va)

    # DDP-aware loaders
    is_dist, rank, local_rank, world_size = init_distributed()
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

    # output paths
    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    trainer = Trainer(init_distributed=init_distributed, is_main_process=is_main_process, device=device, use_amp=True)

    model, thr, summary = trainer.train_full_probe(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        # Idea3 knobs
        phase1_epochs=int(args.phase1_epochs),
        ramp_kind=str(args.ramp_kind),
        ramp_start_epoch=int(args.ramp_start_epoch),
        ramp_end_epoch=int(args.ramp_end_epoch),
        sigmoid_k=float(args.sigmoid_k),
        bce_pos_weight_long=float(args.bce_pos_weight_long),
        bce_dice_mix=float(args.bce_dice_mix),
        ft_alpha=float(args.ft_alpha),
        ft_beta=float(args.ft_beta),
        ft_gamma=float(args.ft_gamma),
        phase2_lr_mult=float(args.phase2_lr_mult),
        # keep baseline-style training budget (from Config defaults)
        seed=int(args.seed),
        init_head_prior=cfg.train.init_head_prior,
        warmup_epochs=1,#cfg.train.warmup_epochs,
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
        print("Final threshold:", thr)
        print("Summary:", summary)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
