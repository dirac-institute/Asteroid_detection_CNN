#!/usr/bin/env python3
"""
Idea 5 — Encourage elongated, coherent trail-like predictions.

Loss-based option (LONG phase only):
- Penalize isolated activations (neighbor support regularizer)
- Penalize compact blobs, encourage elongation (soft covariance eigen-ratio)
- Encourage thin structures (area / perimeter penalty)

Goal:
- Reuse the same training process as default Trainer.train_full_probe (warmup/head/tail/long)
- Change ONLY the loss in the LONG loop by adding shape priors on p = sigmoid(logits)
- Keep training budget comparable across Ideas

Standalone script (does NOT modify train.py).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
        UNetResSEASPP,
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
# Baseline losses
# -------------------------
def bce_with_logits_posw(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    posw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    t = targets.clamp(0, 1)
    return F.binary_cross_entropy_with_logits(logits, t, pos_weight=posw)


# -------------------------
# Idea5 regularizers
# -------------------------
def neighbor_support_penalty(p: torch.Tensor, margin: float = 0.20, eps: float = 1e-6) -> torch.Tensor:
    """
    Penalize "on" pixels that lack neighbor support.
    Uses 3x3 neighbor mean excluding center.
    p: (B,1,H,W) in [0,1]
    """
    B, C, H, W = p.shape
    # 3x3 kernel with zero center
    k = torch.ones((1, 1, 3, 3), device=p.device, dtype=p.dtype)
    k[0, 0, 1, 1] = 0.0
    neigh_sum = F.conv2d(p, k, padding=1)
    neigh_mean = neigh_sum / 8.0
    # penalize high p where neigh_mean is below margin
    pen = p * F.relu(margin - neigh_mean)
    return pen.mean()


def sobel_perimeter(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft perimeter proxy via Sobel gradient magnitude.
    """
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=p.device, dtype=p.dtype).view(1, 1, 3, 3)
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=p.device, dtype=p.dtype).view(1, 1, 3, 3)
    dx = F.conv2d(p, gx, padding=1)
    dy = F.conv2d(p, gy, padding=1)
    g = torch.sqrt(dx * dx + dy * dy + eps)
    return g.mean()


def elongation_ratio_penalty(p: torch.Tensor, eps: float = 1e-6, min_mass: float = 50.0) -> torch.Tensor:
    """
    Encourage elongation by penalizing eigenvalue ratio λ2/λ1 of the weighted covariance of (x,y).
    - blobs: ratio ~ 1 (bad)
    - trails: ratio << 1 (good)

    Computed per-sample, then averaged. If total mass is too small, returns 0 for that sample.
    """
    B, C, H, W = p.shape
    assert C == 1

    ys = torch.linspace(0.0, float(H - 1), H, device=p.device, dtype=p.dtype).view(1, 1, H, 1)
    xs = torch.linspace(0.0, float(W - 1), W, device=p.device, dtype=p.dtype).view(1, 1, 1, W)

    w = p
    mass = w.sum(dim=(2, 3)) + eps  # (B,1)
    valid = (mass >= float(min_mass)).float()

    mx = (w * xs).sum(dim=(2, 3)) / mass
    my = (w * ys).sum(dim=(2, 3)) / mass

    x0 = xs - mx.view(B, 1, 1, 1)
    y0 = ys - my.view(B, 1, 1, 1)

    cxx = (w * x0 * x0).sum(dim=(2, 3)) / mass
    cyy = (w * y0 * y0).sum(dim=(2, 3)) / mass
    cxy = (w * x0 * y0).sum(dim=(2, 3)) / mass

    # eigenvalues of 2x2 covariance:
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = torch.sqrt(torch.clamp(tr * tr - 4.0 * det, min=0.0) + eps)
    lam1 = 0.5 * (tr + disc)  # major
    lam2 = 0.5 * (tr - disc)  # minor

    ratio = (lam2 + eps) / (lam1 + eps)  # in (0,1]
    # apply only if valid mass
    return (valid * ratio).mean()


def thinness_penalty(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Penalize compactness using area/perimeter.
    - compact blob: high area, relatively low perimeter -> larger area/perimeter (bad)
    - thin trail: lower area, relatively higher perimeter -> smaller area/perimeter (good)
    """
    area = p.mean()
    per = sobel_perimeter(p, eps=eps)
    return area / (per + eps)


def trail_shape_regularizer(
    p: torch.Tensor,
    *,
    w_iso: float,
    w_ecc: float,
    w_thin: float,
    iso_margin: float,
    ecc_min_mass: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    iso = neighbor_support_penalty(p, margin=iso_margin)
    ecc = elongation_ratio_penalty(p, min_mass=ecc_min_mass)
    thin = thinness_penalty(p)

    reg = w_iso * iso + w_ecc * ecc + w_thin * thin
    aux = {"iso": float(iso.item()), "ecc": float(ecc.item()), "thin": float(thin.item())}
    return reg, aux


# -------------------------
# Eval helper (DDP-safe)
# -------------------------
@torch.no_grad()
def pix_eval(model, resize_masks_to, loader, thr: float, max_batches: int):
    model.eval()
    dev = next(model.parameters()).device

    tp = 0.0
    fp = 0.0
    fn = 0.0

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
    return {"P": P, "R": R, "F": F1}


# -------------------------
# Trainer (baseline process + Idea5 tweak in LONG only)
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
        # shape regularizer knobs (LONG only)
        shape_weight: float = 0.10,
        w_iso: float = 1.0,
        w_ecc: float = 1.0,
        w_thin: float = 0.5,
        iso_margin: float = 0.20,
        ecc_min_mass: float = 50.0,
        # apply shape reg only when GT has positives
        reg_only_if_pos: bool = True,
        min_pos_pix: int = 10,
        # baseline knobs (match Config defaults)
        seed=1337,
        init_head_prior=0.70,
        warmup_epochs=5,
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

        is_dist, rank, local_rank, world_size = self.init_distributed()
        raw_model = model
        if is_dist:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,  # because we freeze/unfreeze across phases
            )

        # imports needed inside loops
        from phases import freeze_all, _unfreeze_if_exists, make_opt_sched, apply_phase, maybe_init_head_bias_to_prior
        from thresholds import resize_masks_to, pick_thr_with_floor
        from metrics import roc_auc_ddp

        # ---------- Warmup ----------
        raw_model.train()
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=warmup_lr, weight_decay=1e-4)

        for ep in range(1, warmup_epochs + 1):
            self._set_loader_epoch(train_loader, 10_000 + ep)
            model.train()
            seen, loss_sum = 0, 0.0
            for b, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = bce_with_logits_posw(logits, yb_r, pos_weight=float(warmup_pos_weight))

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
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=head_lr, weight_decay=1e-4)

        for ep in range(1, head_epochs + 1):
            self._set_loader_epoch(train_loader, 20_000 + ep)
            model.train()
            seen, loss_sum = 0, 0.0
            for b, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = bce_with_logits_posw(logits, yb_r, pos_weight=float(head_pos_weight))

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

            # DDP-safe: all ranks compute, only rank0 prints
            stats = pix_eval(model, resize_masks_to, train_loader, thr=float(thr0), max_batches=quick_eval_train_batches)
            if self.is_main_process() and verbose >= 2:
                print(f"[HEAD] ep{ep} loss {loss_sum / max(seen, 1):.4f} | F1 {stats['F']:.3f}")

        # ---------- Tail probe ----------
        freeze_all(raw_model)
        if hasattr(raw_model, "head"):
            for p in raw_model.head.parameters():
                p.requires_grad = True
        for g in ["u4", "u3", "aspp"]:
            _unfreeze_if_exists(raw_model, g)

        opt = torch.optim.Adam([p for p in raw_model.parameters() if p.requires_grad], lr=tail_lr, weight_decay=1e-4)
        metric_thr = float(thr0)

        for ep in range(1, tail_epochs + 1):
            self._set_loader_epoch(train_loader, 30_000 + ep)
            model.train()
            seen, loss_sum = 0, 0.0
            for b, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    loss = bce_with_logits_posw(logits, yb_r, pos_weight=float(tail_pos_weight))

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

            pr_min, pr_max = thr_pos_rate_early
            metric_thr, _, aux = pick_thr_with_floor(
                model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                min_pos_rate=pr_min, max_pos_rate=pr_max
            )
            metric_thr = float(metric_thr)
            val_stats = pix_eval(model, resize_masks_to, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)

            if self.is_main_process() and verbose >= 2:
                print(f"[TAIL] ep{ep} loss {loss_sum / max(seen,1):.4f} | val F1 {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux['pos_rate']:.3f}")

        # ---------- Long training (Idea5: BCE + shape regularizer) ----------
        best = {"auc": -1.0, "state": None, "thr": float(metric_thr), "ep": 0, "F": 0.0}

        for ep in range(1, max_epochs + 1):
            self._set_loader_epoch(train_loader, 40_000 + ep)
            _ = apply_phase(raw_model, ep)

            opt, sched = make_opt_sched(raw_model, ep, tuple(float(x) for x in base_lrs), weight_decay)

            model.train()
            seen, loss_sum = 0, 0.0
            reg_sum = 0.0
            t0 = time.time()

            for i, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)

                    # base pixel loss
                    loss_bce = bce_with_logits_posw(logits, yb_r, pos_weight=float(head_pos_weight))

                    # shape regularizer on p
                    p = torch.sigmoid(logits)

                    # optionally apply reg only on tiles with positives
                    if reg_only_if_pos:
                        pos_counts = (yb_r > 0.5).sum(dim=(1, 2, 3))  # (B,)
                        keep = (pos_counts >= int(min_pos_pix)).float().view(-1, 1, 1, 1)
                        p_reg = p * keep
                    else:
                        p_reg = p

                    reg, aux = trail_shape_regularizer(
                        p_reg,
                        w_iso=float(w_iso),
                        w_ecc=float(w_ecc),
                        w_thin=float(w_thin),
                        iso_margin=float(iso_margin),
                        ecc_min_mass=float(ecc_min_mass),
                    )
                    loss = loss_bce + float(shape_weight) * reg

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step(i)

                loss_sum += float(loss.item()) * xb.size(0)
                reg_sum += float(reg.item()) * xb.size(0)
                seen += xb.size(0)

                if (long_batches > 0) and (i >= long_batches):
                    break

            train_loss = loss_sum / max(seen, 1)
            train_reg = reg_sum / max(seen, 1)

            tr_stats = pix_eval(model, resize_masks_to, train_loader, thr=float(metric_thr), max_batches=quick_eval_train_batches)
            if self.is_main_process() and verbose >= 2:
                print(
                    f"[EP{ep:02d}] loss {train_loss:.4f} reg {train_reg:.4f} | "
                    f"train F1 {tr_stats['F']:.3f} | {time.time() - t0:.1f}s"
                )

            do_val = (ep % int(val_every) == 0) or (ep <= 1) or (ep == max_epochs)
            if do_val:
                pr_min, pr_max = (thr_pos_rate_early if ep < 26 else thr_pos_rate_late)
                thr, _, aux_thr = pick_thr_with_floor(
                    model, val_loader, max_batches=120, n_bins=256, beta=thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                metric_thr = float(thr)
                val_stats = pix_eval(model, resize_masks_to, val_loader, thr=metric_thr, max_batches=quick_eval_val_batches)
                auc = roc_auc_ddp(model, val_loader, n_bins=256, max_batches=quick_eval_val_batches)

                if self.is_main_process():
                    print(f"[VAL ep{ep}] AUC {auc:.3f} F1 {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux_thr['pos_rate']:.3f}")

                if auc > best["auc"]:
                    best = {
                        "auc": float(auc),
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": float(metric_thr),
                        "ep": int(ep),
                        "F": float(val_stats["F"]),
                    }
                    if self.is_main_process() and save_best_to:
                        torch.save({"state": best["state"], "thr": best["thr"], "ep": best["ep"], "auc": best["auc"], "F": best["F"]}, save_best_to)

            if self.is_main_process() and save_last_to:
                torch.save({"state": best["state"], "thr": best["thr"], "ep": best["ep"], "auc": best["auc"], "F": best["F"]}, save_last_to)

        if best["state"] is not None:
            raw_model.load_state_dict(best["state"], strict=True)
        return raw_model, float(best["thr"]), best


# -------------------------
# CLI / main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Idea5: BCE + trail-shape regularizer (LONG only).")
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

    # Idea5 knobs
    ap.add_argument("--shape-weight", type=float, default=0.10)
    ap.add_argument("--w-iso", type=float, default=1.0)
    ap.add_argument("--w-ecc", type=float, default=1.0)
    ap.add_argument("--w-thin", type=float, default=0.5)
    ap.add_argument("--iso-margin", type=float, default=0.20)
    ap.add_argument("--ecc-min-mass", type=float, default=50.0)
    ap.add_argument("--reg-only-if-pos", action="store_true", default=True)
    ap.add_argument("--min-pos-pix", type=int, default=10)

    ap.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    ap.add_argument("--tag", type=str, default="idea5")
    ap.add_argument("--verbose", type=int, default=2)
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

    cfg = Config()
    set_seed(int(args.seed))

    # DDP init once
    is_dist, rank, local_rank, world_size = init_distributed()

    base_ds = H5TiledDataset(args.train_h5, tile=int(args.tile), k_sigma=5.0)

    idx_tr, idx_va = split_indices(args.train_h5, val_frac=float(args.val_frac), seed=int(args.seed))
    tiles_tr = filter_tiles_by_panels(base_ds, set(map(int, idx_tr.tolist())))
    tiles_va = filter_tiles_by_panels(base_ds, set(map(int, idx_va.tolist())))

    train_ds = TileSubset(base_ds, tiles_tr)
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

    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    trainer = Trainer(init_distributed=init_distributed, is_main_process=is_main_process, device=device, use_amp=True)

    try:
        model, thr, summary = trainer.train_full_probe(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            # Idea5 knobs
            shape_weight=float(args.shape_weight),
            w_iso=float(args.w_iso),
            w_ecc=float(args.w_ecc),
            w_thin=float(args.w_thin),
            iso_margin=float(args.iso_margin),
            ecc_min_mass=float(args.ecc_min_mass),
            reg_only_if_pos=bool(args.reg_only_if_pos),
            min_pos_pix=int(args.min_pos_pix),
            # baseline knobs from config
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
            print("Final threshold:", thr)
            print("Summary:", summary)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
