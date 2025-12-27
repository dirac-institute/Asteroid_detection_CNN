#!/usr/bin/env python3
"""
Idea 2 — Loss biased toward LSST-missed examples (tile-level hard label, pixel-level weighting on positives)

- Build hard tiles from train.csv (stack_detection==0) using conservative bbox -> touched tiles.
- Truth-filter hard tiles using Y from train.h5 (keep only tiles with >= min_pos_pix positives).
- Train like baseline, but during LONG training only:
    * compute per-pixel BCEWithLogits loss map (reduction='none')
    * boost only POSITIVE pixels in HARD tiles
    * optionally ramp the strength late in training (linear or sigmoid)

This is designed to run on SLURM (no Jupyter assumptions).
"""

from __future__ import annotations

import os
import sys
import math
import time
import copy
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn.functional as F
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# -------------------------
# Path setup (repo imports)
# -------------------------
def add_repo_to_syspath(repo_root: str):
    repo = Path(repo_root).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # ADCNN package root
    if str(repo / "ADCNN") not in sys.path:
        sys.path.insert(0, str(repo / "ADCNN"))


# -------------------------
# Project imports (assumes your repo structure)
# -------------------------
def import_project():
    # NOTE: thresholds.py uses "from utils.dist_utils import ..."
    # so repo_root must be on sys.path for "utils" to resolve.
    from ADCNN.data.h5tiles import H5TiledDataset
    from ADCNN.models.unet_res_se import UNetResSEASPP
    from utils.dist_utils import init_distributed, is_main_process
    from thresholds import resize_masks_to, pick_thr_with_floor
    from metrics import roc_auc_ddp
    from ADCNN.utils.utils import set_seed, split_indices
    return H5TiledDataset, UNetResSEASPP, init_distributed, is_main_process, resize_masks_to, pick_thr_with_floor, roc_auc_ddp, set_seed, split_indices


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

    # optimization
    amp: bool = True
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # imbalance (baseline)
    pos_weight: float = 8.0

    # training schedule
    warmup_epochs: int = 1
    warmup_batches: int = 800
    head_epochs: int = 0
    head_batches: int = 0
    tail_epochs: int = 0
    tail_batches: int = 0
    max_epochs: int = 30
    val_every: int = 3

    # speed: limit long training batches (0 = full epoch)
    long_batches: int = 0

    # hard-tile construction from CSV
    margin_pix: float = 8.0
    min_pos_pix: int = 1  # truth filter threshold
    hard_frac_floor: float = 0.0  # keep for diagnostics

    # Idea 2 weighting
    hard_pos_boost: float = 4.0          # multiply POS pixels in hard tiles by this (>=1)
    apply_only_in_long: bool = True
    ramp_kind: str = "linear"            # "linear" or "sigmoid"
    ramp_start_epoch: int = 1            # relative to LONG training epoch counter (1..max_epochs)
    ramp_end_epoch: int = 30             # relative to LONG training epoch counter
    sigmoid_k: float = 8.0               # steepness for sigmoid ramp

    # threshold selection
    thr_beta: float = 2.0
    thr_pos_rate_early: Tuple[float, float] = (0.03, 0.10)
    thr_pos_rate_late: Tuple[float, float] = (0.08, 0.12)

    # checkpoints
    save_best_to: str = "ckpt_best_idea2.pt"
    save_last_to: str = "ckpt_last_idea2.pt"

    # quick eval
    quick_eval_train_batches: int = 6
    quick_eval_val_batches: int = 12


# -------------------------
# Utilities: hard tiles from CSV + truth filter
# -------------------------
def tiles_touched_by_bbox(x: float, y: float, R: float, H: int, W: int, tile: int) -> Tuple[int, int, int, int]:
    """Return inclusive tile r/c range [r0,r1], [c0,c1] touched by bbox."""
    x0 = max(0.0, x - R)
    x1 = min(float(W - 1), x + R)
    y0 = max(0.0, y - R)
    y1 = min(float(H - 1), y + R)

    c0 = int(math.floor(x0 / tile))
    c1 = int(math.floor(x1 / tile))
    r0 = int(math.floor(y0 / tile))
    r1 = int(math.floor(y1 / tile))
    return r0, r1, c0, c1


def build_hard_mask_base_from_csv(
    train_h5: str,
    train_csv: str,
    base_ds,  # H5TiledDataset
    *,
    margin_pix: float,
) -> np.ndarray:
    """
    Build boolean hard mask in BASE tile index space (len = len(base_ds)).
    A tile is hard if it is touched by any LSST-missed injection bbox (stack_detection==0).
    """
    tile = int(base_ds.tile)
    N, H, W = base_ds.N, base_ds.H, base_ds.W
    Hb = math.ceil(H / tile)
    Wb = math.ceil(W / tile)

    df = pd.read_csv(train_csv)
    # Ensure expected columns exist
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

    # Map (pid,r,c) -> base tile index
    base_map = {(i, r, c): k for k, (i, r, c) in enumerate(base_ds.indices)}
    hard_mask = np.zeros(len(base_ds), dtype=bool)
    for pid, rcset in hard_rc_by_panel.items():
        for (r, c) in rcset:
            k = base_map.get((pid, r, c), None)
            if k is not None:
                hard_mask[k] = True
    return hard_mask


def filter_tiles_by_truth(
    train_h5: str,
    tiled_ds,              # H5TiledDataset (base)
    base_tile_indices: np.ndarray,
    *,
    min_pos_pix: int = 1,
) -> np.ndarray:
    """
    Keep only those BASE tile indices whose truth mask contains at least min_pos_pix positive pixels.
    """
    base_tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
    kept = []

    with h5py.File(train_h5, "r") as f:
        Y = f["masks"]
        H, W = Y.shape[1:]
        t = int(tiled_ds.tile)

        for idx in base_tile_indices:
            i, r, c = tiled_ds.indices[int(idx)]
            r0, c0 = r * t, c * t
            r1, c1 = min(r0 + t, H), min(c0 + t, W)
            y = Y[i, r0:r1, c0:c1]
            if int(np.count_nonzero(y)) >= int(min_pos_pix):
                kept.append(int(idx))

    return np.asarray(kept, dtype=np.int64)


def filter_tiles_by_panels(base_ds, panel_id_set: set[int]) -> np.ndarray:
    kept = []
    for k, (pid, r, c) in enumerate(base_ds.indices):
        if pid in panel_id_set:
            kept.append(k)
    return np.asarray(kept, dtype=np.int64)


class TileSubset(Dataset):
    """
    Subset of base tiled dataset by BASE tile indices.
    Returns (x, y) by default.
    """
    def __init__(self, base, base_tile_indices: np.ndarray):
        self.base = base
        self.tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
    def __len__(self): return len(self.tile_indices)
    def __getitem__(self, i):
        return self.base[int(self.tile_indices[int(i)])]


class TileSubsetWithId(Dataset):
    """
    Same as TileSubset but returns (x, y, tid) where tid is TRAIN-DS index (0..len-1),
    so we can look up per-tile hard labels in train-index space.
    """
    def __init__(self, base, base_tile_indices: np.ndarray):
        self.base = base
        self.tile_indices = np.asarray(base_tile_indices, dtype=np.int64)
    def __len__(self): return len(self.tile_indices)
    def __getitem__(self, i):
        x, y = self.base[int(self.tile_indices[int(i)])]
        return x, y, int(i)


# -------------------------
# Metrics / eval
# -------------------------
@torch.no_grad()
def pix_eval(resize_masks_to, model, loader, thr=0.2, max_batches=12):
    model.eval()
    dev = next(model.parameters()).device
    tp = 0.0
    fp = 0.0
    fn = 0.0
    posm = []
    negm = []

    for bi, batch in enumerate(loader, 1):
        if len(batch) == 3:
            xb, yb, _tid = batch
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
# Ramp for weighting
# -------------------------
def ramp_lambda(ep_long: int, *, kind: str, e_start: int, e_end: int, k: float) -> float:
    """
    ep_long is 1..max_epochs (LONG training epoch index).
    """
    if e_end <= e_start:
        return 1.0 if ep_long >= e_start else 0.0

    if kind == "linear":
        x = (ep_long - e_start) / float(e_end - e_start)
        return float(np.clip(x, 0.0, 1.0))

    if kind == "sigmoid":
        # map e_start..e_end to -1..+1
        x = (ep_long - e_start) / float(e_end - e_start)
        x = np.clip(x, 0.0, 1.0)
        z = (x * 2.0 - 1.0) * float(k)
        return float(1.0 / (1.0 + np.exp(-z)))

    raise ValueError(f"Unknown ramp kind: {kind}")


# -------------------------
# Trainer (baseline + Idea2 loss weighting in long training)
# -------------------------
class Trainer:
    def __init__(self, init_distributed, is_main_process, resize_masks_to, pick_thr_with_floor, roc_auc_ddp, *, device=None, use_amp=True):
        self.init_distributed = init_distributed
        self.is_main_process = is_main_process
        self.resize_masks_to = resize_masks_to
        self.pick_thr_with_floor = pick_thr_with_floor
        self.roc_auc_ddp = roc_auc_ddp
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(use_amp)

    def _set_loader_epoch(self, loader, epoch: int):
        if hasattr(loader, "sampler") and isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
            return
        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
            return

    def train(self, model, train_loader, val_loader, *,
              hard_mask_train: Optional[np.ndarray],
              cfg: Cfg):

        is_dist, rank, local_rank, world_size = self.init_distributed()
        scaler = amp.GradScaler("cuda", enabled=self.amp)

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        model = model.to(self.device)
        if is_dist:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True, gradient_as_bucket_view=True)

        raw_model = model.module if isinstance(model, DDP) else model

        # -------------
        # Warmup (optional)
        # -------------
        posw = torch.tensor(cfg.pos_weight, device=self.device)
        opt = torch.optim.Adam(raw_model.parameters(), lr=cfg.lr, weight_decay=0.0)

        for ep in range(1, cfg.warmup_epochs + 1):
            self._set_loader_epoch(train_loader, ep)
            model.train()
            seen = 0
            loss_sum = 0.0

            for b, batch in enumerate(train_loader, 1):
                if len(batch) == 3:
                    xb, yb, _tid = batch
                else:
                    xb, yb = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = self.resize_masks_to(logits, yb)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_r, pos_weight=posw)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                if b >= cfg.warmup_batches:
                    break

            stats = pix_eval(self.resize_masks_to, model, train_loader, thr=0.2, max_batches=cfg.quick_eval_train_batches)
            if self.is_main_process():
                print(f"[WARMUP] ep{ep} loss {loss_sum/seen:.4f} | F1 {stats['F']:.3f} P {stats['P']:.3f} R {stats['R']:.3f}")

        # initial threshold on val
        thr0, *_ = self.pick_thr_with_floor(
            model, val_loader, max_batches=200, n_bins=256,
            beta=cfg.thr_beta,
            min_pos_rate=cfg.thr_pos_rate_early[0],
            max_pos_rate=cfg.thr_pos_rate_early[1],
        )
        val_stats = pix_eval(self.resize_masks_to, model, val_loader, thr=float(thr0), max_batches=cfg.quick_eval_val_batches)
        auc = self.roc_auc_ddp(model, val_loader, n_bins=256, max_batches=cfg.quick_eval_val_batches)
        if self.is_main_process():
            print(f"[WARMUP VALIDATION] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} F {val_stats['F']:.3f} | thr={float(thr0):.3f}")

        # -------------
        # Long training (Idea 2 applies here)
        # -------------
        best = {"auc": -1.0, "state": None, "thr": float(thr0), "ep": 0, "P": 0.0, "R": 0.0, "F": 0.0}
        metric_thr = float(thr0)

        # weights vector for train tiles (train-index space)
        if hard_mask_train is not None:
            hard_mask_train = np.asarray(hard_mask_train, dtype=bool)
            assert hard_mask_train.ndim == 1

        for ep_long in range(1, cfg.max_epochs + 1):
            # epoch seed for samplers
            self._set_loader_epoch(train_loader, 10_000 + ep_long)

            # ramp strength
            lam = ramp_lambda(
                ep_long,
                kind=cfg.ramp_kind,
                e_start=cfg.ramp_start_epoch,
                e_end=cfg.ramp_end_epoch,
                k=cfg.sigmoid_k,
            ) if cfg.apply_only_in_long else 1.0

            opt = torch.optim.Adam(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

            model.train()
            seen = 0
            loss_sum = 0.0
            t0 = time.time()

            for bi, batch in enumerate(train_loader, 1):
                xb, yb, tid = batch  # MUST be TileSubsetWithId
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = self.resize_masks_to(logits, yb)

                    # base BCE map
                    loss_map = F.binary_cross_entropy_with_logits(
                        logits, yb_r, pos_weight=posw, reduction="none"
                    )  # [B,1,H,W]

                    if hard_mask_train is None or lam <= 0.0 or cfg.hard_pos_boost <= 1.0:
                        loss = loss_map.mean()
                    else:
                        # build per-sample hard flag from tid (train-index space)
                        tid_np = tid.detach().cpu().numpy().astype(np.int64)
                        hard_flags = torch.from_numpy(hard_mask_train[tid_np].astype(np.float32)).to(self.device)  # [B]
                        hard_flags = hard_flags.view(-1, 1, 1, 1)  # [B,1,1,1]

                        # weight only POSITIVE pixels in hard tiles
                        pos = (yb_r > 0.5).to(loss_map.dtype)  # [B,1,H,W]
                        boost = 1.0 + (cfg.hard_pos_boost - 1.0) * hard_flags  # [B,1,1,1]
                        W = 1.0 + (boost - 1.0) * pos  # background stays 1

                        weighted = (loss_map * W)
                        # normalized weighted mean keeps scale stable
                        denom = W.mean().clamp_min(1e-6)
                        loss_weighted = weighted.mean() / denom

                        # blend late weighting in gradually
                        loss = (1.0 - lam) * loss_map.mean() + lam * loss_weighted

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                loss_sum += float(loss.item()) * xb.size(0)
                seen += xb.size(0)

                # optional speed cap
                if cfg.long_batches and bi >= int(cfg.long_batches):
                    break

            train_loss = loss_sum / max(seen, 1)
            tr_stats = pix_eval(self.resize_masks_to, model, train_loader, thr=metric_thr, max_batches=cfg.quick_eval_train_batches)
            if self.is_main_process():
                print(
                    f"[LONG ep{ep_long:02d}] lam={lam:.3f} loss {train_loss:.4f} | "
                    f"train P {tr_stats['P']:.3f} R {tr_stats['R']:.3f} F {tr_stats['F']:.3f} "
                    f"| pos≈{tr_stats['pos_mean']:.3f} neg≈{tr_stats['neg_mean']:.3f} | {time.time()-t0:.1f}s"
                )

            if (ep_long % cfg.val_every == 0) or (ep_long <= 3):
                self._set_loader_epoch(val_loader, 20_000 + ep_long)

                pr_min, pr_max = (cfg.thr_pos_rate_early if ep_long < 26 else cfg.thr_pos_rate_late)
                thr, (VP, VR, VF), aux = self.pick_thr_with_floor(
                    model, val_loader, max_batches=120, n_bins=256, beta=cfg.thr_beta,
                    min_pos_rate=pr_min, max_pos_rate=pr_max
                )
                metric_thr = float(thr)
                val_stats = pix_eval(self.resize_masks_to, model, val_loader, thr=metric_thr, max_batches=cfg.quick_eval_val_batches)
                auc = self.roc_auc_ddp(model, val_loader, n_bins=256, max_batches=cfg.quick_eval_val_batches)

                if self.is_main_process():
                    print(
                        f"[VAL ep{ep_long}] AUC {auc:.3f} P {val_stats['P']:.3f} R {val_stats['R']:.3f} "
                        f"F {val_stats['F']:.3f} | thr={metric_thr:.3f} | pos_rate≈{aux.get('pos_rate', np.nan):.3f}"
                    )

                if auc > best["auc"]:
                    best = {
                        "auc": float(auc),
                        "state": copy.deepcopy(raw_model.state_dict()),
                        "thr": float(metric_thr),
                        "ep": int(ep_long),
                        "P": float(val_stats["P"]),
                        "R": float(val_stats["R"]),
                        "F": float(val_stats["F"]),
                    }
                    if self.is_main_process() and cfg.save_best_to:
                        torch.save(best, cfg.save_best_to)

            if self.is_main_process() and cfg.save_last_to:
                torch.save(best, cfg.save_last_to)

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

        trained_model = raw_model
        return trained_model, best["thr"], summary


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default="../")
    ap.add_argument("--train-h5", type=str, default="/home/karlo/train.h5")
    ap.add_argument("--train-csv", type=str, default="../DATA/train.csv")
    ap.add_argument("--test-h5",  type=str, default="../DATA/test.h5")

    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--warmup-batches", type=int, default=800)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--long-batches", type=int, default=0)

    ap.add_argument("--margin-pix", type=float, default=8.0)
    ap.add_argument("--min-pos-pix", type=int, default=1)

    ap.add_argument("--hard-pos-boost", type=float, default=4.0)
    ap.add_argument("--ramp-kind", type=str, default="linear", choices=["linear", "sigmoid"])
    ap.add_argument("--ramp-start-epoch", type=int, default=1)
    ap.add_argument("--ramp-end-epoch", type=int, default=30)
    ap.add_argument("--sigmoid-k", type=float, default=8.0)

    ap.add_argument("--save-best-to", type=str, default="ckpt_best_idea2.pt")
    ap.add_argument("--save-last-to", type=str, default="ckpt_last_idea2.pt")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = Cfg(
        repo_root=args.repo_root,
        train_h5=args.train_h5,
        train_csv=args.train_csv,
        test_h5=args.test_h5,
        tile=args.tile,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        warmup_batches=args.warmup_batches,
        max_epochs=args.max_epochs,
        val_every=args.val_every,
        long_batches=args.long_batches,
        margin_pix=args.margin_pix,
        min_pos_pix=args.min_pos_pix,
        hard_pos_boost=args.hard_pos_boost,
        ramp_kind=args.ramp_kind,
        ramp_start_epoch=args.ramp_start_epoch,
        ramp_end_epoch=args.ramp_end_epoch,
        sigmoid_k=args.sigmoid_k,
        save_best_to=args.save_best_to,
        save_last_to=args.save_last_to,
    )

    add_repo_to_syspath(cfg.repo_root)
    (H5TiledDataset, UNetResSEASPP,
     init_distributed, is_main_process,
     resize_masks_to, pick_thr_with_floor, roc_auc_ddp,
     set_seed, split_indices) = import_project()

    set_seed(cfg.seed)

    # Base tiled dataset over ALL panels in train_h5
    base_ds = H5TiledDataset(cfg.train_h5, tile=cfg.tile, k_sigma=5.0)

    # Panel split (train/val by panel id)
    idx_tr, idx_va = split_indices(cfg.train_h5, val_frac=0.1, seed=cfg.seed)
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))

    tiles_tr_base = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va_base = filter_tiles_by_panels(base_ds, idx_va_set)

    # Build base hard mask from CSV (missed injections)
    hard_mask_base = build_hard_mask_base_from_csv(
        cfg.train_h5, cfg.train_csv, base_ds,
        margin_pix=cfg.margin_pix
    )

    # Restrict hard candidates to training split (base space)
    tr_set = set(map(int, tiles_tr_base.tolist()))
    hard_candidates_base = np.flatnonzero(hard_mask_base)
    hard_candidates_base = np.array([i for i in hard_candidates_base if int(i) in tr_set], dtype=np.int64)

    # Truth-filter in BASE space
    hard_base_kept = filter_tiles_by_truth(cfg.train_h5, base_ds, hard_candidates_base, min_pos_pix=cfg.min_pos_pix)

    # Train dataset: return tile id in TRAIN-DS space
    train_ds = TileSubsetWithId(base_ds, tiles_tr_base)
    val_ds = TileSubset(base_ds, tiles_va_base)

    # Map hard base indices -> train index space (0..len(train_ds)-1)
    base_to_train = {int(b): int(ti) for ti, b in enumerate(train_ds.tile_indices)}
    hard_train_idx = np.array([base_to_train[int(b)] for b in hard_base_kept if int(b) in base_to_train], dtype=np.int64)

    # Hard mask aligned to train_ds indices
    hard_mask_train = np.zeros(len(train_ds), dtype=bool)
    hard_mask_train[hard_train_idx] = True

    if is_main_process():
        print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")
        print(f"Hard tiles after truth filter: {len(hard_train_idx)} | Other tiles: {len(train_ds)-len(hard_train_idx)}")
        print(f"Hard fraction (train): {hard_mask_train.mean():.6f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = UNetResSEASPP(in_ch=1, out_ch=1)
    trainer = Trainer(
        init_distributed, is_main_process,
        resize_masks_to, pick_thr_with_floor, roc_auc_ddp,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_amp=cfg.amp,
    )

    trainer.train(model, train_loader, val_loader, hard_mask_train=hard_mask_train, cfg=cfg)


if __name__ == "__main__":
    main()
