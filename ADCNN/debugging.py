#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone debug suite for ADCNN training pipeline (single-GPU recommended).

Goals:
- Reuse your existing dataset + model + loss utilities as much as possible
- FORCE EMA OFF (raw model weights only)
- Measure pixel ROC AUC on VALID pixels only (valid = ~real, where "real" marks pixels to ignore)
- Deterministic indexing + deterministic subset selection
- Mask polarity probe (valid=~real vs valid=real) to detect inverted masks
- Single-batch overfit test (decisive: if this fails, pipeline is broken)
- Extra: check whether y and real are accidentally identical (common reason for ~50% AUC)

IMPORTANT:
- Because we import helpers from repo-root main.py, run from repo root as a module:

    cd /home/karlo/Projects/Asteroid_detection_CNN
    python -m ADCNN.debugging --train-h5 ... --train-csv ... --mode smoke

Example runs:
  Smoke (no training, just dataset + random model probes):
    python -m ADCNN.debugging --train-h5 /path/train.h5 --train-csv /path/train.csv --mode smoke

  Single-batch overfit (most important):
    python -m ADCNN.debugging --train-h5 /path/train.h5 --train-csv /path/train.csv --mode overfit1 --steps 1500

Notes:
- This script does NOT require YAML config. It uses ADCNN.core.config.Config defaults and CLI overrides.
"""

from __future__ import annotations

import argparse
import builtins
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from ADCNN.core.config import Config
from ADCNN.core.model import UNetResSEASPP
from ADCNN.train import Trainer
from ADCNN.utils.dist_utils import init_distributed, is_main_process
from ADCNN.utils.helpers import set_seed, split_indices, worker_init_fn
from ADCNN.data.datasets import H5TiledDataset

# In your refactor, these live in repo-root main.py (not in ADCNN.data.datasets)
from ADCNN.main import TileSubsetWithRealAndFlags, filter_tiles_by_panels, build_tile_mask_from_csv

from ADCNN.evaluation.metrics import (
    resize_masks_to,
    valid_mask_from_real,
)

def masked_bce_compat(masked_bce_with_logits_fn, logits, y, valid, pos_w: float):
    """
    Call ADCNN.train.masked_bce_with_logits across possible signatures.

    Some versions expect pos_weight as a Tensor (pos_weight_t) and call .float() on it.
    This wrapper always constructs a Tensor first and tries several call styles.
    """
    pw_t = torch.tensor(float(pos_w), device=logits.device, dtype=logits.dtype)

    # 1) keyword pos_weight (tensor)
    try:
        return masked_bce_with_logits_fn(logits, y, valid, pos_weight=pw_t)
    except TypeError:
        pass

    # 2) keyword pos_weight_t (tensor)
    try:
        return masked_bce_with_logits_fn(logits, y, valid, pos_weight_t=pw_t)
    except TypeError:
        pass

    # 3) keyword pos_w (tensor)
    try:
        return masked_bce_with_logits_fn(logits, y, valid, pos_w=pw_t)
    except TypeError:
        pass

    # 4) positional 4th arg (tensor)
    return masked_bce_with_logits_fn(logits, y, valid, pw_t)

# -----------------------------------------------------------------------------
# Deterministic samplers
# -----------------------------------------------------------------------------

class FixedOrderSampler(Sampler[int]):
    """Yields a fixed list of indices in a fixed order."""
    def __init__(self, indices: Sequence[int]):
        self.indices = [int(i) for i in indices]

    def __iter__(self) -> Iterable[int]:
        yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)


class CyclingSampler(Sampler[int]):
    """Cycles through a fixed list of indices until num_samples is reached."""
    def __init__(self, indices: Sequence[int], num_samples: int):
        self.indices = [int(i) for i in indices]
        if len(self.indices) == 0:
            raise ValueError("CyclingSampler: empty indices")
        self.num_samples = int(num_samples)

    def __iter__(self) -> Iterable[int]:
        n = len(self.indices)
        for k in range(self.num_samples):
            yield self.indices[k % n]

    def __len__(self) -> int:
        return self.num_samples


# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------

def _tstats(name: str, t: torch.Tensor) -> str:
    t = t.detach()
    return (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={t.min().item():.4g} max={t.max().item():.4g} "
            f"mean={t.mean().item():.4g} std={t.std(unbiased=False).item():.4g}")


def _mask_counts(y: torch.Tensor, valid: torch.Tensor) -> Tuple[int, int, int, int]:
    # y, valid: float tensors [B,1,H,W]
    yb = (y > 0.5)
    vb = (valid > 0.5)
    pos = (yb & vb).sum().item()
    neg = ((~yb) & vb).sum().item()
    raw_pos = yb.sum().item()
    raw_neg = (~yb).sum().item()
    return int(pos), int(neg), int(raw_pos), int(raw_neg)


@torch.no_grad()
def auc_from_probs_hist(
    probs: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    n_bins: int = 256,
) -> float:
    """
    Pixel ROC AUC using histogram accumulation (matches your project approach),
    but operates on already-computed probs/targets/valid masks.

    Returns NaN if there are no positive or no negative valid pixels.
    """
    probs = probs.detach().float()
    targets = targets.detach().float().clamp(0.0, 1.0)
    valid = valid.detach().float()

    p = probs.reshape(-1)
    t = targets.reshape(-1)
    v = valid.reshape(-1)

    m = v > 0.5
    if not bool(m.any()):
        return float("nan")

    p = p[m]
    t = t[m]

    # Effective positives/negatives after mask
    P = float(t.sum().item())
    N = float((1.0 - t).sum().item())
    if P <= 0.0 or N <= 0.0:
        return float("nan")

    n_bins = int(n_bins)
    idx = torch.clamp((p * n_bins).to(torch.int64), 0, n_bins - 1)

    hist_pos = torch.bincount(idx, weights=t.to(torch.float64), minlength=n_bins).to(torch.float64)
    hist_neg = torch.bincount(idx, weights=(1.0 - t).to(torch.float64), minlength=n_bins).to(torch.float64)

    # Convert histograms into ROC curve and integrate
    tp = torch.cumsum(torch.flip(hist_pos, dims=[0]), dim=0)
    fp = torch.cumsum(torch.flip(hist_neg, dims=[0]), dim=0)

    tpr = tp / (P + 1e-12)
    fpr = fp / (N + 1e-12)

    # Ensure anchors (0,0) and (1,1) exist for trapezoid integration
    tpr = torch.cat([torch.zeros_like(tpr[:1]), tpr, torch.ones_like(tpr[:1])])
    fpr = torch.cat([torch.zeros_like(fpr[:1]), fpr, torch.ones_like(fpr[:1])])

    auc = torch.trapz(tpr, fpr).item()
    return float(auc)


@torch.no_grad()
def _polarity_auc_probe(
    *,
    probs: torch.Tensor,
    y: torch.Tensor,
    real: torch.Tensor,
    n_bins: int = 256,
) -> dict:
    """
    Compute AUC and P/N under two polarity assumptions:
      A) valid = ~real (expected via valid_mask_from_real)
      B) valid = real  (inversion test)
    """
    y_r = y
    real_r = resize_masks_to(real, y_r)

    # A) expected: valid = ~real
    valid_a = valid_mask_from_real(real_r)
    auc_a = auc_from_probs_hist(probs, y_r, valid_a, n_bins=n_bins)
    Pa, Na, rawPa, rawNa = _mask_counts(y_r, valid_a)

    # B) inverted: valid = real
    if real_r.dtype != torch.bool:
        real_b = (real_r > 0.5)
    else:
        real_b = real_r
    valid_b = real_b.to(dtype=torch.float32)
    auc_b = auc_from_probs_hist(probs, y_r, valid_b, n_bins=n_bins)
    Pb, Nb, rawPb, rawNb = _mask_counts(y_r, valid_b)

    real_frac = float((real_b.float().mean().item()))
    return {
        "A_valid_is_not_real": {
            "auc": float(auc_a), "P": Pa, "N": Na, "raw_pos": rawPa, "raw_neg": rawNa,
            "valid_frac": float(valid_a.mean().item()), "real_frac": real_frac
        },
        "B_valid_is_real": {
            "auc": float(auc_b), "P": Pb, "N": Nb, "raw_pos": rawPb, "raw_neg": rawNb,
            "valid_frac": float(valid_b.mean().item()), "real_frac": real_frac
        },
    }


def _default_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data construction (mirrors your run path)
# -----------------------------------------------------------------------------

def build_train_val_datasets(cfg: Config) -> tuple[Dataset, Dataset, dict]:
    base_ds = H5TiledDataset(cfg.data.train_h5, tile=cfg.data.tile, k_sigma=5.0)

    idx_tr, idx_va = split_indices(cfg.data.train_h5, val_frac=float(cfg.data.val_frac), seed=cfg.train.seed)
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))
    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    if not cfg.data.train_csv:
        raise ValueError("cfg.data.train_csv is not set (pass --train-csv).")

    touched_any_base = build_tile_mask_from_csv(
        cfg.data.train_csv,
        base_ds,
        tile=int(cfg.data.tile),
        margin_pix=float(cfg.data.margin_pix),
        stack_col=str(cfg.data.stack_col),
        stack_value=None,
    )
    missed_base = build_tile_mask_from_csv(
        cfg.data.train_csv,
        base_ds,
        tile=int(cfg.data.tile),
        margin_pix=float(cfg.data.margin_pix),
        stack_col=str(cfg.data.stack_col),
        stack_value=0,
    )
    detected_base = build_tile_mask_from_csv(
        cfg.data.train_csv,
        base_ds,
        tile=int(cfg.data.tile),
        margin_pix=float(cfg.data.margin_pix),
        stack_col=str(cfg.data.stack_col),
        stack_value=1,
    )

    train_ds = TileSubsetWithRealAndFlags(
        base_ds,
        tiles_tr,
        train_h5=cfg.data.train_h5,
        real_labels_key=str(cfg.data.real_labels_key),
        missed_mask_base=missed_base,
        detected_mask_base=detected_base,
    )
    val_ds = TileSubsetWithRealAndFlags(
        base_ds,
        tiles_va,
        train_h5=cfg.data.train_h5,
        real_labels_key=str(cfg.data.real_labels_key),
        missed_mask_base=missed_base,
        detected_mask_base=detected_base,
    )

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
            if bool(touched_any_base[bidx]):
                detected_local.append(j)
            else:
                background_local.append(j)

    info = {
        "base_len": len(base_ds),
        "train_tiles": len(train_ds),
        "val_tiles": len(val_ds),
        "missed_local": len(missed_local),
        "detected_local": len(detected_local),
        "background_local": len(background_local),
    }
    return train_ds, val_ds, info


def make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    sampler: Optional[Sampler[int]] = None,
    ddp_sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    def _wif(worker_id: int):
        return worker_init_fn(worker_id, base_seed=int(seed))

    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler if sampler is not None else ddp_sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(int(num_workers) > 0),
        prefetch_factor=2 if int(num_workers) > 0 else None,
        worker_init_fn=_wif if int(num_workers) > 0 else None,
    )


# -----------------------------------------------------------------------------
# Debug phases
# -----------------------------------------------------------------------------

@torch.no_grad()
def _similarity_y_real(y: torch.Tensor, real: torch.Tensor) -> dict:
    """
    Detect catastrophic bug: real mask == label mask (or highly correlated).
    """
    yb = (y > 0.5).float().reshape(-1)
    rb = (real > 0.5).float().reshape(-1)

    # exact equality rate on boolean masks
    eq = float((yb == rb).float().mean().item())

    # overlap statistics
    inter = float((yb * rb).sum().item())
    ysum = float(yb.sum().item())
    rsum = float(rb.sum().item())
    iou = float(inter / (ysum + rsum - inter + 1e-12))

    # correlation (safe if degenerate)
    yv = yb - yb.mean()
    rv = rb - rb.mean()
    denom = float((yv.norm() * rv.norm()).item())
    corr = float((yv @ rv).item() / denom) if denom > 0 else float("nan")

    return {"eq_rate": eq, "iou": iou, "corr": corr, "y_pos": ysum, "real_pos": rsum}


@torch.no_grad()
def phase_dataset_stats(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    n_bins: int,
    max_batches: int = 16,
) -> None:
    def _one_split(name: str, loader: DataLoader):
        print(f"\n=== DATASET STATS: {name} (first {max_batches} batches) ===")
        P0 = N0 = P1 = N1 = 0
        rb_fracs = []
        valid_fracs = []
        pos_raw_fracs = []
        pos_valid_fracs = []
        sim_reports = []

        for b, batch in enumerate(loader):
            if b >= max_batches:
                break

            # expected: (x, y, real, missed_flag, detected_flag)
            x, y, real = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)

            y_r = y
            real_r = resize_masks_to(real, y_r)
            valid = valid_mask_from_real(real_r)

            pos, neg, raw_pos, raw_neg = _mask_counts(y_r, valid)
            P0 += raw_pos
            N0 += raw_neg
            P1 += pos
            N1 += neg

            rb_fracs.append(float((real_r > 0.5).float().mean().item()))
            valid_fracs.append(float(valid.mean().item()))
            pos_raw_fracs.append(float((y_r > 0.5).float().mean().item()))
            if valid.sum().item() > 0:
                pos_valid_fracs.append(float(((y_r > 0.5) & (valid > 0.5)).float().sum().item() / valid.sum().item()))
            else:
                pos_valid_fracs.append(0.0)

            if b == 0:
                print(_tstats("x", x))
                print(_tstats("y", y_r))
                print(_tstats("real", real_r))
                print(_tstats("valid", valid))

                sim = _similarity_y_real(y_r, real_r)
                print(f"[{name} batch0] y_vs_real_similarity: {sim}")
                sim_reports.append(sim)

            if b < 3:
                # constant predictor sanity through your metric + mask path
                probs = torch.full_like(y_r, 0.5)
                probe = _polarity_auc_probe(probs=probs, y=y_r, real=real_r, n_bins=n_bins)
                print(f"[{name} batch {b}] polarity_probe (constant probs): {probe}")

                # also check similarity on first few batches
                sim = _similarity_y_real(y_r, real_r)
                print(f"[{name} batch {b}] y_vs_real_similarity: {sim}")

        if len(rb_fracs) == 0:
            print(f"{name}: no batches processed")
            return

        print(f"{name}: raw totals P={P0} N={N0} | valid totals P={P1} N={N1}")
        print(f"{name}: mean(real_frac)={np.mean(rb_fracs):.6f} mean(valid_frac)={np.mean(valid_fracs):.6f}")
        print(f"{name}: mean(pos_frac_raw)={np.mean(pos_raw_fracs):.8f} mean(pos_frac_valid)={np.mean(pos_valid_fracs):.8f}")
        print(f"{name}: pos_frac_raw quantiles: p50={np.quantile(pos_raw_fracs,0.5):.8f} p90={np.quantile(pos_raw_fracs,0.9):.8f} p99={np.quantile(pos_raw_fracs,0.99):.8f}")
        print(f"{name}: pos_frac_valid quantiles: p50={np.quantile(pos_valid_fracs,0.5):.8f} p90={np.quantile(pos_valid_fracs,0.9):.8f} p99={np.quantile(pos_valid_fracs,0.99):.8f}")

    _one_split("TRAIN", train_loader)
    _one_split("VAL", val_loader)


@torch.no_grad()
def phase_val_probe(
    model: torch.nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    n_bins: int,
    val_batches: int,
) -> None:
    model.eval()
    aucs = []
    bad_batches = 0
    nan_batches = 0

    for b, batch in enumerate(val_loader):
        if b >= val_batches:
            break
        x, y, real = batch[0], batch[1], batch[2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        real = real.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.sigmoid(logits)

        y_r = resize_masks_to(y, probs).float().clamp(0.0, 1.0)
        real_r = resize_masks_to(real, probs)
        valid = valid_mask_from_real(real_r)

        pos, neg, raw_pos, raw_neg = _mask_counts(y_r, valid)
        if pos == 0 or neg == 0:
            bad_batches += 1

        auc = auc_from_probs_hist(probs, y_r, valid, n_bins=n_bins)
        if not np.isfinite(auc):
            nan_batches += 1
        else:
            aucs.append(float(auc))

        if b < 3:
            print(f"[VAL batch {b}] auc={auc:.4f} P={pos} N={neg} rawP={raw_pos} rawN={raw_neg}")
            print(_tstats("logits", logits))
            print(_tstats("probs", probs))
            probe = _polarity_auc_probe(probs=probs, y=y_r, real=real_r, n_bins=n_bins)
            print(f"[VAL batch {b}] polarity_probe (model probs): {probe}")
            sim = _similarity_y_real(y_r, real_r)
            print(f"[VAL batch {b}] y_vs_real_similarity: {sim}")

    if len(aucs) == 0:
        print(f"\nVAL probe summary: NO finite AUC values. bad_batches={bad_batches} nan_batches={nan_batches}")
        return

    print(
        f"\nVAL probe summary: mean_auc={np.mean(aucs):.4f} std={np.std(aucs):.4f} "
        f"min={np.min(aucs):.4f} max={np.max(aucs):.4f} "
        f"bad_batches(P==0 or N==0)={bad_batches}/{val_batches} "
        f"nan_batches={nan_batches}/{val_batches}"
    )


def phase_overfit_one_batch(
    model: torch.nn.Module,
    trainer: Trainer,
    train_loader: DataLoader,
    *,
    device: torch.device,
    steps: int,
    lr: float,
    pos_weight: float,
    n_bins: int,
    log_every: int = 25,
) -> None:
    """
    Deterministic overfit on a single batch. If this can't push AUC up, the pipeline is broken.
    """
    model.train()

    batch = next(iter(train_loader))
    x, y, real = batch[0], batch[1], batch[2]
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    real = real.to(device, non_blocking=True)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=1e-4)

    # watch a few params
    watch = []
    for n, p in model.named_parameters():
        if p.requires_grad and len(watch) < 3:
            watch.append((n, p))

    # AMP enablement (Trainer API may differ across refactors)
    amp_enabled = bool(getattr(trainer, "amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # import your exact loss helper (used by your training)
    from ADCNN.train import masked_bce_with_logits

    # Pre-check: if y and real are identical, valid=~real will hide most positives
    with torch.no_grad():
        sim0 = _similarity_y_real(y, real)
        print(f"[OVERFIT1 pre] y_vs_real_similarity: {sim0}")
        real_r0 = resize_masks_to(y, real)
        valid0 = valid_mask_from_real(real_r0)
        pos0, neg0, raw_pos0, raw_neg0 = _mask_counts(y, valid0)
        print(f"[OVERFIT1 pre] P={pos0} N={neg0} rawP={raw_pos0} rawN={raw_neg0} valid_frac={valid0.mean().item():.6f}")

    for s in range(1, int(steps) + 1):
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            y_r = resize_masks_to(y, logits).float().clamp(0.0, 1.0)
            real_r = resize_masks_to(real, logits)
            valid = valid_mask_from_real(real_r)
            loss = masked_bce_compat(masked_bce_with_logits, logits, y_r, valid, float(pos_weight))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (s % log_every) == 0 or s in (1, 2, 5, 10):
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)
                y_r = resize_masks_to(y, probs).float().clamp(0.0, 1.0)
                real_r = resize_masks_to(real, probs)
                valid = valid_mask_from_real(real_r)

                pos, neg, raw_pos, raw_neg = _mask_counts(y_r, valid)
                auc = auc_from_probs_hist(probs, y_r, valid, n_bins=n_bins)

                vb = (valid > 0.5)
                yb = (y_r > 0.5)
                p_pos = probs[yb & vb].mean().item() if (yb & vb).any() else float("nan")
                p_neg = probs[(~yb) & vb].mean().item() if ((~yb) & vb).any() else float("nan")

                norms = [(n, float(p.detach().float().norm().item())) for n, p in watch]
                print(
                    f"[OVERFIT1 step {s:04d}] loss={loss.item():.6f} auc={float(auc):.4f} "
                    f"P={pos} N={neg} p_pos={p_pos:.4f} p_neg={p_neg:.4f} param_norms={norms}"
                )


# -----------------------------------------------------------------------------
# Config from CLI
# -----------------------------------------------------------------------------

def build_cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()

    # data
    cfg.data.train_h5 = args.train_h5
    cfg.data.train_csv = args.train_csv
    if args.tile is not None:
        cfg.data.tile = int(args.tile)
    if args.val_frac is not None:
        cfg.data.val_frac = float(args.val_frac)
    if args.real_labels_key is not None:
        cfg.data.real_labels_key = str(args.real_labels_key)
    if args.margin_pix is not None:
        cfg.data.margin_pix = float(args.margin_pix)
    if args.stack_col is not None:
        cfg.data.stack_col = str(args.stack_col)

    # loader
    if args.batch_size is not None:
        cfg.loader.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.loader.num_workers = int(args.num_workers)
    if args.pin_memory is not None:
        cfg.loader.pin_memory = bool(args.pin_memory)

    # train
    cfg.train.seed = int(args.seed)
    cfg.train.deterministic = bool(args.deterministic)

    # Debugger forces EMA off
    cfg.train.use_ema = False
    cfg.train.ema_eval = False

    cfg.validate()
    return cfg


# -----------------------------------------------------------------------------
# CLI / Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # --- Data paths (required) ---
    ap.add_argument("--train-h5", required=True, help="Path to training .h5.")
    ap.add_argument("--train-csv", required=True, help="Path to training .csv.")

    ap.add_argument("--tile", type=int, default=None, help="Tile size override.")
    ap.add_argument("--val-frac", type=float, default=None, help="Validation fraction override.")
    ap.add_argument("--real-labels-key", default=None, help="H5 key for real_labels mask.")
    ap.add_argument("--margin-pix", type=float, default=None, help="CSV -> tile mask margin (pixels).")
    ap.add_argument("--stack-col", default=None, help="CSV stack column name.")

    # --- Loader overrides ---
    ap.add_argument("--batch-size", type=int, default=None, help="Global batch size override.")
    ap.add_argument("--num-workers", type=int, default=None)

    ap.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    ap.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    ap.set_defaults(pin_memory=None)

    # --- Debug execution ---
    ap.add_argument("--mode", choices=["smoke", "overfit1", "subset", "ddpcheck"], default="overfit1")
    ap.add_argument("--device", choices=["single", "ddp"], default="single")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")

    # --- Deterministic subset selection ---
    ap.add_argument("--subset-train", type=int, default=2048)
    ap.add_argument("--subset-val", type=int, default=512)
    ap.add_argument("--subset-seed", type=int, default=1337)

    # --- AUC ---
    ap.add_argument("--auc-bins", type=int, default=None, help="Override number of histogram bins for AUC.")

    # --- Loop controls ---
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--val-batches", type=int, default=24)
    ap.add_argument("--log-every", type=int, default=25)

    # --- Overfit test knobs ---
    ap.add_argument("--overfit-lr", type=float, default=3e-4)
    ap.add_argument("--overfit-posw", type=float, default=8.0)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cfg_from_args(args)

    # DDP init only if requested
    is_dist, rank, local_rank, world_size = (False, 0, 0, 1)
    if args.device == "ddp":
        is_dist, rank, local_rank, world_size = init_distributed()

    if not is_main_process():
        builtins.print = lambda *a, **k: None

    set_seed(args.seed, deterministic=bool(args.deterministic))

    device = _default_device(local_rank if args.device == "ddp" else 0)

    train_ds, val_ds, info = build_train_val_datasets(cfg)

    if is_main_process():
        print("=== DEBUG SUITE START ===")
        print(f"mode={args.mode} device_mode={args.device} world_size={world_size} rank={rank} local_rank={local_rank}")
        print(f"train_h5={cfg.data.train_h5}")
        print(f"train_csv={cfg.data.train_csv}")
        print(f"tile={cfg.data.tile} val_frac={cfg.data.val_frac} real_labels_key={cfg.data.real_labels_key}")
        print(f"dataset_info={info}")
        print("EMA: disabled (forced by debugger, raw model eval)")

    # Deterministic subsets
    rng = np.random.default_rng(args.subset_seed)
    train_idx = np.arange(len(train_ds), dtype=np.int64)
    val_idx = np.arange(len(val_ds), dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_idx = train_idx[: min(args.subset_train, len(train_idx))]
    val_idx = val_idx[: min(args.subset_val, len(val_idx))]

    # Batch sizing (match your DDP behavior: global batch split across ranks)
    global_bs = int(cfg.loader.batch_size)
    per_gpu_bs = max(1, global_bs // int(world_size))

    # Samplers
    if args.mode == "overfit1":
        fixed = train_idx[:per_gpu_bs].tolist()
        train_sampler = CyclingSampler(fixed, num_samples=int(args.steps) * per_gpu_bs)
    else:
        train_sampler = FixedOrderSampler(train_idx.tolist())

    if args.device == "ddp":
        # NOTE: DDP train sampler here is not fully sharded; prefer single-GPU for debugging.
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = make_loader(
            train_ds,
            batch_size=per_gpu_bs,
            num_workers=cfg.loader.num_workers,
            pin_memory=cfg.loader.pin_memory,
            seed=cfg.train.seed,
            sampler=train_sampler,
            ddp_sampler=None,
        )
        val_loader = make_loader(
            val_ds,
            batch_size=per_gpu_bs,
            num_workers=cfg.loader.num_workers,
            pin_memory=cfg.loader.pin_memory,
            seed=cfg.train.seed,
            sampler=None,
            ddp_sampler=val_sampler,
        )
    else:
        train_loader = make_loader(
            train_ds,
            batch_size=per_gpu_bs,
            num_workers=cfg.loader.num_workers,
            pin_memory=cfg.loader.pin_memory,
            seed=cfg.train.seed,
            sampler=train_sampler,
        )
        val_loader = make_loader(
            val_ds,
            batch_size=per_gpu_bs,
            num_workers=cfg.loader.num_workers,
            pin_memory=cfg.loader.pin_memory,
            seed=cfg.train.seed,
            sampler=FixedOrderSampler(val_idx.tolist()),
        )

    # Model + trainer
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    # Trainer signature may vary across refactors; be tolerant.
    try:
        trainer = Trainer(device=device, use_amp=True)
    except TypeError:
        trainer = Trainer(device=device)

    # AUC bins: CLI override > cfg default > fallback
    n_bins = int(args.auc_bins) if args.auc_bins is not None else int(getattr(cfg.train, "auc_bins", 256))

    # Always print dataset stats first
    if is_main_process():
        phase_dataset_stats(train_loader, val_loader, device=device, n_bins=n_bins, max_batches=16)

    if args.mode == "smoke":
        if is_main_process():
            phase_val_probe(model, val_loader, device=device, n_bins=n_bins, val_batches=int(args.val_batches))
        return

    if args.mode == "overfit1":
        if is_main_process():
            phase_overfit_one_batch(
                model,
                trainer,
                train_loader,
                device=device,
                steps=int(args.steps),
                lr=float(args.overfit_lr),
                pos_weight=float(args.overfit_posw),
                n_bins=n_bins,
                log_every=int(args.log_every),
            )
            phase_val_probe(model, val_loader, device=device, n_bins=n_bins, val_batches=int(args.val_batches))
        return

    if args.mode == "subset":
        # Optional: only works if your Trainer exposes this method.
        if not hasattr(trainer, "train_full_probe"):
            raise RuntimeError("Trainer.train_full_probe() not found. Use --mode overfit1 or smoke.")
        if is_main_process():
            print("\n=== SUBSET MODE: calling Trainer.train_full_probe (EMA OFF) ===")

        trainer.train_full_probe(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            seed=int(args.seed),
            warmup_epochs=1,
            warmup_batches=min(400, int(args.steps)),
            head_epochs=1,
            head_batches=min(800, int(args.steps)),
            tail_epochs=1,
            tail_batches=min(800, int(args.steps)),
            max_epochs=3,
            val_every=1,
            auc_batches=min(12, int(args.val_batches)),
            auc_bins=n_bins,
            use_ema=False,
            ema_eval=False,
            save_best_to=str(Path(cfg.train.save_best_to).with_suffix(".debug_best.pt")),
            save_last_to=str(Path(cfg.train.save_last_to).with_suffix(".debug_last.pt")),
            verbose=2,
            expected_tile=int(cfg.data.tile),
            model_hparams={"in_ch": 1, "out_ch": 1},
            norm_name="medmad_clip_k5",
        )
        return

    if args.mode == "ddpcheck":
        if is_main_process():
            print("\n=== DDP CHECK: val-only probe (raw weights, EMA OFF) ===")
        phase_val_probe(model, val_loader, device=device, n_bins=n_bins, val_batches=int(args.val_batches))
        return


if __name__ == "__main__":
    main()