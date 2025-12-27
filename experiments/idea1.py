#!/usr/bin/env python3
import os
import sys
import math
from pathlib import Path
from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader, Sampler

# -------------------------
# Repo / import setup
# -------------------------
def setup_imports(repo_root: str):
    proj = Path(repo_root).resolve()
    if str(proj) not in sys.path:
        sys.path.insert(0, str(proj))
    if str(proj / "ADCNN") not in sys.path:
        sys.path.insert(0, str(proj / "ADCNN"))
    return proj

def import_project():
    from ADCNN.data.h5tiles import H5TiledDataset
    from ADCNN.models.unet_res_se import UNetResSEASPP
    from ADCNN.config import Config
    from ADCNN.train import Trainer
    from ADCNN.utils.utils import set_seed, split_indices
    return H5TiledDataset, UNetResSEASPP, Config, Trainer, set_seed, split_indices

# -------------------------
# Dataset wrapper
# -------------------------
class TileSubset(torch.utils.data.Dataset):
    def __init__(self, base, tile_indices):
        self.base = base
        self.tile_indices = np.asarray(tile_indices, dtype=np.int64)
    def __len__(self):
        return len(self.tile_indices)
    def __getitem__(self, i):
        return self.base[int(self.tile_indices[i])]

def filter_tiles_by_panels(base_ds, panel_id_set):
    kept = []
    for k, (pid, r, c) in enumerate(base_ds.indices):
        if pid in panel_id_set:
            kept.append(k)
    return kept

# -------------------------
# CSV -> hard tile mask
# -------------------------
def compute_tile_grid(H, W, tile):
    return math.ceil(H / tile), math.ceil(W / tile)

def bbox_tiles_touched(x, y, R, tile, Hb, Wb):
    x0, x1 = x - R, x + R
    y0, y1 = y - R, y + R
    c0 = int(math.floor(x0 / tile)); c1 = int(math.floor(x1 / tile))
    r0 = int(math.floor(y0 / tile)); r1 = int(math.floor(y1 / tile))
    c0 = max(0, min(Wb - 1, c0)); c1 = max(0, min(Wb - 1, c1))
    r0 = max(0, min(Hb - 1, r0)); r1 = max(0, min(Hb - 1, r1))
    return r0, r1, c0, c1

def build_hard_tile_mask_from_csv(base_ds, csv_path: str, *, margin: float):
    """
    hard_mask_base[k] = True if base tile k overlaps bbox of any LSST-missed injection.
    """
    df = pd.read_csv(csv_path)
    required = {"image_id", "x", "y", "trail_length", "stack_detection"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    N, H, W = base_ds.N, base_ds.H, base_ds.W
    Hb, Wb = compute_tile_grid(H, W, base_ds.tile)

    # Map (panel, r, c) -> base tile index
    base_map = {(i, r, c): k for k, (i, r, c) in enumerate(base_ds.indices)}

    missed = df[df["stack_detection"].astype(int) == 0].copy()
    if len(missed) == 0:
        print("WARNING: stack_detection==0 yielded 0 rows (no LSST-missed).")

    if len(missed):
        if missed["image_id"].min() < 0 or missed["image_id"].max() >= N:
            raise ValueError(f"image_id out of range for train.h5: N={N}, "
                             f"min={missed['image_id'].min()}, max={missed['image_id'].max()}")

    hard_mask = np.zeros(len(base_ds), dtype=bool)

    for row in missed.itertuples(index=False):
        i = int(getattr(row, "image_id"))
        x = float(getattr(row, "x"))
        y = float(getattr(row, "y"))
        L = float(getattr(row, "trail_length"))
        R = 0.5 * L + float(margin)

        r0, r1, c0, c1 = bbox_tiles_touched(x, y, R, base_ds.tile, Hb, Wb)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                k = base_map.get((i, r, c))
                if k is not None:
                    hard_mask[k] = True

    return hard_mask

def filter_tiles_by_truth(h5_path: str, tiled_ds, tile_indices: np.ndarray, *, min_pos_pix: int = 1):
    """
    Keep only tile indices whose *truth mask* has >= min_pos_pix positives.
    IMPORTANT: tile_indices are indices into tiled_ds.indices (base tile indices).
    """
    tile_indices = np.asarray(tile_indices, dtype=np.int64)
    kept = []

    with h5py.File(h5_path, "r") as f:
        Y = f["masks"]
        H, W = Y.shape[1:]
        t = int(tiled_ds.tile)

        for base_tile_idx in tile_indices:
            panel_i, r, c = tiled_ds.indices[int(base_tile_idx)]
            r0, c0 = r * t, c * t
            r1, c1 = min(r0 + t, H), min(c0 + t, W)
            y = Y[panel_i, r0:r1, c0:c1]
            if np.count_nonzero(y) >= min_pos_pix:
                kept.append(int(base_tile_idx))

    return np.asarray(kept, dtype=np.int64)

# -------------------------
# Curriculum sampler
# -------------------------
def p_hard_linear(epoch, p0, p1, e_start, e_end):
    if epoch <= e_start: return p0
    if epoch >= e_end:   return p1
    t = (epoch - e_start) / max(e_end - e_start, 1e-12)
    return p0 + (p1 - p0) * t

def p_hard_sigmoid(epoch, p0, p1, e_mid, width):
    z = (epoch - e_mid) / max(width, 1e-12)
    s = 1.0 / (1.0 + math.exp(-z))
    return p0 + (p1 - p0) * s

class CurriculumBatchSampler(Sampler[list[int]]):
    """
    Yields batches of indices in [0, len(train_ds)) (i.e. indices into TileSubset),
    mixing `hard_indices` and `other_indices` with epoch-dependent p_hard.
    """
    def __init__(
        self,
        hard_indices: np.ndarray,
        other_indices: np.ndarray,
        batch_size: int,
        seed: int,
        schedule: str,
        p1: float,
        e_start: int,
        e_end: int,
        e_mid: int,
        width: float,
        drop_last: bool = True,
    ):
        self.hard = np.asarray(hard_indices, dtype=np.int64)
        self.other = np.asarray(other_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.schedule = str(schedule)
        self.p1 = float(p1)
        self.e_start = int(e_start)
        self.e_end = int(e_end)
        self.e_mid = int(e_mid)
        self.width = float(width)
        self.drop_last = bool(drop_last)

        if len(self.hard) == 0:
            raise ValueError("Hard set empty after filtering. Nothing to oversample.")
        if len(self.other) == 0:
            raise ValueError("Other set empty. (This should not happen.)")

        self.p0 = len(self.hard) / (len(self.hard) + len(self.other))
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _p_hard(self):
        if self.schedule == "linear":
            return float(p_hard_linear(self.epoch, self.p0, self.p1, self.e_start, self.e_end))
        if self.schedule == "sigmoid":
            return float(p_hard_sigmoid(self.epoch, self.p0, self.p1, self.e_mid, self.width))
        raise ValueError(f"Unknown schedule={self.schedule} (use linear|sigmoid)")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        hard = rng.permutation(self.hard)
        other = rng.permutation(self.other)

        ph = self._p_hard()
        nh = int(round(ph * self.batch_size))
        nh = max(1, min(self.batch_size - 1, nh))
        no = self.batch_size - nh

        hi = oi = 0
        total = len(self.hard) + len(self.other)
        n_batches = total // self.batch_size
        if not self.drop_last:
            n_batches = math.ceil(total / self.batch_size)

        for _ in range(n_batches):
            if hi + nh > len(hard):
                hard = rng.permutation(self.hard); hi = 0
            if oi + no > len(other):
                other = rng.permutation(self.other); oi = 0

            batch = np.concatenate([hard[hi:hi+nh], other[oi:oi+no]])
            rng.shuffle(batch)
            hi += nh
            oi += no
            yield batch.tolist()

    def __len__(self):
        total = len(self.hard) + len(self.other)
        return total // self.batch_size

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Idea1: curriculum oversampling of LSST-missed tiles.")
    p.add_argument("--repo-root", type=str, default="../")
    p.add_argument("--train-h5", type=str, default="/home/karlo/train.h5")
    p.add_argument("--train-csv", type=str, default="../DATA/train.csv")
    p.add_argument("--test-h5",  type=str, default="../DATA/test.h5")
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true", default=False)

    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--val-every", type=int, default=None)

    p.add_argument("--margin", type=float, default=30.0, help="BBox margin in pixels")
    p.add_argument("--min-pos-pix", type=int, default=1, help="Truth purity filter for hard tiles")
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "sigmoid"])
    p.add_argument("--p1", type=float, default=0.20, help="Target hard fraction late")
    p.add_argument("--sigmoid-mid", type=int, default=9)
    p.add_argument("--sigmoid-width", type=float, default=2.0)

    p.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    p.add_argument("--tag", type=str, default="idea1")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    setup_imports(args.repo_root)
    H5TiledDataset, UNetResSEASPP, Config, Trainer, set_seed, split_indices = import_project()
    set_seed(args.seed)

    # base dataset
    base_ds = H5TiledDataset(args.train_h5, tile=args.tile, k_sigma=5.0)

    # panel split -> tile subset
    idx_tr, idx_va = split_indices(args.train_h5, val_frac=args.val_frac, seed=args.seed)
    idx_tr_set = set(idx_tr.tolist())
    idx_va_set = set(idx_va.tolist())

    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    train_ds = TileSubset(base_ds, tiles_tr)
    val_ds   = TileSubset(base_ds, tiles_va)

    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    # hard mask on BASE tiles, then align to train_ds tiles
    hard_mask_base = build_hard_tile_mask_from_csv(base_ds, args.train_csv, margin=args.margin)
    hard_mask_train = hard_mask_base[train_ds.tile_indices]  # length == len(train_ds)

    hard_idx_raw = np.flatnonzero(hard_mask_train).astype(np.int64)

    # IMPORTANT: hard_idx_raw are indices into train_ds, but we need base tile indices for truth filtering:
    # - train_ds.tile_indices maps train_ds idx -> base tile idx
    # We filter in base-tile space, then map back to train_ds index space.
    base_tile_candidates = train_ds.tile_indices[hard_idx_raw]
    base_tile_kept = filter_tiles_by_truth(args.train_h5, base_ds, base_tile_candidates, min_pos_pix=args.min_pos_pix)

    # Map kept base tile indices back to train_ds indices
    base_to_train = {int(b): int(ti) for ti, b in enumerate(train_ds.tile_indices)}
    hard_idx = np.array([base_to_train[int(b)] for b in base_tile_kept if int(b) in base_to_train], dtype=np.int64)

    hard_mask = np.zeros(len(train_ds), dtype=bool)
    hard_mask[hard_idx] = True
    other_idx = np.flatnonzero(~hard_mask).astype(np.int64)

    print(f"Hard tiles after truth filter: {len(hard_idx)} | Other tiles: {len(other_idx)}")

    # loaders
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # schedule boundaries: ramp only across long training
    cfg = Config()
    cfg.train.max_epochs = args.max_epochs
    cfg.train.val_every = args.val_every if args.val_every is not None else args.max_epochs

    e_start = cfg.train.warmup_epochs + cfg.train.head_epochs + cfg.train.tail_epochs
    e_end   = e_start + cfg.train.max_epochs

    batch_sampler = CurriculumBatchSampler(
        hard_indices=hard_idx,
        other_indices=other_idx,
        batch_size=args.batch_size,
        seed=args.seed,
        schedule=args.schedule,
        p1=args.p1,
        e_start=e_start,
        e_end=e_end,
        e_mid=args.sigmoid_mid,
        width=args.sigmoid_width,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # output
    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    # train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)
    trainer = Trainer(device=device)

    model, thr, summary = trainer.train_full_probe(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed=cfg.train.seed,
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

        max_epochs=cfg.train.max_epochs,
        long_batches=cfg.train.long_batches,
        val_every=cfg.train.val_every,
        base_lrs=cfg.train.base_lrs,
        weight_decay=cfg.train.weight_decay,

        thr_beta=cfg.train.thr_beta,
        thr_pos_rate_early=cfg.train.thr_pos_rate_early,
        thr_pos_rate_late=cfg.train.thr_pos_rate_late,

        save_best_to=best_path,
        save_last_to=last_path,
        verbose=args.verbose,
    )

    print("Final thr:", float(thr))
    print("Summary:", summary)
    print("Best ckpt:", best_path)
    print("Last ckpt:", last_path)

if __name__ == "__main__":
    main()
