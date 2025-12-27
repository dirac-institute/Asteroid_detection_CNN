#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# -------------------------
# Repo / import setup
# -------------------------
def setup_imports(repo_root: str):
    proj = Path(repo_root).resolve()
    # allow: import ADCNN.*
    if str(proj) not in sys.path:
        sys.path.insert(0, str(proj))
    # allow: import utils.* where utils == ADCNN/utils (your code expects `utils.*`)
    if str(proj / "ADCNN") not in sys.path:
        sys.path.insert(0, str(proj / "ADCNN"))
    return proj

# -------------------------
# Local project imports (after sys.path)
# -------------------------
def import_project():
    from ADCNN.data.h5tiles import H5TiledDataset
    from ADCNN.models.unet_res_se import UNetResSEASPP
    from ADCNN.config import Config
    from ADCNN.train import Trainer
    from ADCNN.utils.utils import set_seed, split_indices
    return H5TiledDataset, UNetResSEASPP, Config, Trainer, set_seed, split_indices

# -------------------------
# Data helpers
# -------------------------
class TileSubset(torch.utils.data.Dataset):
    """
    Wrap H5TiledDataset and select only certain base tile indices.
    base[i] expects i to be an index into base_ds.indices (tile index).
    """
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

def build_datasets(H5TiledDataset, train_h5: str, test_h5: str, tile: int, seed: int, val_frac: float):
    # panel split
    from ADCNN.utils.utils import split_indices
    idx_tr, idx_va = split_indices(train_h5, val_frac=val_frac, seed=seed)
    idx_tr_set = set(idx_tr.tolist())
    idx_va_set = set(idx_va.tolist())

    ds_base = H5TiledDataset(train_h5, tile=tile, k_sigma=5.0)
    ds_te   = H5TiledDataset(test_h5,  tile=tile, k_sigma=5.0)

    tiles_tr = filter_tiles_by_panels(ds_base, idx_tr_set)
    tiles_va = filter_tiles_by_panels(ds_base, idx_va_set)

    train_ds = TileSubset(ds_base, tiles_tr)
    val_ds   = TileSubset(ds_base, tiles_va)

    return train_ds, val_ds, ds_te

def build_loaders(train_ds, val_ds, test_ds, batch_size: int, num_workers: int, pin_memory: bool):
    # On HPC: num_workers>0 is fine; on notebooks you used 0. Keep it configurable.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader, test_loader

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Baseline training (no curriculum, no reweighting).")
    p.add_argument("--repo-root", type=str, default="../", help="Repo root that contains ADCNN/")
    p.add_argument("--train-h5", type=str, default="/home/karlo/train.h5")
    p.add_argument("--test-h5",  type=str, default="../DATA/test.h5")
    p.add_argument("--train-csv", type=str, default="../DATA/train.csv")

    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true", default=False)

    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--val-every", type=int, default=None, help="If None: evaluate only at end (epochs).")

    p.add_argument("--save-dir", type=str, default="../checkpoints/Experiments")
    p.add_argument("--tag", type=str, default="Baseline")
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    proj = setup_imports(args.repo_root)
    H5TiledDataset, UNetResSEASPP, Config, Trainer, set_seed, split_indices = import_project()

    # Repro
    set_seed(args.seed)

    # Data
    train_ds, val_ds, test_ds = build_datasets(
        H5TiledDataset,
        train_h5=args.train_h5,
        test_h5=args.test_h5,
        tile=args.tile,
        seed=args.seed,
        val_frac=args.val_frac,
    )
    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)} | Test tiles: {len(test_ds)}")

    # Config
    cfg = Config()
    cfg.train.max_epochs = args.max_epochs
    cfg.train.val_every = args.val_every if args.val_every is not None else args.max_epochs

    # Output
    save_dir = Path(args.save_dir).resolve()
    (save_dir / "Best").mkdir(parents=True, exist_ok=True)
    (save_dir / "Last").mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "Best" / f"{args.tag}.pt")
    last_path = str(save_dir / "Last" / f"{args.tag}.pt")

    # Model + train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)
    trainer = Trainer(device=device)

    model, thr, summary = trainer.train_full_probe(
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
        verbose=args.verbose
    )

    print("Final thr:", float(thr))
    print("Summary:", summary)
    print("Best ckpt:", best_path)
    print("Last ckpt:", last_path)

if __name__ == "__main__":
    main()
