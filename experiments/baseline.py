import os, sys, math, time, copy, gc, h5py
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = "../"
proj = Path(REPO_ROOT).resolve()

# allow: import ADCNN.*
sys.path.insert(0, str(proj))

# allow: import utils.*  (where utils == ADCNN/utils)
sys.path.insert(0, str(proj / "ADCNN"))

from ADCNN.data.h5tiles import H5TiledDataset
from ADCNN.models.unet_res_se import UNetResSEASPP
from ADCNN.predict import load_model, predict_tiles_to_full
from ADCNN.config import Config
from ADCNN.train import Trainer
from ADCNN.utils.utils import set_seed, split_indices
import ADCNN.evaluation as evaluation
from ADCNN.utils.utils import draw_one_line
from tqdm import tqdm

EPOCHS = 10
SAVE_PATH = "../checkpoints/Experiments/"

cfg_baseline = Config()
cfg_baseline.train.max_epochs = EPOCHS
cfg_baseline.train.val_every = EPOCHS

@dataclass
class BaselineCfg:
    train_h5: str = "/home/karlo/train.h5"
    train_csv: str = "../DATA/train.csv"
    test_h5:  str = "../DATA/test.h5"
    test_csv: str = "../DATA/test.csv"
    tile: int = 128
    seed: int = 1337

    # loader
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True

    # training
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pos_weight: float = 8.0            # baseline imbalance handling
    grad_clip: float = 1.0
    amp: bool = True

    # quick eval
    val_every: int = 1
    max_val_batches: int = 60          # keep small for sandbox speed

cfg_data = BaselineCfg()

set_seed(cfg_data.seed)

idx_tr, idx_va = split_indices(cfg_data.train_h5, val_frac=0.1, seed=cfg_data.seed)

ds_tr = H5TiledDataset(cfg_data.train_h5, tile=cfg_data.tile, k_sigma=5.0)
ds_va = H5TiledDataset(cfg_data.train_h5, tile=cfg_data.tile, k_sigma=5.0)
ds_te = H5TiledDataset(cfg_data.test_h5,  tile=cfg_data.tile, k_sigma=5.0)

# Restrict to panels via indices (panels->tiles handled inside H5TiledDataset indices list),
# simplest baseline: just use full tiled dataset and do a tile-level split by panel id.
# We'll filter tiles by panel id membership.
idx_tr_set = set(idx_tr.tolist())
idx_va_set = set(idx_va.tolist())

def filter_tiles_by_panels(base_ds, panel_id_set):
    kept = []
    for k, (pid, r, c) in enumerate(base_ds.indices):
        if pid in panel_id_set:
            kept.append(k)
    return kept

tiles_tr = filter_tiles_by_panels(ds_tr, idx_tr_set)
tiles_va = filter_tiles_by_panels(ds_va, idx_va_set)

class TileSubset(torch.utils.data.Dataset):
    def __init__(self, base, tile_indices):
        self.base = base
        self.tile_indices = np.asarray(tile_indices, dtype=np.int64)
    def __len__(self): return len(self.tile_indices)
    def __getitem__(self, i): return self.base[int(self.tile_indices[i])]

train_ds = TileSubset(ds_tr, tiles_tr)
val_ds   = TileSubset(ds_va, tiles_va)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg_data.batch_size,
    shuffle=True,
    num_workers=cfg_data.num_workers,
    pin_memory=cfg_data.pin_memory,
    persistent_workers=(cfg_data.num_workers > 0),
    prefetch_factor=2 if cfg_data.num_workers > 0 else None,
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg_data.batch_size,
    shuffle=False,
    num_workers=cfg_data.num_workers,
    pin_memory=cfg_data.pin_memory,
    persistent_workers=(cfg_data.num_workers > 0),
    prefetch_factor=2 if cfg_data.num_workers > 0 else None,
)

test_loader = DataLoader(
    ds_te,
    batch_size=cfg_data.batch_size,
    shuffle=False,
    num_workers=cfg_data.num_workers,
    pin_memory=cfg_data.pin_memory,
    persistent_workers=(cfg_data.num_workers > 0),
    prefetch_factor=2 if cfg_data.num_workers > 0 else None,
)

test_catalog = pd.read_csv(cfg_data.test_csv)
train_catalog = pd.read_csv(cfg_data.train_csv)
with h5py.File(cfg_data.test_h5, "r") as _f:
    gt_test = _f["masks"][:].astype(np.uint8)
print("Train tiles:", len(train_ds), "Val tiles:", len(val_ds), "Test tiles:", len(ds_te))

# ------------ model and trainer ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_baseline = UNetResSEASPP(in_ch=1, out_ch=1).to(device)
trainer = Trainer(device=device)
model_baseline, thr, summary = trainer.train_full_probe(
        model_baseline, train_loader=train_loader, val_loader=val_loader,
        seed=cfg_baseline.train.seed,
        init_head_prior=cfg_baseline.train.init_head_prior,
        warmup_epochs=cfg_baseline.train.warmup_epochs, warmup_batches=cfg_baseline.train.warmup_batches,
        warmup_lr=cfg_baseline.train.warmup_lr, warmup_pos_weight=cfg_baseline.train.warmup_pos_weight,
        head_epochs=cfg_baseline.train.head_epochs, head_batches=cfg_baseline.train.head_batches,
        head_lr=cfg_baseline.train.head_lr, head_pos_weight=cfg_baseline.train.head_pos_weight,
        tail_epochs=cfg_baseline.train.tail_epochs, tail_batches=cfg_baseline.train.tail_batches,
        tail_lr=cfg_baseline.train.tail_lr, tail_pos_weight=cfg_baseline.train.tail_pos_weight,
        max_epochs=cfg_baseline.train.max_epochs, val_every=cfg_baseline.train.val_every,
        base_lrs=cfg_baseline.train.base_lrs, weight_decay=cfg_baseline.train.weight_decay,
        thr_beta=cfg_baseline.train.thr_beta, long_batches=cfg_baseline.train.long_batches,
        thr_pos_rate_early=cfg_baseline.train.thr_pos_rate_early, thr_pos_rate_late=cfg_baseline.train.thr_pos_rate_late,
        save_best_to=SAVE_PATH+"Best/Baseline.pt", save_last_to=SAVE_PATH+"Last/Baseline.pt",
        verbose = 2
    )

