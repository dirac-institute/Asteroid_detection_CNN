import argparse
import builtins
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from ADCNN.core.config import Config
from ADCNN.utils.helpers import set_seed, split_indices, worker_init_fn
from ADCNN.data.datasets import H5TiledDataset
from ADCNN.utils.dist_utils import init_distributed, is_main_process
from ADCNN.core.model import UNetResSEASPP
from ADCNN.train import Trainer


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def tiles_touched_by_bbox(xc: float, yc: float, R: float, H: int, W: int, tile: int) -> Tuple[int, int, int, int]:
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
    Returns boolean mask of length len(base_ds) marking base tiles touched by injection bbox.
    stack_value:
      - None: any injection
      - 0: only rows where stack_col==0 (LSST missed)
      - 1: only rows where stack_col==1 (LSST detected)
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if margin_pix < 0.0:
        raise ValueError(f"margin_pix must be non-negative, got {margin_pix}")

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

    # iterrows is OK for moderate CSV sizes; if huge, switch to itertuples().
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


def filter_tiles_by_panels(base_ds, panel_id_set: set[int]) -> np.ndarray:
    kept = [k for k, (pid, _r, _c) in enumerate(base_ds.indices) if int(pid) in panel_id_set]
    return np.asarray(kept, dtype=np.int64)


class TileSubsetWithRealAndFlags(Dataset):
    """
    Dataset yielding (x, y, real, missed_flag, detected_flag)
    - x,y from base_ds tiles
    - real from H5 dataset real_labels_key (if missing -> zeros)
    - missed/detected flags from base masks (len(base_ds))
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
        self._real_labels_ds = None

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(self.base_tile_indices.size)

    def _ensure_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.train_h5, "r")
            self._has_real = (self.real_labels_key in self._h5)
            self._real_labels_ds = self._h5[self.real_labels_key] if self._has_real else None

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

        rl = self._real_labels_ds  # [N,H,W]
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


class DistributedMixtureSampler(Sampler[int]):
    """
    DDP-safe sampler that enforces mixture fractions across:
      - missed tiles
      - detected tiles
      - background tiles
    Produces LOCAL dataset indices (0..len(ds)-1) then shards across ranks.
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

        # Per-rank epoch size default: ceil(N / world)
        if epoch_size is None:
            self.epoch_size = int(math.ceil(self.dataset_size / max(self.num_replicas, 1)))
        else:
            self.epoch_size = int(epoch_size)

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self) -> int:
        # per-rank samples produced
        return int(self.epoch_size)

    def __iter__(self):
        g = np.random.default_rng(self.seed + 10_000 * self._epoch + self.rank)

        total = self.epoch_size * self.num_replicas

        fm, fd, fb = self.frac_missed, self.frac_detected, self.frac_background
        if self.missed_ids.size == 0 and fm > 0:
            fb += fm
            fm = 0.0
        if self.detected_ids.size == 0 and fd > 0:
            fb += fd
            fd = 0.0

        bg_ids = self.background_ids
        if bg_ids.size == 0 and fb > 0:
            bg_ids = np.arange(self.dataset_size, dtype=np.int64)

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
        ids_b = g.choice(bg_ids, size=nb, replace=True) if nb > 0 else np.empty((0,), np.int64)

        ids = np.concatenate([ids_m, ids_d, ids_b], axis=0)
        g.shuffle(ids)

        shard = ids[self.rank : ids.size : self.num_replicas]
        shard = shard[: self.epoch_size]
        return iter(shard.tolist())


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------


def run(cfg: Config, args: argparse.Namespace):
    is_dist, rank, local_rank, world_size = init_distributed()

    # silence non-main prints
    if not is_main_process():
        builtins.print = lambda *a, **k: None

    # determinism toggle
    set_seed(cfg.train.seed, deterministic=bool(args.deterministic))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base tiles dataset
    base_ds = H5TiledDataset(cfg.data.train_h5, tile=cfg.data.tile, k_sigma=5.0)

    # split by PANELS
    idx_tr, idx_va = split_indices(cfg.data.train_h5, val_frac=float(cfg.data.val_frac), seed=cfg.train.seed)
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))
    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    if not cfg.data.train_csv:
        raise ValueError(
            "train_csv is not set. Provide it in config.py (cfg.data.train_csv) or via --train-csv."
        )

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

    # build TRAIN-local bucket ids (indices in train_ds: 0..len(train_ds)-1)
    missed_local: List[int] = []
    detected_local: List[int] = []
    background_local: List[int] = []

    base_indices = train_ds.base_tile_indices
    for j in range(len(train_ds)):
        bidx = int(base_indices[j])  # index into base_ds
        if bool(missed_base[bidx]):
            missed_local.append(j)
        elif bool(detected_base[bidx]):
            detected_local.append(j)
        else:
            # touched but neither missed nor detected -> treat as detected-like (hard negatives)
            if bool(touched_any_base[bidx]):
                detected_local.append(j)
            else:
                background_local.append(j)

    missed_local = np.asarray(missed_local, dtype=np.int64)
    detected_local = np.asarray(detected_local, dtype=np.int64)
    background_local = np.asarray(background_local, dtype=np.int64)

    use_ddp = bool(is_dist and world_size > 1)
    epoch_size = None if int(cfg.train.epoch_size) <= 0 else int(cfg.train.epoch_size)

    # samplers
    if use_ddp:
        train_sampler = DistributedMixtureSampler(
            dataset_size=len(train_ds),
            missed_ids=missed_local,
            detected_ids=detected_local,
            background_ids=background_local,
            num_replicas=world_size,
            rank=rank,
            seed=int(cfg.train.seed),
            frac_missed=float(cfg.train.frac_missed),
            frac_detected=float(cfg.train.frac_detected),
            frac_background=float(cfg.train.frac_background),
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
            seed=int(cfg.train.seed),
            frac_missed=float(cfg.train.frac_missed),
            frac_detected=float(cfg.train.frac_detected),
            frac_background=float(cfg.train.frac_background),
            epoch_size=epoch_size,
        )
        val_sampler = None

    # Seed workers deterministically using cfg.train.seed
    def _wif(worker_id: int):
        return worker_init_fn(worker_id, base_seed=int(cfg.train.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=(cfg.loader.num_workers > 0),
        prefetch_factor=2 if cfg.loader.num_workers > 0 else None,
        worker_init_fn=_wif if cfg.loader.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=(cfg.loader.num_workers > 0),
        prefetch_factor=2 if cfg.loader.num_workers > 0 else None,
        worker_init_fn=_wif if cfg.loader.num_workers > 0 else None,
    )

    # model (trainer handles full resume from save_last_to)
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    # sanity-check resume file existence early
    if args.resume_epoch is not None:
        ckpt_path = Path(cfg.train.save_last_to)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume-epoch set but checkpoint not found: {ckpt_path}")

    trainer = Trainer(device=device)

    model_hparams = {"in_ch": 1, "out_ch": 1}
    # If your model exposes widths, include them for strict validation.
    if hasattr(model, "widths"):
        try:
            model_hparams["widths"] = tuple(getattr(model, "widths"))
        except Exception:
            pass

    # If you have a named normalization scheme, pass it here; otherwise keep None.
    norm_name = "medmad_clip_k5"

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
        val_every=cfg.train.val_every,
        base_lrs=cfg.train.base_lrs,
        weight_decay=cfg.train.weight_decay,
        ramp_kind=cfg.train.ramp_kind,
        ramp_start_epoch=cfg.train.ramp_start_epoch,
        ramp_end_epoch=cfg.train.ramp_end_epoch,
        sigmoid_k=cfg.train.sigmoid_k,
        bce_pos_weight_long=cfg.train.bce_pos_weight_long,
        ft_alpha=cfg.train.ft_alpha,
        ft_gamma=cfg.train.ft_gamma,
        fixed_thr=cfg.train.fixed_thr,
        auc_batches=cfg.train.auc_batches,
        long_batches=cfg.train.long_batches,
        resume_epoch=args.resume_epoch,
        save_best_to=cfg.train.save_best_to,
        save_last_to=cfg.train.save_last_to,
        verbose=cfg.train.verbose,
        expected_tile=int(cfg.data.tile),
        model_hparams=model_hparams,
        norm_name=norm_name,
        use_ema=not bool(args.no_ema),
        ema_decay=float(args.ema_decay),
        ema_eval=True,
    )

    if is_main_process():
        print("Final threshold:", thr)
        print("Summary:", summary)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str, default=None)
    ap.add_argument("--resume-epoch", dest="resume_epoch", type=int, default=None)
    ap.add_argument("--train-h5", type=str, default="/home/karlo/test.h5", help="Path to training H5 file (required)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)

    # determinism toggle
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms (slower).")

    # EMA toggles
    ap.add_argument("--no-ema", action="store_true", help="Disable EMA.")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay.")

    return ap.parse_args()


if __name__ == "__main__":
    args = cli()
    cfg = Config()

    # overrides
    if args.train_h5 is not None:
        cfg.data.train_h5 = args.train_h5
    if args.train_csv is not None:
        cfg.data.train_csv = args.train_csv
    if args.batch is not None:
        cfg.loader.batch_size = args.batch
    if args.epochs is not None:
        cfg.train.max_epochs = args.epochs

    cfg.validate()

    run(cfg, args)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()