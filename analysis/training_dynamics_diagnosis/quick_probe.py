from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from ADCNN.core.config import Config
from ADCNN.main import run


def cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Small local training probe for rescue dynamics.")
    ap.add_argument("--target-mask-mode", choices=["hard", "soft"], required=True)
    ap.add_argument("--loss-mode", choices=["bce", "bce_ft", "bce_dice", "blend"], default="bce")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epoch-size", type=int, default=1024)
    ap.add_argument("--main-lr", type=float, default=1.0e-4)
    ap.add_argument("--warmup-lr", type=float, default=1.0e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--lam-max", type=float, default=0.0)
    ap.add_argument("--ramp-start", type=int, default=10)
    ap.add_argument("--ramp-end", type=int, default=18)
    ap.add_argument("--soft-mask-cache-dir", type=str, default="analysis/training_dynamics_diagnosis/cache")
    return ap.parse_args()


def main() -> None:
    args = cli()
    cfg = Config()
    cfg.data.train_h5 = "DATA/test.h5"
    cfg.data.train_csv = "DATA/test.csv"
    cfg.data.target_mask_mode = args.target_mask_mode
    if args.target_mask_mode == "soft":
        cfg.data.soft_mask_cache_dir = args.soft_mask_cache_dir
        Path(cfg.data.soft_mask_cache_dir).mkdir(parents=True, exist_ok=True)
    else:
        cfg.data.soft_mask_cache_dir = None
    cfg.loader.batch_size = int(args.batch)
    cfg.loader.num_workers = 2
    cfg.train.max_epochs = int(args.epochs)
    cfg.train.epoch_size = int(args.epoch_size)
    cfg.train.warmup_epochs = int(args.warmup_epochs)
    cfg.train.main_lr = float(args.main_lr)
    cfg.train.warmup_lr = float(args.warmup_lr)
    cfg.train.loss_mode = str(args.loss_mode)
    cfg.train.lam_max = float(args.lam_max)
    cfg.train.ramp_start_epoch = int(args.ramp_start)
    cfg.train.ramp_end_epoch = int(args.ramp_end)
    cfg.train.use_ema = False
    cfg.train.ema_eval = False
    cfg.train.ema_decay = 0.999
    cfg.train.rescue_val_max_images = 4
    cfg.train.rescue_val_every = 1
    cfg.train.rescue_val_every_early = 1
    cfg.train.rescue_val_early_epochs = int(args.epochs)
    cfg.train.auc_batches = 4
    cfg.train.save_last_to = f"analysis/training_dynamics_diagnosis/{args.target_mask_mode}_{args.loss_mode}_ckpt_last.pt"
    cfg.validate()

    ns = SimpleNamespace(
        deterministic=False,
        resume_epoch=None,
        no_ema=True,
        ema_decay=None,
    )
    run(cfg, ns)


if __name__ == "__main__":
    main()
