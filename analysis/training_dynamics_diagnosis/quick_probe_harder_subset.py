from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ADCNN.core.config import Config
from ADCNN.core.model import UNetResSEASPP
from ADCNN.data.datasets import H5TiledDataset
from ADCNN.main import (
    TileSubsetWithRealAndFlags,
    DistributedMixtureSampler,
    build_tile_mask_from_csv,
    filter_tiles_by_panels,
)
from ADCNN.train import Trainer
from ADCNN.training.rescue_validation import RescueValidator
from ADCNN.utils.helpers import worker_init_fn, split_indices
from analysis.training_dynamics_diagnosis.rescue_subset_utils import (
    select_harder_val_panels,
    save_panel_subset,
)


def cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Short local single-GPU training probe on a harder train-derived rescue subset.")
    ap.add_argument("--train-h5", type=str, default="DATA/train.h5")
    ap.add_argument("--train-csv", type=str, default="DATA/train.csv")
    ap.add_argument("--target-mask-mode", choices=["hard", "soft"], required=True)
    ap.add_argument("--loss-mode", choices=["bce", "bce_dice", "asl", "bce_ft", "blend"], default="bce")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--epoch-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--main-lr", type=float, default=1.0e-4)
    ap.add_argument("--warmup-lr", type=float, default=1.0e-4)
    ap.add_argument("--lam-max", type=float, default=0.0)
    ap.add_argument("--ramp-start", type=int, default=10)
    ap.add_argument("--ramp-end", type=int, default=18)
    ap.add_argument("--soft-target-gain", type=float, default=1.0)
    ap.add_argument("--soft-mask-cache-dir", type=str, default="analysis/training_dynamics_diagnosis/cache_probe")
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--subset-mode", choices=["missed_count", "missed_fraction"], default="missed_count")
    ap.add_argument("--budgets", type=str, default="50,200,1000,15000")
    ap.add_argument("--summary-path", type=str, default="analysis/training_dynamics_diagnosis/probe_frontier.jsonl")
    return ap.parse_args()


def main() -> None:
    args = cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())

    cfg = Config()
    cfg.data.train_h5 = args.train_h5
    cfg.data.train_csv = args.train_csv
    cfg.data.target_mask_mode = args.target_mask_mode
    cfg.data.soft_target_gain = float(args.soft_target_gain)
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
    cfg.train.rescue_budget_grid = budgets
    cfg.train.rescue_budget_primary = int(budgets[0])
    cfg.train.rescue_budget_secondary = int(budgets[-1])
    cfg.train.rescue_val_summary_path = str(args.summary_path)
    cfg.train.rescue_val_every = 1
    cfg.train.rescue_val_every_early = 1
    cfg.train.rescue_val_early_epochs = int(args.epochs)
    cfg.train.auc_batches = 4
    cfg.train.save_last_to = f"analysis/training_dynamics_diagnosis/{args.target_mask_mode}_{args.loss_mode}_probe_ckpt_last.pt"
    cfg.validate()

    base_ds = H5TiledDataset(
        cfg.data.train_h5,
        tile=cfg.data.tile,
        k_sigma=5.0,
        target_mask_mode=str(cfg.data.target_mask_mode),
        target_csv=str(cfg.data.train_csv) if cfg.data.target_mask_mode == "soft" else None,
        soft_mask_sigma_pix=float(cfg.data.soft_mask_sigma_pix),
        soft_mask_line_width=int(cfg.data.soft_mask_line_width),
        soft_mask_truncate=float(cfg.data.soft_mask_truncate),
        soft_mask_cache_dir=cfg.data.soft_mask_cache_dir,
        soft_mask_cache_size=int(cfg.data.soft_mask_cache_size),
        soft_mask_cache_dtype=str(cfg.data.soft_mask_cache_dtype),
        soft_target_gain=float(cfg.data.soft_target_gain),
    )

    idx_tr, idx_va = split_indices(cfg.data.train_h5, val_frac=float(cfg.data.val_frac), seed=cfg.train.seed)
    idx_tr_set = set(map(int, idx_tr.tolist()))
    idx_va_set = set(map(int, idx_va.tolist()))
    tiles_tr = filter_tiles_by_panels(base_ds, idx_tr_set)
    tiles_va = filter_tiles_by_panels(base_ds, idx_va_set)

    touched_any_base = build_tile_mask_from_csv(cfg.data.train_csv, base_ds, tile=int(cfg.data.tile), margin_pix=float(cfg.data.margin_pix), stack_col=str(cfg.data.stack_col), stack_value=None)
    missed_base = build_tile_mask_from_csv(cfg.data.train_csv, base_ds, tile=int(cfg.data.tile), margin_pix=float(cfg.data.margin_pix), stack_col=str(cfg.data.stack_col), stack_value=0)
    detected_base = build_tile_mask_from_csv(cfg.data.train_csv, base_ds, tile=int(cfg.data.tile), margin_pix=float(cfg.data.margin_pix), stack_col=str(cfg.data.stack_col), stack_value=1)

    train_ds = TileSubsetWithRealAndFlags(
        base_ds, tiles_tr,
        train_h5=cfg.data.train_h5,
        real_labels_key=str(cfg.data.real_labels_key),
        missed_mask_base=missed_base,
        detected_mask_base=detected_base,
    )
    val_ds = TileSubsetWithRealAndFlags(
        base_ds, tiles_va,
        train_h5=cfg.data.train_h5,
        real_labels_key=str(cfg.data.real_labels_key),
        missed_mask_base=missed_base,
        detected_mask_base=detected_base,
    )

    missed_local = []
    detected_local = []
    background_local = []
    base_indices = train_ds.base_tile_indices
    for j in range(len(train_ds)):
        bidx = int(base_indices[j])
        if bool(missed_base[bidx]):
            missed_local.append(j)
        elif bool(detected_base[bidx]):
            detected_local.append(j)
        elif bool(touched_any_base[bidx]):
            detected_local.append(j)
        else:
            background_local.append(j)
    sampler = DistributedMixtureSampler(
        dataset_size=len(train_ds),
        missed_ids=np.asarray(missed_local, dtype=np.int64),
        detected_ids=np.asarray(detected_local, dtype=np.int64),
        background_ids=np.asarray(background_local, dtype=np.int64),
        num_replicas=1,
        rank=0,
        seed=int(cfg.train.seed),
        frac_missed=float(cfg.train.frac_missed),
        frac_detected=float(cfg.train.frac_detected),
        frac_background=float(cfg.train.frac_background),
        epoch_size=int(cfg.train.epoch_size),
    )
    def _wif(worker_id: int):
        return worker_init_fn(worker_id, base_seed=int(cfg.train.seed))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.loader.batch_size), shuffle=False, sampler=sampler, num_workers=cfg.loader.num_workers, pin_memory=cfg.loader.pin_memory, persistent_workers=(cfg.loader.num_workers > 0), prefetch_factor=2 if cfg.loader.num_workers > 0 else None, worker_init_fn=_wif if cfg.loader.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.loader.batch_size), shuffle=False, num_workers=cfg.loader.num_workers, pin_memory=cfg.loader.pin_memory, persistent_workers=(cfg.loader.num_workers > 0), prefetch_factor=2 if cfg.loader.num_workers > 0 else None, worker_init_fn=_wif if cfg.loader.num_workers > 0 else None)

    harder_panels = select_harder_val_panels(
        train_h5=cfg.data.train_h5,
        train_csv=cfg.data.train_csv,
        val_frac=float(cfg.data.val_frac),
        seed=int(cfg.train.seed),
        max_images=int(args.max_images),
        mode=str(args.subset_mode),
    )
    save_panel_subset(
        "analysis/training_dynamics_diagnosis/probe_subset.json",
        harder_panels,
        meta={"subset_mode": args.subset_mode, "budgets": budgets},
    )
    rescue_validator = RescueValidator(
        h5_path=cfg.data.train_h5,
        csv_path=str(cfg.data.train_csv),
        val_panel_ids=harder_panels,
        seed=int(cfg.train.seed),
        max_images=0,
        batch_size=max(1, int(cfg.loader.batch_size)),
        num_workers=0,
        rescue_budget_primary=int(budgets[0]),
        rescue_budget_secondary=int(budgets[-1]),
        rescue_budget_grid=budgets,
        rescue_overlap_policy=str(cfg.train.rescue_overlap_policy),
        psf_width=int(cfg.train.rescue_val_psf_width),
        threshold=float(cfg.train.rescue_post_threshold),
        pixel_gap=int(cfg.train.rescue_post_pixel_gap),
        min_area=int(cfg.train.rescue_post_min_area),
        max_area=cfg.train.rescue_post_max_area,
        min_score=float(cfg.train.rescue_post_min_score),
        min_peak_probability=float(cfg.train.rescue_post_min_peak_probability),
        score_method=str(cfg.train.rescue_post_score_method),
        topk_fraction=float(cfg.train.rescue_post_topk_fraction),
    )
    print("[probe-subset] panels:", ",".join(map(str, harder_panels)))

    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)
    trainer = Trainer(device=device)
    trainer.train_full_probe(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        seed=cfg.train.seed,
        init_head_prior=cfg.train.init_head_prior,
        warmup_epochs=cfg.train.warmup_epochs,
        warmup_batches=cfg.train.warmup_batches,
        warmup_lr=cfg.train.warmup_lr,
        warmup_pos_weight=cfg.train.warmup_pos_weight,
        max_epochs=cfg.train.max_epochs,
        val_every=cfg.train.val_every,
        main_lr=cfg.train.main_lr,
        min_lr_ratio=cfg.train.min_lr_ratio,
        lr_schedule=cfg.train.lr_schedule,
        weight_decay=cfg.train.weight_decay,
        loss_mode=cfg.train.loss_mode,
        ramp_kind=cfg.train.ramp_kind,
        ramp_start_epoch=cfg.train.ramp_start_epoch,
        ramp_end_epoch=cfg.train.ramp_end_epoch,
        sigmoid_k=cfg.train.sigmoid_k,
        lam_max=cfg.train.lam_max,
        bce_pos_weight_main=cfg.train.bce_pos_weight_main,
        ft_alpha=cfg.train.ft_alpha,
        ft_gamma=cfg.train.ft_gamma,
        asl_gamma_neg=cfg.train.asl_gamma_neg,
        asl_gamma_pos=cfg.train.asl_gamma_pos,
        asl_clip=cfg.train.asl_clip,
        train_real_label_mode=cfg.train.train_real_label_mode,
        train_real_label_weight=cfg.train.train_real_label_weight,
        val_real_label_mode=cfg.train.val_real_label_mode,
        val_real_label_weight=cfg.train.val_real_label_weight,
        fixed_thr=cfg.train.fixed_thr,
        auc_batches=cfg.train.auc_batches,
        main_batches=cfg.train.main_batches,
        save_best_to=None,
        save_last_to=cfg.train.save_last_to,
        verbose=cfg.train.verbose,
        expected_tile=int(cfg.data.tile),
        model_hparams={"in_ch": 1, "out_ch": 1},
        norm_name="medmad_clip_k5",
        use_ema=cfg.train.use_ema,
        ema_decay=cfg.train.ema_decay,
        ema_eval=cfg.train.ema_eval,
        rescue_validator=rescue_validator,
        enable_rescue_val=True,
        rescue_val_every=1,
        rescue_val_every_early=1,
        rescue_val_early_epochs=int(cfg.train.max_epochs),
        rescue_val_summary_path=str(cfg.train.rescue_val_summary_path),
    )


if __name__ == "__main__":
    main()
