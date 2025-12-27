import torch, numpy as np
from config import Config
from utils.utils import set_seed, split_indices
from data.h5tiles import H5TiledDataset, SubsetDS, panels_with_positives
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.dist_utils import init_distributed, is_main_process
from models.unet_res_se import UNetResSEASPP
from train import Trainer
import argparse


def run(cfg: Config):
    is_dist, rank, local_rank, world_size = init_distributed()
    import builtins
    if not is_main_process():
        builtins.print = lambda *a, **k: None
    set_seed(cfg.train.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # splits
    idx_tr, idx_va = split_indices(cfg.data.train_h5, val_frac=0.1, seed=cfg.train.seed)

    # dataset + positive filtering (as before)
    ds_full = H5TiledDataset(cfg.data.train_h5, tile=cfg.data.tile, k_sigma=5.0)
    pos_panels = panels_with_positives(cfg.data.train_h5, max_panels=2000)
    rng = np.random.default_rng(cfg.train.seed)
    sub_tr = rng.choice(np.intersect1d(idx_tr, pos_panels), size=min(200, len(pos_panels)), replace=False)
    sub_va = np.random.default_rng(cfg.train.seed+1).choice(np.intersect1d(idx_va, pos_panels), size=min(80, len(pos_panels)), replace=False)

    train_ds = SubsetDS(ds_full, idx_tr)
    val_ds = SubsetDS(ds_full, idx_va)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.loader.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.loader.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory, persistent_workers=True, prefetch_factor=2,
    )

    # model
    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)

    # train
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
        thr_beta=cfg.train.thr_beta,
        thr_pos_rate_early=cfg.train.thr_pos_rate_early, thr_pos_rate_late=cfg.train.thr_pos_rate_late,
        save_best_to=cfg.train.save_best_to, save_last_to=cfg.train.save_last_to
    )
    if is_main_process():
        print("Final threshold:", thr)
        print("Summary:", summary)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_h5", type=str, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = cli()
    cfg = Config()
    if args.train_h5: cfg.data.train_h5 = args.train_h5
    if args.batch: cfg.loader.batch_size = args.batch
    if args.epochs: cfg.train.max_epochs = args.epochs
    run(cfg)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
