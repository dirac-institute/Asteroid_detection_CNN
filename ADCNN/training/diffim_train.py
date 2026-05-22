"""Diffim NN training entry point (formerly experiments/diffim_pilot/v5_train.py).

The training shape that produced the canonical v7 weights:
  - 3-channel input (signed diffim, log1p local std, real_labels binary)
  - UNetResSEOrientHough model (UNet + orientation aux head + line aggregator)
  - Random-crop sampler over per-injection anchors (positive-anchored
    with `stk_balance` bias toward LSST-missed positives)
  - AFTL primary loss + small BCE anchor + masked orientation MSE
  - EMA with `agg_alpha` excluded (set --ema-exclude agg_alpha)

Run via the CLI:
    python -m ADCNN.training.diffim_train --run-name pilot_v7 \\
        --widths 24 48 96 192 384 ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ADCNN.training.ema import EMAModel
from ADCNN.data.diffim_dataset import (
    DiffimRandomCropDataset3ch,
    collate_v5,
)
from ADCNN.core.diffim_model import UNetResSEOrientHough

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Losses (inlined from the v4_train.py prototype during consolidation).
# ---------------------------------------------------------------------------
def masked_aftl_loss(
    seg_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.3,
    bce_anchor_weight: float = 0.1,
    eps: float = 1e-6,
):
    """Masked Asymmetric Focal Tversky + a small unweighted BCE anchor."""
    mask = 1.0 - ignore
    p = torch.sigmoid(seg_logits).clamp(eps, 1.0 - eps) * mask
    t = (targets * mask).clamp(0.0, 1.0)
    p_flat = p.view(p.size(0), -1)
    t_flat = t.view(t.size(0), -1)
    TP = (p_flat * t_flat).sum(dim=1)
    FP = ((1.0 - t_flat) * p_flat).sum(dim=1)
    FN = (t_flat * (1.0 - p_flat)).sum(dim=1)
    tv = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    aftl = torch.pow(1.0 - tv, gamma).mean()
    if bce_anchor_weight > 0:
        bce_full = F.binary_cross_entropy_with_logits(seg_logits, targets, reduction="none") * mask
        denom = mask.sum().clamp(min=1.0)
        bce = bce_full.sum() / denom
    else:
        bce = torch.tensor(0.0, device=seg_logits.device)
    return aftl + bce_anchor_weight * bce, aftl.detach(), bce.detach()


def masked_orient_mse(
    pred_sin: torch.Tensor,
    pred_cos: torch.Tensor,
    true_sin: torch.Tensor,
    true_cos: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    """MSE on the unit-circle (sin 2β, cos 2β) only where mask > 0.5."""
    m = mask
    err = (pred_sin - true_sin).pow(2) * m + (pred_cos - true_cos).pow(2) * m
    n = m.sum().clamp(min=1.0)
    return err.sum() / n


def pick_train_val_panels(n_total: int, n_train: int, n_val: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    chosen = perm[: n_train + n_val]
    val = sorted(chosen[:n_val].tolist())
    train = sorted(chosen[n_val:].tolist())
    return train, val


@torch.no_grad()
def validate(model, loader, device, *, lambda_orient: float):
    model.eval()
    seg_loss_sum, n = 0.0, 0
    orient_mse_sum = 0.0
    pos_p, neg_p = [], []
    for xb, yb, igb, ys, yc, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        igb = igb.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        yc = yc.to(device, non_blocking=True)
        seg_logits, p_sin, p_cos, _raw, _agg = model(xb)
        mask = 1.0 - igb
        bce_full = F.binary_cross_entropy_with_logits(seg_logits, yb, reduction="none") * mask
        denom = mask.sum().clamp(min=1.0)
        seg_loss_sum += float((bce_full.sum() / denom)) * xb.size(0)
        omse = masked_orient_mse(p_sin, p_cos, ys, yc, yb * mask)
        orient_mse_sum += float(omse) * xb.size(0)
        n += xb.size(0)
        p = torch.sigmoid(seg_logits)
        m = (mask > 0)
        yy = (yb > 0.5) & m
        nn = (yb <= 0.5) & m
        pp = p.cpu().numpy().ravel()
        yy = yy.cpu().numpy().ravel()
        nn = nn.cpu().numpy().ravel()
        if yy.any():
            pos_p.append(pp[yy])
        if nn.any():
            ng = pp[nn]
            if ng.size > 100_000:
                idx = np.random.choice(ng.size, size=100_000, replace=False)
                ng = ng[idx]
            neg_p.append(ng)

    out = {"val_seg_bce": seg_loss_sum / max(n, 1),
           "val_orient_mse": orient_mse_sum / max(n, 1)}
    if pos_p and neg_p:
        ps = np.concatenate(pos_p)
        ns = np.concatenate(neg_p)
        ns_sub = ns if ns.size <= 1_000_000 else np.random.choice(ns, size=1_000_000, replace=False)
        try:
            from sklearn.metrics import roc_auc_score
            y_true = np.concatenate([np.ones_like(ps), np.zeros_like(ns_sub)])
            y_score = np.concatenate([ps, ns_sub])
            out["val_pixel_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["val_pixel_auc"] = float("nan")
        out["val_pos_pixels"] = int(ps.size)
        out["val_neg_pixels"] = int(ns.size)
        out["val_pos_mean_p"] = float(ps.mean())
        out["val_neg_mean_p"] = float(ns.mean())
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-h5", default=str(REPO_ROOT / "DATA_DIFFIM" / "train.h5"))
    ap.add_argument("--data-csv", default=str(REPO_ROOT / "DATA_DIFFIM" / "train.csv"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "experiments" / "diffim_runs"))
    ap.add_argument("--n-train-panels", type=int, default=150)
    ap.add_argument("--n-val-panels", type=int, default=30)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=12,
                    help="Smaller than v4 (16) because the line aggregator adds memory.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--n-pos-anchors-per-epoch", type=int, default=1800)
    ap.add_argument("--n-neg-anchors-per-epoch", type=int, default=600)
    ap.add_argument("--stk-balance", type=float, default=0.6)
    ap.add_argument("--anchor-jitter", type=int, default=48)
    ap.add_argument("--aftl-alpha", type=float, default=0.3)
    ap.add_argument("--aftl-beta", type=float, default=0.7)
    ap.add_argument("--aftl-gamma", type=float, default=1.3)
    ap.add_argument("--aftl-bce-anchor", type=float, default=0.1)
    ap.add_argument("--lambda-orient", type=float, default=0.5)
    ap.add_argument("--kernel-lens", type=int, nargs="+", default=[11, 21, 41])
    ap.add_argument("--n-angles", type=int, default=12)
    ap.add_argument("--widths", type=int, nargs="+", default=[48, 96, 192, 384, 768],
                    help="UNet channel widths per level. Default is v5/v6. "
                         "Use [24, 48, 96, 192, 384] for v7-half (4x cheaper).")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--orient-cache-size", type=int, default=24,
                    help="Panels cached in the orientation-map cache (per worker).")
    ap.add_argument("--ema-exclude", nargs="*", default=[],
                    help="Parameter names to exclude from EMA tracking (e.g. agg_alpha).")
    ap.add_argument("--data-sources", nargs="*", default=None,
                    help="Train on multiple full-panel h5 datasets (no merge) via "
                         "DiffimConcatDataset. Each arg is 'h5path:csvpath'. Val uses "
                         "--data-h5/--data-csv (realistic val panels). For 'much more data'.")
    ap.add_argument("--stream-buffer", default="",
                    help="Train from a streaming rolling buffer (stream_producer.py) of "
                         "full panels -- lossless, same statistics, unbounded data. "
                         "Run the producer alongside. Val uses the fixed realistic val "
                         "panels (--data-h5/--data-csv) for a stable metric.")
    ap.add_argument("--tile-data-h5", default="",
                    help="Train from a storage-efficient tile corpus (tile_gen.py "
                         "tiles.h5) instead of full panels. Panel-disjoint train/val "
                         "split by panel_key. Bypasses --data-h5/--n-train-panels.")
    ap.add_argument("--tile-data-csv", default="")
    ap.add_argument("--intensity-aug", action="store_true",
                    help="Train-set intensity+noise augmentation (vary effective SNR/"
                         "background) on top of --augment. Data-like regularizer for the "
                         "faint-trail regime.")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Spatial dropout p (bottleneck + pre-head). 0 = off (default, "
                         "identical to before). ~0.1-0.2 regularizes the post-ep10 overfit.")
    ap.add_argument("--augment", action="store_true",
                    help="Enable D4 dihedral augmentation (flips + 90deg rotations "
                         "with sin2b/cos2b orientation-label transform) on the TRAIN "
                         "set. Trails are orientation-agnostic so this is label-"
                         "preserving; counters the no-augmentation overfitting.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-from", default="",
                    help="Trainable checkpoint (last.pt/best.pt) to "
                         "initialize model + EMA weights from before "
                         "fine-tuning. Fresh optimizer/scheduler — use a low "
                         "--lr and few --epochs. Architecture flags (--widths "
                         "--kernel-lens --n-angles) must match the checkpoint.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_dir = Path(args.out_root) / args.run_name
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    log(f"v5 config: {json.dumps(vars(args), indent=2)}")

    if args.data_sources:
        # train on MULTIPLE full-panel h5 datasets (no merge) via DiffimConcatDataset.
        # --data-sources "h5a:csva" "h5b:csvb" ... ; val from --data-h5/--data-csv val panels.
        from ADCNN.data.diffim_dataset import DiffimConcatDataset
        srcs = [tuple(s.split(":")) for s in args.data_sources]
        log(f"MULTI-SOURCE training on {len(srcs)} h5 datasets: {[s[0].split('/')[-2] for s in srcs]}")
        train_ds = DiffimConcatDataset(
            srcs, tile=args.tile,
            n_pos_anchors_per_epoch=args.n_pos_anchors_per_epoch,
            n_neg_anchors_per_epoch=args.n_neg_anchors_per_epoch,
            stk_balance=args.stk_balance, anchor_jitter=args.anchor_jitter,
            seed=args.seed, augment=args.augment, intensity_aug=bool(args.intensity_aug))
        import json as _json
        vp = sorted(_json.loads((REPO_ROOT / "experiments/diffim_runs/pilot_v7_realistic/split.json").read_text())["val_panels"])
        val_ds = DiffimRandomCropDataset3ch(
            args.data_h5, pd.read_csv(args.data_csv), panel_ids=vp, tile=args.tile,
            n_pos_anchors_per_epoch=500, n_neg_anchors_per_epoch=200,
            stk_balance=args.stk_balance, anchor_jitter=args.anchor_jitter,
            orient_cache_size=args.orient_cache_size, seed=args.seed + 1)
    elif args.stream_buffer:
        # LOSSLESS streaming: full panels from a rolling buffer (stream_producer.py),
        # identical anchor statistics. Producer must be running alongside.
        from ADCNN.data.diffim_dataset import DiffimStreamDataset
        log(f"STREAMING from buffer {args.stream_buffer}")
        train_ds = DiffimStreamDataset(
            args.stream_buffer, tile=args.tile,
            n_pos_per_epoch=args.n_pos_anchors_per_epoch,
            n_neg_per_epoch=args.n_neg_anchors_per_epoch,
            anchor_jitter=args.anchor_jitter, seed=args.seed, augment=args.augment)
        train_ds.intensity_aug = bool(args.intensity_aug)
        # held-out val from the FIXED realistic val panels (stable metric across epochs)
        csv_df = pd.read_csv(args.data_csv)
        import json as _json
        vp = sorted(_json.loads((REPO_ROOT / "experiments/diffim_runs/pilot_v7_realistic/split.json").read_text())["val_panels"])
        val_ds = DiffimRandomCropDataset3ch(
            args.data_h5, csv_df, panel_ids=vp, tile=args.tile,
            n_pos_anchors_per_epoch=500, n_neg_anchors_per_epoch=200,
            stk_balance=args.stk_balance, anchor_jitter=args.anchor_jitter,
            orient_cache_size=args.orient_cache_size, seed=args.seed + 1)
    elif args.tile_data_h5:
        # storage-efficient tile corpus (DiffimTileDataset); panel-disjoint train/val
        # split by panel_key so no leakage between them.
        from ADCNN.data.diffim_dataset import DiffimTileDataset
        meta = pd.read_csv(args.tile_data_csv)
        keys = sorted(meta["panel_key"].unique())
        rng = np.random.default_rng(args.seed); rng.shuffle(keys)
        n_val = max(1, int(round(0.05 * len(keys))))
        val_keys, train_keys = set(keys[:n_val]), set(keys[n_val:])
        log(f"TILE data: {len(keys)} panels -> train {len(train_keys)} / val {len(val_keys)}; "
            f"{len(meta)} tiles ({int((meta.is_pos==1).sum())} pos)")
        (run_dir / "split.json").write_text(json.dumps(
            {"tile_train_panels": sorted(train_keys), "tile_val_panels": sorted(val_keys)}, indent=2))
        train_ds = DiffimTileDataset(
            args.tile_data_h5, args.tile_data_csv, tile=args.tile,
            n_pos_per_epoch=args.n_pos_anchors_per_epoch,
            n_neg_per_epoch=args.n_neg_anchors_per_epoch,
            seed=args.seed, augment=args.augment, panel_keys=train_keys)
        train_ds.intensity_aug = bool(args.intensity_aug)
        val_ds = DiffimTileDataset(
            args.tile_data_h5, args.tile_data_csv, tile=args.tile,
            n_pos_per_epoch=500, n_neg_per_epoch=200, seed=args.seed + 1,
            augment=False, panel_keys=val_keys)
    else:
        log("opening HDF5…")
        with h5py.File(args.data_h5, "r") as f:
            n_total = int(f["images"].shape[0])
        train_panels, val_panels = pick_train_val_panels(n_total, args.n_train_panels, args.n_val_panels, seed=args.seed)
        log(f"train_panels={len(train_panels)} val_panels={len(val_panels)} / {n_total}")
        (run_dir / "split.json").write_text(json.dumps({
            "train_panels": train_panels, "val_panels": val_panels, "n_total": n_total,
        }, indent=2))

        csv_df = pd.read_csv(args.data_csv)
        train_ds = DiffimRandomCropDataset3ch(
            args.data_h5, csv_df, panel_ids=train_panels,
            tile=args.tile,
            n_pos_anchors_per_epoch=args.n_pos_anchors_per_epoch,
            n_neg_anchors_per_epoch=args.n_neg_anchors_per_epoch,
            stk_balance=args.stk_balance,
            anchor_jitter=args.anchor_jitter,
            orient_cache_size=args.orient_cache_size,
            seed=args.seed,
            augment=args.augment,
        )
        train_ds.intensity_aug = bool(args.intensity_aug)
        val_ds = DiffimRandomCropDataset3ch(
            args.data_h5, csv_df, panel_ids=val_panels,
            tile=args.tile,
            n_pos_anchors_per_epoch=500,
            n_neg_anchors_per_epoch=200,
            stk_balance=args.stk_balance,
            anchor_jitter=args.anchor_jitter,
            orient_cache_size=args.orient_cache_size,
            seed=args.seed + 1,
        )
    log(f"train ds size (anchors/epoch): {len(train_ds)}")
    log(f"val ds size (anchors/epoch):   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_v5,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, collate_fn=collate_v5,
        persistent_workers=(args.num_workers > 0),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNetResSEOrientHough(
        in_ch=3, widths=tuple(args.widths),
        kernel_lens=tuple(args.kernel_lens), n_angles=args.n_angles,
        p_drop=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model params: {n_params/1e6:.2f} M  device={device}")
    log(f"kernel_lens={args.kernel_lens} n_angles={args.n_angles}")

    _init_ck = None
    if args.init_from:
        _init_ck = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(_init_ck["model"])
        log(f"[init-from] loaded model weights from {args.init_from} "
            f"(orig epoch {_init_ck.get('epoch', '?')}) — fine-tune mode")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = EMAModel(model, decay=args.ema_decay, device="cpu")
    if _init_ck is not None and "ema" in _init_ck:
        try:
            ema.load_state_dict(_init_ck["ema"])
            log("[init-from] loaded EMA shadow from checkpoint")
        except Exception as e:
            log(f"[init-from] EMA load skipped ({e}); EMA seeded from weights")
    # Drop excluded names from EMA shadow + intercept update() to keep them out.
    if args.ema_exclude:
        for nm in args.ema_exclude:
            if nm in ema.shadow:
                ema.shadow.pop(nm)
                log(f"  ema: excluded '{nm}' from shadow")
        _orig_update = ema.update

        @torch.no_grad()
        def _filtered_update(m):
            # Save the model's parameter dict, then temporarily flip
            # requires_grad on excluded params so the base update() skips them.
            handles = []
            for nm in args.ema_exclude:
                for pname, p in m.named_parameters():
                    if pname == nm:
                        handles.append((p, p.requires_grad))
                        p.requires_grad_(False)
            try:
                _orig_update(m)
            finally:
                for p, was in handles:
                    p.requires_grad_(was)
        ema.update = _filtered_update

    best_metric = -1.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_seen = 0
        for step, (xb, yb, igb, ys, yc, _) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            igb = igb.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                seg_logits, p_sin, p_cos, _raw, _agg = model(xb)
                seg_loss, aftl, bce = masked_aftl_loss(
                    seg_logits, yb, igb,
                    alpha=args.aftl_alpha, beta=args.aftl_beta, gamma=args.aftl_gamma,
                    bce_anchor_weight=args.aftl_bce_anchor,
                )
                orient_mask = yb * (1.0 - igb)
                orient_loss = masked_orient_mse(p_sin, p_cos, ys, yc, orient_mask)
                loss = seg_loss + args.lambda_orient * orient_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            ema.update(model)
            epoch_loss += float(loss.detach()) * xb.size(0)
            n_seen += xb.size(0)
            if step % 50 == 0:
                log(f"  ep{epoch:02d} step{step:04d} loss={float(loss):.4f} "
                    f"aftl={float(aftl):.4f} bce={float(bce):.4f} orient={float(orient_loss):.4f} "
                    f"agg_alpha={float(model.agg_alpha):.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = epoch_loss / max(n_seen, 1)
        ema.apply_to(model)
        try:
            val_metrics = validate(model, val_loader, device, lambda_orient=args.lambda_orient)
        finally:
            ema.restore(model)
        ep_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_sec": time.time() - t0,
            "agg_alpha": float(model.agg_alpha),
            **val_metrics,
        }
        history.append(ep_summary)
        log(f"== ep{epoch:02d} train_loss={train_loss:.4f} "
            f"val_seg_bce={val_metrics.get('val_seg_bce', float('nan')):.4f} "
            f"val_auc={val_metrics.get('val_pixel_auc', float('nan')):.4f} "
            f"val_neg_mean_p={val_metrics.get('val_neg_mean_p', float('nan')):.3f} "
            f"agg_alpha={float(model.agg_alpha):.3f} "
            f"elapsed={time.time()-t0:.1f}s ==")

        torch.save({
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, run_dir / "ckpts" / "last.pt")

        metric = float(val_metrics.get("val_pixel_auc", float("nan")))
        if math.isfinite(metric) and metric > best_metric:
            best_metric = metric
            torch.save({
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "val_pixel_auc": metric,
                "args": vars(args),
            }, run_dir / "ckpts" / "best.pt")
            log(f"  >> best updated (val_pixel_auc={metric:.4f})")

        (run_dir / "metrics" / "history.json").write_text(json.dumps(history, indent=2))

    summary = {
        "best_val_pixel_auc": best_metric,
        "epochs_run": args.epochs,
        "final_agg_alpha": float(model.agg_alpha),
        "train_panels": len(train_panels),
        "val_panels": len(val_panels),
    }
    (run_dir / "metrics" / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE. {json.dumps(summary)}")


if __name__ == "__main__":
    main()
