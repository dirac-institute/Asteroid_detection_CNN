import copy
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from ADCNN.utils.dist_utils import init_distributed, is_main_process
from ADCNN.evaluation.metrics import (
    resize_masks_to,
    masked_pixel_auc_agg,
    valid_mask_from_real,
)
from ADCNN.training.ema import EMAModel
from ADCNN.phases import maybe_init_head_bias_to_prior

# =============================================================================
# Small utils
# =============================================================================


def ensure_parent_dir(path: str) -> None:
    if not path:
        return
    p = Path(path)
    if p.parent and str(p.parent) not in (".", ""):
        p.parent.mkdir(parents=True, exist_ok=True)



def _ddp_allreduce_inplace(t: torch.Tensor) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


# =============================================================================
# Losses
# =============================================================================


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    pos_weight_t: torch.Tensor,
) -> torch.Tensor:
    """
    pos_weight_t must already be a tensor on correct device/dtype.
    Recommended: keep pos_weight_t float32 on device (no per-batch .to()).
    """
    logits_f = logits.float()
    t = targets.float().clamp(0.0, 1.0)
    v = valid.float()

    pw = pos_weight_t.float()
    loss_map = F.binary_cross_entropy_with_logits(
        logits_f, t, pos_weight=pw, reduction="none"
    )
    num = (loss_map * v).sum()
    den = v.sum().clamp_min(1.0)
    return num / den


def focal_tversky_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits_f = logits.float()
    t = targets.float().clamp(0.0, 1.0)
    v = valid.float()

    p = torch.sigmoid(logits_f).clamp(eps, 1.0 - eps)
    w = v

    TP = (w * p * t).sum(dim=(1, 2, 3))
    FP = (w * p * (1.0 - t)).sum(dim=(1, 2, 3))
    FN = (w * (1.0 - p) * t).sum(dim=(1, 2, 3))

    denom = TP + float(alpha) * FP + (1.0 - float(alpha)) * FN + eps
    tv = (TP + eps) / denom
    tv = tv.clamp(0.0, 1.0)

    loss = torch.pow(1.0 - tv, float(gamma))
    return loss.mean()


def ramp_lambda(ep: int, *, kind: str, e0: int, e1: int, k: float, max_value: float = 1.0) -> float:
    """
    Blend coefficient for (1-lam)*BCE + lam*FocalTversky
    """
    if e1 <= e0:
        return float(max_value) if ep >= e0 else 0.0
    if kind == "linear":
        x = (ep - e0) / float(e1 - e0)
        return float(max_value) * float(np.clip(x, 0.0, 1.0))
    if kind == "sigmoid":
        x = np.clip((ep - e0) / float(e1 - e0), 0.0, 1.0)
        z = (x * 2.0 - 1.0) * float(k)
        return float(max_value) * float(1.0 / (1.0 + np.exp(-z)))
    raise ValueError(kind)



# =============================================================================
# Trainer
# =============================================================================


class Trainer:
    def __init__(self, device: Optional[torch.device] = None, use_amp: bool = True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(use_amp)
        self._posw_cache: Dict[Tuple[str, str, float], torch.Tensor] = {}

    def _posw(self, *, dtype: torch.dtype, value: float) -> torch.Tensor:
        key = (str(self.device), str(dtype), float(value))
        if key not in self._posw_cache:
            self._posw_cache[key] = torch.tensor(float(value), device=self.device, dtype=dtype)
        return self._posw_cache[key]

    def _set_loader_epoch(self, loader, epoch: int) -> None:
        s = getattr(loader, "sampler", None)
        if s is not None and hasattr(s, "set_epoch"):
            s.set_epoch(int(epoch))

    @staticmethod
    def _current_lr(opt: torch.optim.Optimizer) -> float:
        if not opt.param_groups:
            return float("nan")
        return float(opt.param_groups[0]["lr"])

    def train_full_probe(
        self,
        model,
        train_loader,
        val_loader,
        *,
        seed: int = 1337,
        init_head_prior: float = 0.70,

        warmup_epochs: int = 0,
        warmup_batches: int = 0,
        warmup_lr: float = 2e-4,
        warmup_pos_weight: float = 12.0,

        # Kept as no-op compatibility knobs for older helper scripts.
        head_epochs: int = 0,
        head_batches: int = 0,
        head_lr: float = 0.0,
        head_pos_weight: float = 0.0,
        tail_epochs: int = 0,
        tail_batches: int = 0,
        tail_lr: float = 0.0,
        tail_pos_weight: float = 0.0,
        base_lrs=None,
        long_batches: int = 0,
        bce_pos_weight_long: Optional[float] = None,

        max_epochs: int = 60,
        val_every: int = 3,
        main_lr: float = 3e-4,
        min_lr_ratio: float = 0.10,
        lr_schedule: str = "cosine",
        weight_decay: float = 1e-4,

        loss_mode: str = "blend",
        ramp_kind: str = "linear",
        ramp_start_epoch: int = 4,
        ramp_end_epoch: int = 20,
        sigmoid_k: float = 8.0,
        lam_max: float = 0.70,

        bce_pos_weight_main: float = 8.0,
        ft_alpha: float = 0.45,
        ft_gamma: float = 1.3,

        train_real_label_mode: str = "downweight",
        train_real_label_weight: float = 0.25,
        val_real_label_mode: str = "ignore",
        val_real_label_weight: float = 0.0,

        fixed_thr: float = 0.5,

        auc_batches: int = 12,
        auc_bins: int = 256,

        main_batches: int = 0,

        resume_epoch: Optional[int] = None,

        save_best_to: Optional[str] = None,
        save_last_to: str = "ckpt_last.pt",

        verbose: int = 2,

        # metadata (pass from config)
        expected_tile: Optional[int] = None,
        model_hparams: Optional[Dict[str, Any]] = None,
        norm_name: Optional[str] = None,

        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_eval: bool = True,
        rescue_validator=None,
        enable_rescue_val: bool = False,
        rescue_val_every: int = 3,
        rescue_val_every_early: int = 1,
        rescue_val_early_epochs: int = 8,
    ):
        is_dist, rank, local_rank, world_size = init_distributed()
        use_ddp = bool(is_dist and world_size > 1)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.to(self.device)
        if use_ddp and not isinstance(model, DDP):
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        raw_model = model.module if isinstance(model, DDP) else model
        scaler = amp.GradScaler("cuda", enabled=self.amp)

        maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        # EMA tracks raw_model
        print ("EMA enabled:", use_ema)
        ema: Optional[EMAModel] = None
        if use_ema:
            ema = EMAModel(raw_model, decay=float(ema_decay), device="cpu", use_fp32=True)

        # -------------------------
        # Optional full resume (model + optimizer + scheduler + scaler + ema)
        # -------------------------
        start_epoch = 1
        train_phase = "warmup" if int(warmup_epochs) > 0 else "main"
        val_auc_last = float("nan")
        optimizer = torch.optim.AdamW(
            [p for p in raw_model.parameters() if p.requires_grad],
            lr=float(warmup_lr if int(warmup_epochs) > 0 else main_lr),
            weight_decay=float(weight_decay),
        )
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        if resume_epoch is not None:
            ckpt_path = Path(save_last_to) if save_last_to else None
            if ckpt_path is None or not ckpt_path.exists():
                raise FileNotFoundError(f"resume_epoch set but save_last_to not found: {save_last_to}")

            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            if "state" not in ckpt:
                raise KeyError(f"Checkpoint missing 'state': {ckpt_path}")

            # strict-ish validation (but parameterized, no hardcodes)
            md = ckpt.get("model_metadata", {})
            if md:
                ckpt_model_name = md.get("model_name")
                if ckpt_model_name and ckpt_model_name != raw_model.__class__.__name__:
                    raise ValueError(
                        f"Model mismatch: checkpoint has '{ckpt_model_name}', current is '{raw_model.__class__.__name__}'"
                    )
                if expected_tile is not None:
                    ckpt_tile = md.get("tile")
                    if ckpt_tile is not None and int(ckpt_tile) != int(expected_tile):
                        raise ValueError(f"Tile mismatch: checkpoint tile={ckpt_tile}, expected_tile={expected_tile}")
                if model_hparams is not None:
                    ckpt_hp = md.get("model_hparams")
                    if ckpt_hp is not None and dict(ckpt_hp) != dict(model_hparams):
                        raise ValueError(f"Model hparams mismatch: ckpt={ckpt_hp} expected={model_hparams}")
                if norm_name is not None:
                    ckpt_norm = md.get("norm")
                    if ckpt_norm is not None and str(ckpt_norm) != str(norm_name):
                        raise ValueError(f"Norm mismatch: ckpt={ckpt_norm} expected={norm_name}")

            raw_model.load_state_dict(ckpt["state"], strict=True)

            if ckpt.get("scaler") is not None:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass

            train_phase = str(ckpt.get("train_phase", train_phase))
            val_auc_last = float(ckpt.get("val_auc", float("nan")))

            if ckpt.get("ep") is not None:
                start_epoch = int(ckpt["ep"]) + 1
            else:
                start_epoch = int(resume_epoch) + 1

            if ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])

            if train_phase == "main" and str(lr_schedule) == "cosine":
                t_max = max(1, int(max_epochs) - int(warmup_epochs))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=t_max,
                    eta_min=float(main_lr) * float(min_lr_ratio),
                )
                if ckpt.get("scheduler") is not None:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler"])
                    except Exception:
                        scheduler = None

            if ema is not None and ckpt.get("ema") is not None:
                try:
                    ema.load_state_dict(ckpt["ema"])
                except Exception:
                    pass

            if verbose >= 1 and is_main_process():
                print(
                    f"[RESUME] loaded {ckpt_path} -> start_epoch={start_epoch} phase={train_phase} val_auc={val_auc_last}"
                )

        nan_checks = False # set True to enable NaN checks (expensive, only for debugging)
        for ep in range(int(start_epoch), int(max_epochs) + 1):
            t0 = time.time()
            self._set_loader_epoch(train_loader, seed + 1000 + ep)
            raw_model.train()
            epoch_phase = "warmup" if ep <= int(warmup_epochs) else "main"

            if epoch_phase == "warmup":
                train_phase = "warmup"
                for pg in optimizer.param_groups:
                    pg["lr"] = float(warmup_lr)
            else:
                if train_phase != "main":
                    train_phase = "main"
                    current_lr = min(self._current_lr(optimizer), float(main_lr))
                    for pg in optimizer.param_groups:
                        pg["lr"] = float(current_lr)
                    if str(lr_schedule) == "cosine":
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=max(1, int(max_epochs) - int(warmup_epochs)),
                            eta_min=float(current_lr) * float(min_lr_ratio),
                        )

            main_ep = max(0, ep - int(warmup_epochs))
            if epoch_phase == "warmup" or str(loss_mode) == "bce":
                lam = 0.0
            else:
                lam = ramp_lambda(
                    main_ep,
                    kind=str(ramp_kind),
                    e0=int(ramp_start_epoch),
                    e1=int(ramp_end_epoch),
                    k=float(sigmoid_k),
                    max_value=float(lam_max),
                )

            posw = self._posw(
                dtype=torch.float32,
                value=float(warmup_pos_weight if epoch_phase == "warmup" else bce_pos_weight_main),
            )

            loss_sum_t = torch.tensor(0.0, device=self.device)
            seen_t = torch.tensor(0, device=self.device)

            for i, batch in enumerate(train_loader, 1):
                xb, yb, rb, _missed, _detected = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                # NaN check 1/3
                if nan_checks:
                    if i == 2 or (i % 100 == 0):
                        for name, p in raw_model.named_parameters():
                            if not torch.isfinite(p).all():
                                raise RuntimeError(f"Non-finite PARAM before forward at ep{ep} it{i}: {name}")

                with amp.autocast("cuda", enabled=self.amp, dtype=torch.bfloat16):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = valid_mask_from_real(
                        rb_r,
                        mode=str(train_real_label_mode),
                        real_weight=float(train_real_label_weight),
                    )

                    loss_bce = masked_bce_with_logits(logits, yb_r, valid, pos_weight_t=posw)
                    if float(lam) <= 0.0:
                        loss = loss_bce
                    else:
                        loss_ft = focal_tversky_masked(
                            logits, yb_r, valid, alpha=float(ft_alpha), gamma=float(ft_gamma)
                        )
                        loss = (1.0 - float(lam)) * loss_bce + float(lam) * loss_ft

                    if not torch.isfinite(loss):
                        # dump batch stats
                        xb_f = xb.float()
                        logits_f = logits.float()
                        print("xb:", xb_f.min().item(), xb_f.max().item(), torch.isnan(xb_f).any().item(),
                              torch.isinf(xb_f).any().item())
                        print("logits:", logits_f.min().item(), logits_f.max().item(),
                              torch.isnan(logits_f).any().item(), torch.isinf(logits_f).any().item())
                        print("yb:", yb.float().min().item(), yb.float().max().item(),
                              torch.isnan(yb.float()).any().item())
                        print("rb:", rb.float().min().item(), rb.float().max().item(),
                              torch.isnan(rb.float()).any().item())
                        raise RuntimeError(f"Non-finite loss at long epoch {ep} iter {i}: {loss.item()}")

                    if verbose >= 3 and is_main_process():
                        print(f"[TRAIN ep{ep} iter {i}] loss {loss.item():.4f} | lam {lam:.3f}")

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # NaN check 2/3
                if nan_checks:
                    if i == 2 or (i % 100 == 0):
                        bad = None
                        for name, p in raw_model.named_parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                bad = name
                                break
                        if bad is not None:
                            if is_main_process():
                                print(f"[ep{ep} it{i}] non-finite grad in {bad} -> skipping step")
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            continue


                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # NaN check 3/3
                if nan_checks:
                    if i == 2 or (i % 100 == 0):
                        for name, p in raw_model.named_parameters():
                            if not torch.isfinite(p).all():
                                raise RuntimeError(f"Non-finite PARAM after step at ep{ep} it{i}: {name}")

                # EMA update
                if ema is not None:
                    ema.update(raw_model)

                loss_sum_t += loss.detach() * xb.size(0)
                seen_t += xb.size(0)

                limit_batches = int(warmup_batches) if epoch_phase == "warmup" else int(main_batches)
                if limit_batches > 0 and i >= limit_batches:
                    break

            train_loss = float((loss_sum_t / seen_t.clamp_min(1)).item())
            lr_actual = self._current_lr(optimizer)

            if verbose >= 2 and is_main_process():
                dt = time.time() - t0
                print(
                    f"[EP{ep:03d}] phase {epoch_phase.upper()} | loss {train_loss:.4f} | "
                    f"lam {lam:.3f} | lr {lr_actual:.2e} | {dt:.1f}s"
                )

            # Validation
            do_val = (ep % int(val_every) == 0) or (ep <= 3)
            auc_eval = None
            rescue_eval = None

            if do_val:
                self._set_loader_epoch(val_loader, seed + 2000 + ep)

                # evaluate RAW or EMA
                if ema is not None and ema_eval:
                    ema.apply_to(raw_model)
                    try:
                        auc_eval = masked_pixel_auc_agg(
                            model,
                            val_loader,
                            device=self.device,
                            n_bins=int(auc_bins),
                            max_batches=int(auc_batches),
                            real_label_mode=str(val_real_label_mode),
                            real_label_weight=float(val_real_label_weight),
                        )
                    finally:
                        ema.restore(raw_model)
                else:
                    auc_eval = masked_pixel_auc_agg(
                        model,
                        val_loader,
                        device=self.device,
                        n_bins=int(auc_bins),
                        max_batches=int(auc_batches),
                        real_label_mode=str(val_real_label_mode),
                        real_label_weight=float(val_real_label_weight),
                    )

                if verbose >= 1 and is_main_process():
                    tag = "EMA" if (ema is not None and ema_eval) else "RAW"
                    print(f"[VAL ep{ep:03d}] {tag} AUC {float(auc_eval):.4f}")
                val_auc_last = float(auc_eval)

            rescue_every = int(rescue_val_every_early) if ep <= int(rescue_val_early_epochs) else int(rescue_val_every)
            do_rescue_val = bool(enable_rescue_val and rescue_validator is not None) and (
                (ep % max(1, rescue_every) == 0) or (ep == 1)
            )
            if do_rescue_val:
                rescue_model = raw_model
                if ema is not None and ema_eval:
                    ema.apply_to(raw_model)
                    try:
                        rescue_eval = rescue_validator.evaluate(rescue_model, device=self.device)
                    finally:
                        ema.restore(raw_model)
                else:
                    rescue_eval = rescue_validator.evaluate(rescue_model, device=self.device)

                if verbose >= 1 and is_main_process():
                    primary = rescue_eval["primary"]
                    msg = (
                        f"[VAL ep{ep:03d}] rescue@{primary.budget} | newTP {primary.new_tp} | "
                        f"missedRecall {primary.missed_recall:.3f} | unionRecall {primary.union_recall:.3f} | "
                        f"addedFP {primary.added_fp} | nCand {primary.n_candidates}"
                    )
                    if auc_eval is not None:
                        msg += f" | AUC {float(auc_eval):.3f}"
                    msg += (
                        f" | subset {rescue_eval['subset_images']} img / {rescue_eval['subset_objects']} obj"
                        f" | {rescue_eval['elapsed_s']:.1f}s"
                    )
                    print(msg)
                    secondary = rescue_eval.get("secondary")
                    if secondary is not None:
                        print(
                            f"[VAL ep{ep:03d}] rescue@{secondary.budget} | newTP {secondary.new_tp} | "
                            f"missedRecall {secondary.missed_recall:.3f} | unionRecall {secondary.union_recall:.3f} | "
                            f"addedFP {secondary.added_fp}"
                        )

            if epoch_phase == "main" and scheduler is not None:
                scheduler.step()

            # Save last (full resume)
            if is_main_process() and save_last_to:
                ensure_parent_dir(save_last_to)
                torch.save(
                    {
                        "state": raw_model.state_dict(),
                        "ep": int(ep),
                        "train_phase": str(train_phase),
                        "train_loss": float(train_loss),
                        "lam": float(lam),
                        "lr": float(lr_actual),
                        "val_auc": float(auc_eval) if auc_eval is not None else float(val_auc_last),
                        "rescue_val": copy.deepcopy(rescue_eval) if rescue_eval is not None else None,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "ema": ema.state_dict() if ema is not None else None,
                        "model_metadata": {
                            "model_name": raw_model.__class__.__name__,
                            "model_hparams": dict(model_hparams) if model_hparams is not None else None,
                            "tile": int(expected_tile) if expected_tile is not None else None,
                            "norm": str(norm_name) if norm_name is not None else None,
                        },
                    },
                    save_last_to,
                )

        summary = {
            "last_ep": int(max_epochs),
            "last_val_auc": float(val_auc_last),
            "ema_enabled": bool(use_ema),
            "ema_decay": float(ema_decay),
            "ema_eval": bool(ema_eval),
            "primary_checkpoint": str(save_last_to),
        }

        if is_main_process() and verbose >= 1:
            print("=== DONE ===")
            print("Last summary:", summary)

        trained_model = raw_model
        return trained_model, float(fixed_thr), summary
