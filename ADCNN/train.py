import copy
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP

from ADCNN.utils.dist_utils import init_distributed, is_main_process
from ADCNN.evaluation.metrics import (
    resize_masks_to,
    masked_pixel_auc,
    valid_mask_from_real,
)
from ADCNN.training.ema import EMAModel
from ADCNN.phases import (
    maybe_init_head_bias_to_prior,
    apply_phase,
    freeze_all,
    _unfreeze_if_exists,
)

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
    t = targets.float().clamp(0.0, 1.0)
    loss_map = F.binary_cross_entropy_with_logits(logits, t, pos_weight=pos_weight_t, reduction="none")
    num = (loss_map * valid).sum()
    den = valid.sum().clamp_min(1.0)
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


def ramp_lambda(ep: int, *, kind: str, e0: int, e1: int, k: float) -> float:
    """
    Blend coefficient for (1-lam)*BCE + lam*FocalTversky
    """
    if e1 <= e0:
        return 1.0 if ep >= e0 else 0.0
    if kind == "linear":
        x = (ep - e0) / float(e1 - e0)
        return float(np.clip(x, 0.0, 1.0))
    if kind == "sigmoid":
        x = np.clip((ep - e0) / float(e1 - e0), 0.0, 1.0)
        z = (x * 2.0 - 1.0) * float(k)
        return float(1.0 / (1.0 + np.exp(-z)))
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

    def _sync_optimizer_params(self, opt: torch.optim.Optimizer, model: torch.nn.Module) -> None:
        tracked = set()
        for g in opt.param_groups:
            for p in g["params"]:
                tracked.add(id(p))

        new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in tracked]
        if new_params:
            base = opt.param_groups[0]
            opt.add_param_group(
                {
                    "params": new_params,
                    "lr": base.get("lr", 1e-4),
                    "weight_decay": base.get("weight_decay", 0.0),
                    "betas": base.get("betas", (0.9, 0.999)),
                    "eps": base.get("eps", 1e-8),
                }
            )

    def _stage_lr(self, ep: int, base_lrs: Tuple[float, float, float]) -> float:
        if ep <= 12:
            return float(base_lrs[0])
        if ep <= 25:
            return float(base_lrs[1])
        return float(base_lrs[2])

    def train_full_probe(
        self,
        model,
        train_loader,
        val_loader,
        *,
        seed: int = 1337,
        init_head_prior: float = 0.70,

        warmup_epochs: int = 1,
        warmup_batches: int = 800,
        warmup_lr: float = 2e-4,
        warmup_pos_weight: float = 40.0,

        head_epochs: int = 2,
        head_batches: int = 2000,
        head_lr: float = 3e-5,
        head_pos_weight: float = 5.0,

        tail_epochs: int = 2,
        tail_batches: int = 2500,
        tail_lr: float = 1.5e-4,
        tail_pos_weight: float = 2.0,

        max_epochs: int = 60,
        val_every: int = 3,
        base_lrs: Tuple[float, float, float] = (3e-4, 2e-4, 1e-4),
        weight_decay: float = 1e-4,

        ramp_kind: str = "linear",
        ramp_start_epoch: int = 11,
        ramp_end_epoch: int = 40,
        sigmoid_k: float = 8.0,

        bce_pos_weight_long: float = 8.0,
        ft_alpha: float = 0.45,
        ft_gamma: float = 1.3,

        fixed_thr: float = 0.5,

        auc_batches: int = 12,
        auc_bins: int = 256,

        long_batches: int = 0,

        resume_epoch: Optional[int] = None,

        save_best_to: str = "ckpt_best.pt",
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
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )

        raw_model = model.module if isinstance(model, DDP) else model
        scaler = amp.GradScaler("cuda", enabled=self.amp)

        maybe_init_head_bias_to_prior(raw_model, float(init_head_prior))

        # EMA tracks raw_model
        ema: Optional[EMAModel] = None
        if use_ema:
            ema = EMAModel(raw_model, decay=float(ema_decay), device="cpu", use_fp32=True)

        # -------------------------
        # Optional full resume (model + optimizer + scheduler + scaler + ema)
        # -------------------------
        start_epoch = 1
        best_auc = -1.0
        best_ep = 0
        best_state = None
        best_state_ema = None

        long_opt: Optional[torch.optim.Optimizer] = None
        long_sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        long_stage_lr: Optional[float] = None

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

            best_auc = float(ckpt.get("best_auc", -1.0))
            best_ep = int(ckpt.get("best_ep", 0))
            best_state = ckpt.get("best_state", None)
            best_state_ema = ckpt.get("best_state_ema", None)

            if ckpt.get("ep") is not None:
                start_epoch = int(ckpt["ep"]) + 1
            else:
                start_epoch = int(resume_epoch) + 1

            if ckpt.get("long_opt") is not None:
                # placeholder; will be synced after apply_phase in first long epoch
                long_opt = torch.optim.Adam(
                    [p for p in raw_model.parameters() if p.requires_grad],
                    lr=1e-4,
                    weight_decay=float(weight_decay),
                )
                long_opt.load_state_dict(ckpt["long_opt"])

            if ckpt.get("long_sched") is not None and long_opt is not None:
                long_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    long_opt, T_0=6, T_mult=2, eta_min=1e-5
                )
                try:
                    long_sched.load_state_dict(ckpt["long_sched"])
                except Exception:
                    long_sched = None

            long_stage_lr = ckpt.get("long_stage_lr", None)
            if long_stage_lr is not None:
                long_stage_lr = float(long_stage_lr)

            if ema is not None and ckpt.get("ema") is not None:
                try:
                    ema.load_state_dict(ckpt["ema"])
                except Exception:
                    pass

            if verbose >= 1 and is_main_process():
                print(
                    f"[RESUME] loaded {ckpt_path} -> start_epoch={start_epoch} best_auc={best_auc:.4f} best_ep={best_ep}"
                )

        # -------------------------
        # Warmup / Head / Tail (only if not resuming)
        # -------------------------
        if resume_epoch is None:
            # Warmup
            freeze_all(raw_model)
            for p in raw_model.parameters():
                p.requires_grad = True

            opt = torch.optim.Adam(
                [p for p in raw_model.parameters() if p.requires_grad],
                lr=float(warmup_lr),
                weight_decay=0.0,
            )

            posw_warm = self._posw(dtype=torch.float32, value=float(warmup_pos_weight))

            for ep in range(1, int(warmup_epochs) + 1):
                t0 = time.time()
                self._set_loader_epoch(train_loader, seed + 100 + ep)
                model.train()

                loss_sum_t = torch.tensor(0.0, device=self.device)
                seen_t = torch.tensor(0, device=self.device)

                for b, batch in enumerate(train_loader, 1):
                    xb, yb, rb, _missed, _detected = batch
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    rb = rb.to(self.device, non_blocking=True)

                    with amp.autocast("cuda", enabled=self.amp):
                        logits = model(xb)
                        yb_r = resize_masks_to(logits, yb)
                        rb_r = resize_masks_to(logits, rb)
                        valid = valid_mask_from_real(rb_r)
                        loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight_t=posw_warm)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite loss at warmup epoch {ep} iter {b}: {loss.item()}")

                    if verbose == 1 and is_main_process():
                        print(f"[WARMUP ep{ep} iter {b}] loss {loss.item():.4f}")
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                    # EMA update per step
                    if ema is not None:
                        ema.update(raw_model)

                    loss_sum_t += loss.detach() * xb.size(0)
                    seen_t += xb.size(0)

                    if int(warmup_batches) > 0 and b >= int(warmup_batches):
                        break

                train_loss = float((loss_sum_t / seen_t.clamp_min(1)).item())

                if verbose >= 2 and is_main_process():
                    if ep == int(warmup_epochs):
                        if ema is not None and ema_eval:
                            ema.apply_to(raw_model)
                            try:
                                auc = masked_pixel_auc(
                                    model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                                )
                            finally:
                                ema.restore(raw_model)
                        else:
                            auc = masked_pixel_auc(
                                model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                            )
                        dt = time.time() - t0
                        print(f"[WARMUP ep{ep}] train_loss {train_loss:.4f} | val AUC {auc:.4f} | {dt:.1f}s")
                    else:
                        dt = time.time() - t0
                        print(f"[WARMUP ep{ep}] train_loss {train_loss:.4f} | {dt:.1f}s")

            # Head
            freeze_all(raw_model)
            if hasattr(raw_model, "head"):
                for p in raw_model.head.parameters():
                    p.requires_grad = True
            for g in ["u4", "u3", "aspp"]:
                _unfreeze_if_exists(raw_model, g)

            opt = torch.optim.Adam(
                [p for p in raw_model.parameters() if p.requires_grad],
                lr=float(head_lr),
                weight_decay=float(weight_decay),
            )
            posw_head = self._posw(dtype=torch.float32, value=float(head_pos_weight))

            for ep in range(1, int(head_epochs) + 1):
                t0 = time.time()
                self._set_loader_epoch(train_loader, seed + 200 + ep)
                model.train()

                loss_sum_t = torch.tensor(0.0, device=self.device)
                seen_t = torch.tensor(0, device=self.device)

                for b, batch in enumerate(train_loader, 1):
                    xb, yb, rb, _missed, _detected = batch
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    rb = rb.to(self.device, non_blocking=True)

                    with amp.autocast("cuda", enabled=self.amp):
                        logits = model(xb)
                        yb_r = resize_masks_to(logits, yb)
                        rb_r = resize_masks_to(logits, rb)
                        valid = valid_mask_from_real(rb_r)
                        loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight_t=posw_head)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite loss at head epoch {ep} iter {b}: {loss.item()}")

                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                    if verbose == 1 and is_main_process():
                        print(f"[HEAD ep{ep} iter {b}] loss {loss.item():.4f}")

                    if ema is not None:
                        ema.update(raw_model)

                    loss_sum_t += loss.detach() * xb.size(0)
                    seen_t += xb.size(0)

                    if int(head_batches) > 0 and b >= int(head_batches):
                        break

                train_loss = float((loss_sum_t / seen_t.clamp_min(1)).item())

                if verbose >= 2 and is_main_process():
                    if ep == int(head_epochs):
                        if ema is not None and ema_eval:
                            ema.apply_to(raw_model)
                            try:
                                auc = masked_pixel_auc(
                                    model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                                )
                            finally:
                                ema.restore(raw_model)
                        else:
                            auc = masked_pixel_auc(
                                model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                            )
                        dt = time.time() - t0
                        print(f"[HEAD ep{ep}] train_loss {train_loss:.4f} | val AUC {auc:.4f} | {dt:.1f}s")
                    else:
                        dt = time.time() - t0
                        print(f"[HEAD ep{ep}] train_loss {train_loss:.4f} | {dt:.1f}s")

            # Tail
            freeze_all(raw_model)
            if hasattr(raw_model, "head"):
                for p in raw_model.head.parameters():
                    p.requires_grad = True
            for g in ["u4", "u3", "aspp"]:
                _unfreeze_if_exists(raw_model, g)

            opt = torch.optim.Adam(
                [p for p in raw_model.parameters() if p.requires_grad],
                lr=float(tail_lr),
                weight_decay=float(weight_decay),
            )
            posw_tail = self._posw(dtype=torch.float32, value=float(tail_pos_weight))

            for ep in range(1, int(tail_epochs) + 1):
                t0 = time.time()
                self._set_loader_epoch(train_loader, seed + 300 + ep)
                model.train()

                loss_sum_t = torch.tensor(0.0, device=self.device)
                seen_t = torch.tensor(0, device=self.device)

                for b, batch in enumerate(train_loader, 1):
                    xb, yb, rb, _missed, _detected = batch
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                    rb = rb.to(self.device, non_blocking=True)

                    with amp.autocast("cuda", enabled=self.amp):
                        logits = model(xb)
                        yb_r = resize_masks_to(logits, yb)
                        rb_r = resize_masks_to(logits, rb)
                        valid = valid_mask_from_real(rb_r)
                        loss = masked_bce_with_logits(logits, yb_r, valid, pos_weight_t=posw_tail)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite loss at tail epoch {ep} iter {b}: {loss.item()}")

                        if verbose == 1 and is_main_process():
                            print(f"[TAIL ep{ep} iter {b}] loss {loss.item():.4f}")

                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                    if ema is not None:
                        ema.update(raw_model)

                    loss_sum_t += loss.detach() * xb.size(0)
                    seen_t += xb.size(0)

                    if int(tail_batches) > 0 and b >= int(tail_batches):
                        break

                train_loss = float((loss_sum_t / seen_t.clamp_min(1)).item())

                if verbose >= 2 and is_main_process():
                    if ep == int(tail_epochs):
                        if ema is not None and ema_eval:
                            ema.apply_to(raw_model)
                            try:
                                auc = masked_pixel_auc(
                                    model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                                )
                            finally:
                                ema.restore(raw_model)
                        else:
                            auc = masked_pixel_auc(
                                model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                            )
                        dt = time.time() - t0
                        print(f"[TAIL ep{ep}] train_loss {train_loss:.4f} | val AUC {auc:.4f} | {dt:.1f}s")
                    else:
                        dt = time.time() - t0
                        print(f"[TAIL ep{ep}] train_loss {train_loss:.4f} | {dt:.1f}s")

        # -------------------------
        # Long training
        # -------------------------
        best: Dict[str, Any] = {
            "auc": float(best_auc),
            "ep": int(best_ep),
            "state": best_state,
            "state_ema": best_state_ema,
        }

        for ep in range(int(start_epoch), int(max_epochs) + 1):
            t0 = time.time()
            self._set_loader_epoch(train_loader, seed + 1000 + ep)

            # schedule freeze/unfreeze
            _groups = apply_phase(raw_model, ep)

            stage_lr = self._stage_lr(ep, base_lrs)

            if long_opt is None:
                long_opt = torch.optim.Adam(
                    [p for p in raw_model.parameters() if p.requires_grad],
                    lr=stage_lr,
                    weight_decay=float(weight_decay),
                )
                long_stage_lr = stage_lr
                long_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    long_opt, T_0=6, T_mult=2, eta_min=stage_lr / 10.0
                )
            else:
                self._sync_optimizer_params(long_opt, raw_model)
                # update LR in-place; do NOT recreate scheduler (keeps continuity)
                if long_stage_lr is None or abs(stage_lr - float(long_stage_lr)) > 0:
                    for pg in long_opt.param_groups:
                        pg["lr"] = stage_lr
                        pg["weight_decay"] = float(weight_decay)
                    long_stage_lr = stage_lr
                    # keep existing scheduler; optionally adjust eta_min if present
                    if hasattr(long_sched, "eta_min"):
                        try:
                            long_sched.eta_min = stage_lr / 10.0
                        except Exception:
                            pass

            assert long_opt is not None
            assert long_sched is not None

            model.train()

            lam = ramp_lambda(
                ep,
                kind=str(ramp_kind),
                e0=int(ramp_start_epoch),
                e1=int(ramp_end_epoch),
                k=float(sigmoid_k),
            )

            posw_long = self._posw(dtype=torch.float32, value=float(bce_pos_weight_long))

            loss_sum_t = torch.tensor(0.0, device=self.device)
            seen_t = torch.tensor(0, device=self.device)

            for i, batch in enumerate(train_loader, 1):
                xb, yb, rb, _missed, _detected = batch
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                rb = rb.to(self.device, non_blocking=True)

                with amp.autocast("cuda", enabled=self.amp):
                    logits = model(xb)
                    yb_r = resize_masks_to(logits, yb)
                    rb_r = resize_masks_to(logits, rb)
                    valid = valid_mask_from_real(rb_r)

                    loss_bce = masked_bce_with_logits(logits, yb_r, valid, pos_weight_t=posw_long)
                    if float(lam) <= 0.0:
                        loss = loss_bce
                    else:
                        loss_ft = focal_tversky_masked(
                            logits, yb_r, valid, alpha=float(ft_alpha), gamma=float(ft_gamma)
                        )
                        loss = (1.0 - float(lam)) * loss_bce + float(lam) * loss_ft

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at long epoch {ep} iter {i}: {loss.item()}")

                    if verbose == 1 and is_main_process():
                        print(f"[LONG ep{ep} iter {i}] loss {loss.item():.4f} | lam {lam:.3f}")

                long_opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(long_opt)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(long_opt)
                scaler.update()

                # scheduler step with fractional epoch
                long_sched.step(ep + i / max(1, len(train_loader)))

                # EMA update
                if ema is not None:
                    ema.update(raw_model)

                loss_sum_t += loss.detach() * xb.size(0)
                seen_t += xb.size(0)

                if int(long_batches) > 0 and i >= int(long_batches):
                    break

            train_loss = float((loss_sum_t / seen_t.clamp_min(1)).item())

            if verbose >= 2 and is_main_process():
                dt = time.time() - t0
                print(f"[EP{ep:03d}] loss {train_loss:.4f} | lam {lam:.3f} | lr {stage_lr:.2e} | {dt:.1f}s")

            # Validation
            do_val = (ep % int(val_every) == 0) or (ep <= 3)
            auc_eval = None

            if do_val:
                self._set_loader_epoch(val_loader, seed + 2000 + ep)

                # evaluate RAW or EMA
                if ema is not None and ema_eval:
                    ema.apply_to(raw_model)
                    try:
                        auc_eval = masked_pixel_auc(
                            model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                        )
                    finally:
                        ema.restore(raw_model)
                else:
                    auc_eval = masked_pixel_auc(
                        model, val_loader, device=self.device, n_bins=int(auc_bins), max_batches=int(auc_batches)
                    )

                if verbose >= 1 and is_main_process():
                    tag = "EMA" if (ema is not None and ema_eval) else "RAW"
                    print(f"[VAL ep{ep:03d}] {tag} AUC {float(auc_eval):.4f}")

                # best selection based on eval AUC
                if float(auc_eval) > float(best["auc"]):
                    best["auc"] = float(auc_eval)
                    best["ep"] = int(ep)
                    best["state"] = copy.deepcopy(raw_model.state_dict())

                    # also store EMA state if enabled (shadow weights)
                    if ema is not None:
                        best["state_ema"] = copy.deepcopy(ema.state_dict())

                    if is_main_process() and save_best_to:
                        ensure_parent_dir(save_best_to)
                        torch.save(
                            {
                                "state": best["state"],
                                "ep": best["ep"],
                                "auc": best["auc"],
                                "best_state_ema": best.get("state_ema", None),
                                "model_metadata": {
                                    "model_name": raw_model.__class__.__name__,
                                    "model_hparams": dict(model_hparams) if model_hparams is not None else None,
                                    "tile": int(expected_tile) if expected_tile is not None else None,
                                    "norm": str(norm_name) if norm_name is not None else None,
                                },
                            },
                            save_best_to,
                        )

            # Save last (full resume)
            if is_main_process() and save_last_to:
                ensure_parent_dir(save_last_to)
                torch.save(
                    {
                        "state": raw_model.state_dict(),
                        "ep": int(ep),
                        "train_loss": float(train_loss),
                        "lam": float(lam),
                        "stage_lr": float(stage_lr),
                        "auc": float(auc_eval) if auc_eval is not None else None,
                        "long_opt": long_opt.state_dict() if long_opt is not None else None,
                        "long_sched": long_sched.state_dict() if long_sched is not None else None,
                        "long_stage_lr": float(long_stage_lr) if long_stage_lr is not None else None,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_auc": float(best["auc"]),
                        "best_ep": int(best["ep"]),
                        "best_state": best["state"],
                        "best_state_ema": best.get("state_ema", None),
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

        # Load best (raw)
        if best["state"] is not None:
            raw_model.load_state_dict(best["state"], strict=True)

        summary = {
            "best_ep": int(best["ep"]),
            "best_auc": float(best["auc"]),
            "ema_enabled": bool(use_ema),
            "ema_decay": float(ema_decay),
            "ema_eval": bool(ema_eval),
        }

        if is_main_process() and verbose >= 1:
            print("=== DONE ===")
            print("Best summary:", summary)

        trained_model = raw_model
        return trained_model, float(fixed_thr), summary