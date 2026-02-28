"""
Consolidated metrics module - single source of truth for all evaluation metrics.

Includes:
- Masked pixel AUC (DDP-safe)
- Precision, Recall, F-scores
- Mask resizing utilities
"""

from typing import Iterable, Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


# =============================================================================
# Mask Utilities
# =============================================================================

@torch.no_grad()
def resize_masks_to(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Resize mask tensor to match logits spatial size using nearest neighbor.

    Accepts masks with shape:
      - (B, H, W) or (B, 1, H, W) or (H, W) / (1, H, W)

    Returns float mask in {0,1}.
    """
    H, W = int(logits.shape[-2]), int(logits.shape[-1])

    if masks.dim() == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif masks.dim() == 3:
        masks = masks.unsqueeze(1)  # (B,1,H,W)

    masks = masks.float()
    if tuple(masks.shape[-2:]) == (H, W):
        return (masks > 0.5).float()

    out = F.interpolate(masks, size=(H, W), mode="nearest")
    return (out > 0.5).float()


def valid_mask_from_real(real: torch.Tensor) -> torch.Tensor:
    """
    Convert real source mask to valid evaluation mask.

    real: mask of pixels that belong to REAL sources/labels (to be ignored).
    Returns valid mask = 1 where training/eval should count pixels.
    """
    if real.dtype != torch.bool:
        real = real > 0.5
    return (~real).to(dtype=torch.float32)


# =============================================================================
# Masked Pixel AUC (DDP-safe, histogram-based)
# =============================================================================

@torch.no_grad()
def masked_pixel_auc(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    n_bins: int = 256,
    max_batches: int = 12,
) -> float:
    """
    Compute ROC AUC over pixels, masking out "real" pixels.

    Expects loader batches:
      - (xb, yb) or
      - (xb, yb, real, ...) where real marks pixels to ignore (real>0.5)

    Uses histogram accumulation + DDP allreduce for efficiency.

    Args:
        model: Model to evaluate
        loader: DataLoader yielding (x, y, real, ...) batches
        device: Device for computation
        n_bins: Number of histogram bins for score discretization
        max_batches: Max batches to evaluate (0 = all)

    Returns:
        AUC score (float)

    DEPRICATED
    """
    model.eval()
    n_bins = int(n_bins)

    hist_pos = torch.zeros((n_bins,), dtype=torch.float64, device=device)
    hist_neg = torch.zeros((n_bins,), dtype=torch.float64, device=device)

    for bi, batch in enumerate(loader, 1):
        # Unpack batch - support both formats for compatibility
        if len(batch) == 5:
            # Training format: (xb, yb, rb, missed, detected)
            xb, yb, rb = batch[0], batch[1], batch[2]
        elif len(batch) >= 3:
            xb, yb, rb = batch[0], batch[1], batch[2]
        else:
            xb, yb = batch[0], batch[1]
            rb = None

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        rb = rb.to(device, non_blocking=True) if rb is not None else None

        # Forward pass
        logits = model(xb)
        yb_r = resize_masks_to(logits, yb).float().clamp(0.0, 1.0)

        # Compute valid mask
        if rb is None:
            valid = torch.ones_like(yb_r, dtype=torch.float32, device=device)
        else:
            rb_r = resize_masks_to(logits, rb)
            valid = valid_mask_from_real(rb_r)

        # Compute scores and accumulate histograms
        p = torch.sigmoid(logits.float()).reshape(-1)
        t = yb_r.reshape(-1)
        v = valid.reshape(-1)

        m = v > 0.5
        if m.any():
            p = p[m]
            t = t[m]

            idx = torch.clamp((p * n_bins).to(torch.int64), 0, n_bins - 1)
            wpos = t.to(torch.float64)
            wneg = (1.0 - t).to(torch.float64)

            hist_pos += torch.bincount(idx, weights=wpos, minlength=n_bins).to(torch.float64)
            hist_neg += torch.bincount(idx, weights=wneg, minlength=n_bins).to(torch.float64)

        if max_batches > 0 and bi >= int(max_batches):
            break

    # DDP reduction
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(hist_neg, op=dist.ReduceOp.SUM)

    # Compute AUC on rank 0
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    if rank == 0:
        pos = hist_pos.detach().cpu().numpy()
        neg = hist_neg.detach().cpu().numpy()
        P = float(pos.sum())
        N = float(neg.sum())

        if P <= 0.0 or N <= 0.0:
            auc = 0.5
        else:
            tps = np.cumsum(pos[::-1])
            fps = np.cumsum(neg[::-1])
            tpr = tps / P
            fpr = fps / N
            auc = float(np.trapz(tpr, fpr))

        out = torch.tensor([auc], dtype=torch.float32, device=device)
    else:
        out = torch.tensor([0.0], dtype=torch.float32, device=device)

    # Broadcast result
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return float(out.item())

@torch.no_grad()
def masked_pixel_auc_agg(model, loader, device, *, n_bins=256, max_batches=120):
    """
    Aggregated pixel ROC-AUC via histogram accumulation across many batches.
    DDP-safe (all_reduce on histograms).
    Returns NaN if no positives or no negatives were seen overall.
    """
    model.eval()
    hist_pos = torch.zeros(int(n_bins), device=device, dtype=torch.float64)
    hist_neg = torch.zeros(int(n_bins), device=device, dtype=torch.float64)

    nb = 0
    for nb, batch in enumerate(loader, 1):
        xb, yb, rb, *_ = batch  # your loader yields (x, y, real, missed, detected) :contentReference[oaicite:3]{index=3}
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        rb = rb.to(device, non_blocking=True)

        logits = model(xb)
        # same resizing logic you already use in training
        yb_r = resize_masks_to(logits, yb)          # [B,1,H,W] in {0,1}
        rb_r = resize_masks_to(logits, rb)          # [B,1,H,W] in {0,1}
        valid = valid_mask_from_real(rb_r)          # [B,1,H,W] in {0,1}

        probs = torch.sigmoid(logits).detach().float()
        t = yb_r.detach().float()
        v = valid.detach().float()

        m = (v > 0.5).reshape(-1)
        if not bool(m.any()):
            if max_batches and nb >= int(max_batches): break
            continue

        p = probs.reshape(-1)[m]
        t = t.reshape(-1)[m]

        idx = torch.clamp((p * int(n_bins)).to(torch.int64), 0, int(n_bins) - 1)
        hist_pos += torch.bincount(idx, weights=t.to(torch.float64), minlength=int(n_bins)).to(torch.float64)
        hist_neg += torch.bincount(idx, weights=(1.0 - t).to(torch.float64), minlength=int(n_bins)).to(torch.float64)

        if max_batches and nb >= int(max_batches):
            break

    # DDP: sum histograms across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(hist_neg, op=dist.ReduceOp.SUM)

    P = float(hist_pos.sum().item())
    N = float(hist_neg.sum().item())
    if P <= 0.0 or N <= 0.0:
        return float("nan")

    tp = torch.cumsum(torch.flip(hist_pos, dims=[0]), dim=0)
    fp = torch.cumsum(torch.flip(hist_neg, dims=[0]), dim=0)
    tpr = tp / max(P, 1e-12)
    fpr = fp / max(N, 1e-12)

    # trapezoid integral over FPR
    auc = torch.trapz(tpr, fpr).item()
    return float(auc)
# =============================================================================
# Classification Metrics (NumPy-based)
# =============================================================================

def precision(tp, fp):
    """
    Precision = TP / (TP + FP)

    Handles arrays or scalars. Returns 0 where denominator is 0.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = tp + fp
    m = denom > 0
    out[m] = tp[m] / denom[m]
    return out


def recall(tp, fn):
    """
    Recall = TP / (TP + FN)

    Handles arrays or scalars. Returns 0 where denominator is 0.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = tp + fn
    m = denom > 0
    out[m] = tp[m] / denom[m]
    return out


def f1_score(tp, fp, fn):
    """
    F1 Score = 2*TP / (2*TP + FP + FN)

    Handles arrays or scalars. Returns 0 where denominator is 0.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = 2 * tp + fp + fn
    m = denom > 0
    out[m] = (2 * tp[m]) / denom[m]
    return out


def f2_score(tp, fp, fn):
    """
    F2 Score = 5*TP / (5*TP + FP + 4*FN)

    Uses beta=2 which weights recall higher than precision.
    Handles arrays or scalars. Returns 0 where denominator is 0.
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = 5 * tp + fp + 4 * fn
    m = denom > 0
    out[m] = (5 * tp[m]) / denom[m]
    return out


def fppi(fp: np.ndarray, denom_images: int) -> np.ndarray:
    """False Positives Per Image."""
    fp = np.asarray(fp, dtype=np.float64)
    if denom_images <= 0:
        raise ValueError("denom_images must be > 0")
    return fp / float(denom_images)


def make_monotone_increasing(y: np.ndarray) -> np.ndarray:
    """Force array to be monotone increasing (useful for recall in FROC)."""
    y = np.asarray(y, dtype=np.float64)
    return np.maximum.accumulate(y)

