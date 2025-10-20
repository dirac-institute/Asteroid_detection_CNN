# thresholds.py
import torch, torch.distributed as dist
import torch.nn.functional as F
from typing import Tuple, Dict
from ADCNN.utils.dist_utils import broadcast_scalar_float

def resize_masks_to(logits, masks):
    H, W = logits.shape[-2:]
    if masks.dtype != torch.float32:
        masks = masks.float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    if masks.shape[-2:] == (H, W):
        return (masks > 0.5).float()
    out = F.interpolate(masks, size=(H, W), mode="nearest")
    return (out > 0.5).float()

@torch.no_grad()
def _global_histograms(model, loader, n_bins=256, max_batches=40):
    """DDP-safe pos/neg histograms of predicted probabilities."""
    model.eval(); dev = next(model.parameters()).device
    hist_pos = torch.zeros(n_bins, device=dev)
    hist_neg = torch.zeros(n_bins, device=dev)
    for i, (xb, yb) in enumerate(loader, 1):
        xb, yb = xb.to(dev), yb.to(dev)
        p = torch.sigmoid(model(xb))
        yb_r = resize_masks_to(p, yb)
        pv = p.reshape(-1)
        tv = (yb_r > 0.5).reshape(-1)
        # torch.histogram is preferred; torch.histc still ok.
        hist_pos += torch.histc(pv[tv], bins=n_bins, min=0.0, max=1.0)
        hist_neg += torch.histc(pv[~tv], bins=n_bins, min=0.0, max=1.0)
        if i >= max_batches:
            break
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(hist_neg, op=dist.ReduceOp.SUM)
    return hist_pos, hist_neg

def _thr_from_index(edges: torch.Tensor, idx: int) -> float:
    # pick midpoint between bin edges
    return float((edges[idx] + edges[idx+1]) * 0.5)

@torch.no_grad()
def pick_thr_with_band(
    model,
    loader,
    *,
    n_bins: int = 256,
    max_batches: int = 120,
    beta: float = 1.0,
    min_pos_rate: float = 0.03,
    max_pos_rate: float = 0.10,
) -> Tuple[float, Tuple[float, float, float], Dict[str, float]]:
    """
    Choose threshold: prefer F_beta optimum if its pos_rate is inside [min,max];
    otherwise choose threshold inside the band closest to band center.
    """
    dev = next(model.parameters()).device
    hist_pos, hist_neg = _global_histograms(model, loader, n_bins=n_bins, max_batches=max_batches)
    edges = torch.linspace(0, 1, n_bins + 1, device=dev)

    # Cumulative tails: counts of scores >= t for each bin index
    cpos = torch.flip(torch.cumsum(torch.flip(hist_pos, dims=[0]), 0), dims=[0])
    cneg = torch.flip(torch.cumsum(torch.flip(hist_neg, dims=[0]), 0), dims=[0])

    TP = cpos
    FP = cneg
    FN = (hist_pos.sum() - TP).clamp(min=0)

    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    fbeta = (1 + beta * beta) * P * R / (beta * beta * P + R + 1e-8)

    total = (hist_pos.sum() + hist_neg.sum()).clamp_min(1.0)
    pos_rate = (TP + FP) / total  # fraction of pixels predicted positive at each threshold bin

    # F_beta-optimal index
    idx_f = int(torch.argmax(fbeta).item())
    thr_f = _thr_from_index(edges, idx_f)

    # If F-beta solution already satisfies band -> use it
    pr_f = float(pos_rate[idx_f].item())
    if min_pos_rate <= pr_f <= max_pos_rate:
        thr = thr_f; i_best = idx_f
    else:
        # pick inside band: nearest to band center (or nearest boundary if band empty)
        band_mask = (pos_rate >= min_pos_rate) & (pos_rate <= max_pos_rate)
        if band_mask.any():
            center = 0.5 * (min_pos_rate + max_pos_rate)
            # minimize |pos_rate - center|
            i_best = int(torch.argmin(torch.abs(pos_rate[band_mask] - center)).item())
            # map back to original indices
            band_indices = torch.nonzero(band_mask, as_tuple=False).squeeze(1)
            i_best = int(band_indices[i_best].item())
            thr = _thr_from_index(edges, i_best)
        else:
            # band unreachable: project to nearest boundary
            # choose the index with pos_rate closest to [min_pos_rate, max_pos_rate]
            target = min_pos_rate if pr_f < min_pos_rate else max_pos_rate
            i_best = int(torch.argmin(torch.abs(pos_rate - target)).item())
            thr = _thr_from_index(edges, i_best)

    # Prepare return metrics at chosen index
    P_b = float(P[i_best].item()); R_b = float(R[i_best].item()); F_b = float(fbeta[i_best].item())
    pr_b = float(pos_rate[i_best].item())

    # Broadcast single scalar thr so all ranks agree
    thr = broadcast_scalar_float(thr, src=0, device=dev)
    return thr, (P_b, R_b, F_b), {"pos_rate": pr_b}

@torch.no_grad()
def pick_thr_with_floor(model, loader, max_batches=40, n_bins=256, beta=1.0, min_pos_rate=0.05, max_pos_rate=0.10):
    return pick_thr_with_band(
        model, loader,
        n_bins=n_bins, max_batches=max_batches, beta=beta,
        min_pos_rate=min_pos_rate, max_pos_rate=max_pos_rate,
    )
