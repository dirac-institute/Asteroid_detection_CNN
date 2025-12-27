import torch, torch.distributed as dist
import torch.nn.functional as F

@torch.no_grad()
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
def roc_auc_ddp(model, loader, n_bins: int = 256, max_batches: int = 120) -> float:
    """
    DDP-safe ROC-AUC computed via probability histograms.
    Approximates exact AUC well for n_bins>=256 without holding all pixels in RAM.
    """
    model.eval()
    dev = next(model.parameters()).device
    hist_pos = torch.zeros(n_bins, device=dev)
    hist_neg = torch.zeros(n_bins, device=dev)

    # Build histograms on each rank
    for i, (xb, yb) in enumerate(loader, 1):
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        p = torch.sigmoid(model(xb))
        yb_r = resize_masks_to(p, yb)
        pv = p.reshape(-1)
        tv = (yb_r > 0.5).reshape(-1)
        # torch.histc is fine; swap to torch.histogram if you prefer
        hist_pos += torch.histc(pv[tv],   bins=n_bins, min=0.0, max=1.0)
        hist_neg += torch.histc(pv[~tv],  bins=n_bins, min=0.0, max=1.0)
        if i >= max_batches:
            break

    # All-reduce to get global histograms
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(hist_pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(hist_neg, op=dist.ReduceOp.SUM)

    P_tot = hist_pos.sum().clamp_min(1.0)
    N_tot = hist_neg.sum().clamp_min(1.0)

    # For thresholds from 1 -> 0, cumulative tails are TP (>=t), FP (>=t)
    cpos = torch.flip(torch.cumsum(torch.flip(hist_pos, dims=[0]), 0), dims=[0])
    cneg = torch.flip(torch.cumsum(torch.flip(hist_neg, dims=[0]), 0), dims=[0])

    TPR = (cpos / P_tot).clamp(0, 1)   # recall
    FPR = (cneg / N_tot).clamp(0, 1)

    # Ensure FPR is increasing for trapezoid rule (it already is: thresholds go high->low)
    # Reverse to low->high just to be explicit
    TPR = torch.flip(TPR, dims=[0])
    FPR = torch.flip(FPR, dims=[0])

    # Trapz integrate TPR(FPR)
    auc = torch.trapz(TPR, FPR).item()
    return float(auc)
