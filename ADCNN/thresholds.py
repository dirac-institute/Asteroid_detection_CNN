import torch, torch.nn.functional as F

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
def pick_thr_under_min(model, loader, max_batches=40, n_bins=256, beta=2.0):
    model.eval(); dev = next(model.parameters()).device
    hist_pos = torch.zeros(n_bins, device=dev); hist_neg = torch.zeros(n_bins, device=dev)
    edges = torch.linspace(0,1,n_bins+1, device=dev)
    for i,(xb,yb) in enumerate(loader,1):
        xb,yb = xb.to(dev), yb.to(dev)
        p = torch.sigmoid(model(xb))
        yb_r = resize_masks_to(p, yb)
        pv = p.reshape(-1); tv = (yb_r>0.5).reshape(-1)
        hist_pos += torch.histc(pv[tv], bins=n_bins, min=0, max=1)
        hist_neg += torch.histc(pv[~tv], bins=n_bins, min=0, max=1)
        if i>=max_batches: break
    cpos = torch.flip(torch.cumsum(torch.flip(hist_pos, dims=[0]), 0), dims=[0])
    cneg = torch.flip(torch.cumsum(torch.flip(hist_neg, dims=[0]), 0), dims=[0])
    TP = cpos; FP = cneg; FN = (hist_pos.sum() - TP).clamp(min=0)
    P = TP / (TP + FP + 1e-8); R = TP / (TP + FN + 1e-8)
    fbeta = (1+beta*beta)*P*R / (beta*beta*P + R + 1e-8)
    idx = int(torch.argmax(fbeta).item())
    thr = float((edges[idx] + edges[idx+1])/2)
    return thr, (float(P[idx]), float(R[idx]), float(fbeta[idx])), dict(pos_rate=float((TP[idx]+FP[idx])/(hist_pos.sum()+hist_neg.sum()+1e-8)))

@torch.no_grad()
def pick_thr_with_floor(model, loader, max_batches=40, n_bins=256, beta=1.0, min_pos_rate=0.05, max_pos_rate=0.10):
    thr, (P,R,F), aux = pick_thr_under_min(model, loader, max_batches=max_batches, n_bins=n_bins, beta=beta)
    # Placeholder for “floor” clamp; keep as-is for now
    return thr, (P,R,F), aux
