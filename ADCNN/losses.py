import torch, torch.nn as nn, torch.nn.functional as F

class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits); t = targets.clamp(0,1)
        inter = (p*t).sum(dim=(1,2,3))
        union = (p + t - p*t).sum(dim=(1,2,3)) + self.eps
        iou = inter/union
        return (1 - iou).mean()

class AFTL(nn.Module):
    def __init__(self, alpha=0.45, beta=0.55, gamma=1.3, eps=1e-6):
        super().__init__(); self.alpha, self.beta, self.gamma, self.eps = alpha,beta,gamma,eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(self.eps, 1-self.eps)
        t = targets.clamp(0,1)
        p = p.view(p.size(0), -1); t = t.view(t.size(0), -1)
        TP = (p*t).sum(1); FP = ((1-t)*p).sum(1); FN = (t*(1-p)).sum(1)
        tv = (TP+self.eps)/(TP + self.alpha*FP + self.beta*FN + self.eps)
        return torch.pow(1.0 - tv, self.gamma).mean()

class BCEIoUEdge(nn.Module):
    """λ_bce * BCE(pos_weight) + (1-λ_bce) * SoftIoU [+ λ_edge * Sobel L1]"""
    def __init__(self, lambda_bce=0.6, pos_weight=8.0, lambda_edge=0.0):
        super().__init__()
        self.lambda_bce = float(lambda_bce)
        self.lambda_edge = float(lambda_edge)
        self.iou = SoftIoULoss()
        self.posw = float(pos_weight)
        kx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).unsqueeze(0)
        ky = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).unsqueeze(0)
        self.register_buffer("kx", kx); self.register_buffer("ky", ky)
    def _edge(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-12)
    def forward(self, logits, targets):
        t = targets.clamp(0,1)
        posw = torch.tensor(self.posw, device=logits.device)
        bce  = F.binary_cross_entropy_with_logits(logits, t, pos_weight=posw)
        siou = self.iou(logits, t)
        loss = self.lambda_bce*bce + (1.0-self.lambda_bce)*siou
        if self.lambda_edge>0:
            p = torch.sigmoid(logits)
            loss = loss + self.lambda_edge * F.l1_loss(self._edge(p), self._edge(t))
        return loss

def blended_loss(core, aftl, w, logits, targets):
    loss = w["w_core"] * core(logits, targets)
    if aftl is not None and w.get("w_aftl", 0) > 0:
        loss = loss + w["w_aftl"] * aftl(logits, targets)
    return loss

def make_loss_for_epoch(ep: int, device: torch.device):
    if ep <= 10:
        core = BCEIoUEdge(lambda_bce=0.6, pos_weight=8.0, lambda_edge=0.00).to(device); aftl=None
        return core, aftl, {"w_core":1.0, "w_aftl":0.0}
    elif ep <= 25:
        core = BCEIoUEdge(lambda_bce=0.6, pos_weight=8.0, lambda_edge=0.00).to(device)
        aftl = AFTL(alpha=0.45, beta=0.55, gamma=1.3).to(device)
        return core, aftl, {"w_core":0.85, "w_aftl":0.15}
    else:
        core = BCEIoUEdge(lambda_bce=0.8, pos_weight=8.0, lambda_edge=0.03).to(device)
        aftl = AFTL(alpha=0.45, beta=0.55, gamma=1.3).to(device)
        return core, aftl, {"w_core":0.85, "w_aftl":0.15}
