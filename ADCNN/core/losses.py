"""Training losses for the v7 diffim detector.

``masked_aftl_loss`` — masked Asymmetric Focal Tversky (the primary segmentation loss,
favouring recall on the thin-trail positives) plus a small unweighted BCE anchor.
``masked_orient_mse`` — unit-circle (sin2β, cos2β) MSE on the orientation head, only
where the trail mask is set. Both ignore pixels flagged by the ignore mask.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


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
