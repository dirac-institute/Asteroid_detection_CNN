"""Two-stage candidate extraction from a panel-level probability map.

stage 1: t_low binarize -> connected components (8-connectivity)
stage 2: per-candidate features (max p, area, elongation, bbox center,
         top-k mean p, integrated logit). The candidate score is `max_p`
         by default; emit all features so we can replace the scorer later
         without re-running the network.

This module is intentionally separate from training. `evaluate.py` calls
extract_candidates(panel_prob, real_labels=…) and gets a pandas DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


@dataclass
class CandidateExtractorConfig:
    t_low: float = 0.05
    min_area: int = 4
    connectivity: int = 2  # ndimage uses 2 for 8-connectivity in 2D
    # Adaptive t_low: if the panel's prob distribution is saturated (the model
    # collapsed to a near-constant output), the static t_low produces one giant
    # connected component. Override with max(t_low, panel_mean + k * panel_std)
    # so candidates are always anomalies on top of the model's typical output.
    adaptive_t_low: bool = True
    adaptive_t_low_k: float = 6.0
    # Hard ceiling so we don't suppress legitimate detections when the model
    # behaves normally (sparse positives, mean ≈ 0).
    adaptive_t_low_max: float = 0.5


def _component_features(prob: np.ndarray, mask: np.ndarray) -> dict:
    """Features for one connected component. `mask` is the component, `prob`
    is the same-shape probability slice over its bbox."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return {}
    area = int(mask.sum())
    cy = float(ys.mean())
    cx = float(xs.mean())

    # PCA-elongation
    if area >= 4:
        coords = np.stack([ys - cy, xs - cx], axis=1).astype(np.float64)
        cov = coords.T @ coords / max(area - 1, 1)
        w = np.linalg.eigvalsh(cov)
        w = np.clip(w, 1e-6, None)
        eigratio = float(w[-1] / w[0])
    else:
        eigratio = 1.0

    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    aspect = float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1))

    p_inside = prob[mask]
    max_p = float(p_inside.max())
    mean_p = float(p_inside.mean())
    p_sorted = np.sort(p_inside)[::-1]
    topk = p_sorted[: min(5, p_sorted.size)]
    top5_mean = float(topk.mean())
    integrated_logit = float(np.log(np.clip(p_inside, 1e-6, 1 - 1e-6)).sum() -
                             np.log1p(-np.clip(p_inside, 1e-6, 1 - 1e-6)).sum())

    return {
        "area": area,
        "y_centroid": cy + 0,
        "x_centroid": cx + 0,
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "aspect": aspect,
        "elongation": eigratio,
        "max_p": max_p,
        "mean_p": mean_p,
        "top5_mean_p": top5_mean,
        "integrated_logit": integrated_logit,
    }


def extract_candidates(
    panel_prob: np.ndarray,
    *,
    real_labels: np.ndarray | None = None,
    cfg: CandidateExtractorConfig | None = None,
    panel_id: int | None = None,
) -> pd.DataFrame:
    """Return one row per connected component above `t_low`.

    `panel_prob` is the full-resolution (H, W) probability map for the panel.
    `real_labels` is the optional (H, W) array (>0 = LSST clean-diffim
    footprint). For each candidate, we compute the fraction of its area that
    overlaps real_labels — eval uses this to tag "informational" candidates.
    """
    cfg = cfg or CandidateExtractorConfig()
    effective_t_low = float(cfg.t_low)
    if cfg.adaptive_t_low:
        mu = float(panel_prob.mean())
        sd = float(panel_prob.std())
        adaptive = mu + cfg.adaptive_t_low_k * sd
        effective_t_low = min(max(cfg.t_low, adaptive), cfg.adaptive_t_low_max)
    binary = panel_prob > effective_t_low
    if not binary.any():
        return pd.DataFrame(columns=[
            "panel_id", "candidate_id",
            "y_centroid", "x_centroid", "area",
            "bbox_h", "bbox_w", "aspect", "elongation",
            "max_p", "mean_p", "top5_mean_p", "integrated_logit",
            "frac_real_label_overlap",
            "y_min", "y_max", "x_min", "x_max",
        ])
    structure = ndi.generate_binary_structure(2, cfg.connectivity)
    labels, n_lab = ndi.label(binary, structure=structure)

    rows = []
    objects = ndi.find_objects(labels)
    for cid in range(1, n_lab + 1):
        sl = objects[cid - 1]
        if sl is None:
            continue
        comp = labels[sl] == cid
        if comp.sum() < cfg.min_area:
            continue
        feats = _component_features(panel_prob[sl], comp)
        y_off = sl[0].start
        x_off = sl[1].start
        feats["y_centroid"] = y_off + feats["y_centroid"]
        feats["x_centroid"] = x_off + feats["x_centroid"]
        feats["y_min"] = int(sl[0].start)
        feats["y_max"] = int(sl[0].stop - 1)
        feats["x_min"] = int(sl[1].start)
        feats["x_max"] = int(sl[1].stop - 1)

        if real_labels is not None:
            rl_slice = (real_labels[sl] > 0)
            n_overlap = int((rl_slice & comp).sum())
            feats["frac_real_label_overlap"] = float(n_overlap / max(comp.sum(), 1))
        else:
            feats["frac_real_label_overlap"] = 0.0

        feats["panel_id"] = int(panel_id) if panel_id is not None else -1
        feats["candidate_id"] = int(cid)
        feats["effective_t_low"] = float(effective_t_low)
        rows.append(feats)

    cols = ["panel_id", "candidate_id",
            "y_centroid", "x_centroid", "area",
            "bbox_h", "bbox_w", "aspect", "elongation",
            "max_p", "mean_p", "top5_mean_p", "integrated_logit",
            "frac_real_label_overlap", "effective_t_low",
            "y_min", "y_max", "x_min", "x_max"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return df[cols]


def candidate_pixel_mask(
    shape: tuple[int, int],
    candidate: pd.Series,
    panel_prob: np.ndarray,
    *,
    t_low: float,
) -> np.ndarray:
    """Re-derive the binary footprint of one candidate from the original prob
    map (cheaper than caching all component masks). Used by the object-level
    matcher in evaluate.py."""
    y0, y1 = int(candidate["y_min"]), int(candidate["y_max"])
    x0, x1 = int(candidate["x_min"]), int(candidate["x_max"])
    sub = panel_prob[y0:y1 + 1, x0:x1 + 1] > t_low
    labels, n = ndi.label(sub)
    if n == 0:
        return np.zeros(shape, dtype=bool)
    # The candidate's centroid (relative to bbox) maps to one label.
    cy = int(round(candidate["y_centroid"] - y0))
    cx = int(round(candidate["x_centroid"] - x0))
    cy = int(np.clip(cy, 0, sub.shape[0] - 1))
    cx = int(np.clip(cx, 0, sub.shape[1] - 1))
    target_label = labels[cy, cx]
    if target_label == 0:
        # Pick the largest component as fallback.
        sizes = ndi.sum(sub, labels, index=np.arange(1, n + 1))
        target_label = int(np.argmax(sizes) + 1)
    out = np.zeros(shape, dtype=bool)
    out[y0:y1 + 1, x0:x1 + 1] = (labels == target_label)
    return out
