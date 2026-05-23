"""Per-detection parameter recovery (x, y, beta, trail_length).

After the V2 two-stage pipeline yields a mask of detections, this module
matches each detection back to its catalog injection (same matching the
notebook's `objectwise_confusion` uses) and produces a side-by-side table of
truth vs estimated parameters. Helpers for residual statistics and plotting
are included so you can see per-axis bias and dispersion.

Typical use:

    from ADCNN.evaluation.parameter_recovery import (
        evaluate_parameter_recovery, summarize_residuals, plot_residuals,
    )

    # After computing cand_df + filtering at rf_thr:
    matched = evaluate_parameter_recovery(
        catalog=test_catalog,
        cand_df=cand_df[cand_df.score_rf >= rf_thr],
        panel_probs=p,                       # (N, H, W)
        orient_sin=p_sin, orient_cos=p_cos,  # optional (improves beta estimate)
    )
    print(summarize_residuals(matched))
    plot_residuals(matched)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import cv2

from ADCNN.utils.angle_utils import deg2rad
from ADCNN.utils.helpers import draw_one_line
from ADCNN.inference.matched_filter import panel_mad_sigma


__all__ = [
    "estimate_component_params",
    "estimate_params_for_candidate",
    "mf_length_scan",
    "evaluate_parameter_recovery",
    "summarize_residuals",
    "plot_residuals",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_panel_dict(arr):
    if isinstance(arr, dict):
        return {int(k): np.asarray(v) for k, v in arr.items()}
    return {int(pid): np.asarray(arr[pid]) for pid in range(len(arr))}


def _wrap_beta_diff_deg(estimate_deg: float, truth_deg: float) -> float:
    """Signed minimal-angle difference for line-orientations (mod 180°).
    Result is in (-90, 90].
    """
    d = (estimate_deg - truth_deg) % 180.0
    if d > 90.0:
        d -= 180.0
    return float(d)


# ---------------------------------------------------------------------------
# Per-component parameter estimation
# ---------------------------------------------------------------------------

def estimate_component_params(
    ys: np.ndarray, xs: np.ndarray,
    probs: Optional[np.ndarray] = None,
    *,
    sin_vals: Optional[np.ndarray] = None,
    cos_vals: Optional[np.ndarray] = None,
) -> dict:
    """Estimate (x, y, beta, trail_length) from a connected-component pixel set.

    Args:
        ys, xs: 1-D arrays of pixel coordinates.
        probs: optional 1-D array of per-pixel weights (NN probabilities).
            If None, all pixels weighted equally.
        sin_vals, cos_vals: optional 1-D arrays of the model's orient
            sin(2β)/cos(2β) at the same pixels. When supplied, an
            orient-based beta estimate is also returned.

    Returns a dict with keys: x, y, beta_pca_deg, length_pca,
    eigratio (elongation), n_pix; plus beta_orient_deg + or_r when
    orient values are provided.
    """
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if probs is None:
        w = np.ones_like(ys)
    else:
        w = np.asarray(probs, dtype=np.float64)
        w = np.maximum(w, 1e-12)
    Z = float(w.sum())
    if ys.size == 0 or Z <= 0:
        return {"x": np.nan, "y": np.nan, "beta_pca_deg": np.nan,
                "length_pca": np.nan, "eigratio": np.nan, "n_pix": 0}

    cx = float((xs * w).sum() / Z)
    cy = float((ys * w).sum() / Z)

    dx = xs - cx; dy = ys - cy
    Sxx = float((w * dx * dx).sum() / Z)
    Syy = float((w * dy * dy).sum() / Z)
    Sxy = float((w * dx * dy).sum() / Z)
    cov = np.array([[Syy, Sxy], [Sxy, Sxx]], dtype=np.float64)
    eig_w, V = np.linalg.eigh(cov)  # ascending
    primary = V[:, -1]              # (dy, dx) of principal axis
    proj = dy * primary[0] + dx * primary[1]
    length = float(proj.max() - proj.min()) if proj.size else 0.0
    beta_pca = math.degrees(math.atan2(primary[0], primary[1])) % 180.0
    eigratio = float(eig_w[-1] / max(eig_w[0], 1e-9))

    out = {
        "x": cx, "y": cy,
        "beta_pca_deg": beta_pca,
        "length_pca": length,
        "eigratio": eigratio,
        "n_pix": int(ys.size),
    }

    if sin_vals is not None and cos_vals is not None:
        sin_arr = np.asarray(sin_vals, dtype=np.float64)
        cos_arr = np.asarray(cos_vals, dtype=np.float64)
        sin_m = float((sin_arr * w).sum() / Z)
        cos_m = float((cos_arr * w).sum() / Z)
        out["or_r"] = math.hypot(sin_m, cos_m)
        # orient sin/cos encode 2β, so β = 0.5 * atan2(sin_2b, cos_2b)
        out["beta_orient_deg"] = math.degrees(0.5 * math.atan2(sin_m, cos_m)) % 180.0
    return out


def mf_length_scan(
    diffim: np.ndarray, sigma: float, cy: float, cx: float, beta_deg: float,
    *,
    L_min: int = 5, L_max: int = 100, L_step: int = 2, line_width: int = 2,
) -> tuple[float, float, float]:
    """Scan candidate trail lengths and return the L maximising the line-
    integral SNR through (cy, cx) at angle beta.

    Returns (L_best, snr_best, flux_best). When the streak truly has length
    L*, per-pixel SNR (flux / sigma / sqrt(n_line)) peaks at L = L* — beyond
    that we add noise without signal, below it we miss signal.
    """
    H, W = diffim.shape
    beta_rad = math.radians(beta_deg)
    dy = math.sin(beta_rad); dx = math.cos(beta_rad)
    best_L = float(L_min); best_snr = 0.0; best_flux = 0.0
    for L in range(L_min, L_max + 1, L_step):
        half = int(math.ceil(0.5 * L)) + line_width + 2
        y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half) + 1)
        x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half) + 1)
        Hl = y1 - y0; Wl = x1 - x0
        if Hl <= 0 or Wl <= 0:
            continue
        cy_l = cy - y0; cx_l = cx - x0
        x1l = cx_l - 0.5 * L * dx
        y1l = cy_l - 0.5 * L * dy
        x2l = cx_l + 0.5 * L * dx
        y2l = cy_l + 0.5 * L * dy
        mask = np.zeros((Hl, Wl), np.uint8)
        cv2.line(mask, (int(round(x1l)), int(round(y1l))),
                 (int(round(x2l)), int(round(y2l))),
                 color=1, thickness=line_width)
        n_line = int(mask.sum())
        if n_line < 3:
            continue
        flux = float(diffim[y0:y1, x0:x1][mask > 0].sum())
        snr = flux / max(sigma * math.sqrt(n_line), 1e-6)
        if snr > best_snr:
            best_snr = snr; best_L = float(L); best_flux = flux
    return best_L, best_snr, best_flux


def estimate_params_for_candidate(
    panel_prob: np.ndarray,
    effective_t_low: float,
    candidate_id: int,
    *,
    orient_sin: Optional[np.ndarray] = None,
    orient_cos: Optional[np.ndarray] = None,
) -> dict:
    """Recompute one candidate's connected component on its panel and
    estimate its parameters. Returns the dict from
    `estimate_component_params`."""
    structure = ndi.generate_binary_structure(2, 2)
    labels, n_lab = ndi.label(panel_prob > effective_t_low, structure=structure)
    if candidate_id < 1 or candidate_id > n_lab:
        return {"x": np.nan, "y": np.nan, "beta_pca_deg": np.nan,
                "length_pca": np.nan, "eigratio": np.nan, "n_pix": 0}
    ys, xs = np.where(labels == candidate_id)
    probs = panel_prob[ys, xs]
    sin_vals = None if orient_sin is None else orient_sin[ys, xs].astype(np.float32)
    cos_vals = None if orient_cos is None else orient_cos[ys, xs].astype(np.float32)
    return estimate_component_params(ys, xs, probs,
                                     sin_vals=sin_vals, cos_vals=cos_vals)


# ---------------------------------------------------------------------------
# Catalog matching + per-injection estimates
# ---------------------------------------------------------------------------

def _trail_mask(H: int, W: int, x: float, y: float, beta_deg: float,
                trail_length: float, *, psf_width: int = 40) -> np.ndarray:
    """Same trail-mask construction `objectwise_confusion` uses.
    Returns a bool (H, W) array."""
    half_psf = psf_width // 2
    pad = half_psf + 4
    beta_rad = deg2rad(beta_deg)
    dx = abs(math.cos(beta_rad)) * trail_length
    dy = abs(math.sin(beta_rad)) * trail_length
    x0 = int(max(0, math.floor(x - dx - pad)))
    x1 = int(min(W, math.ceil(x + dx + pad)))
    y0 = int(max(0, math.floor(y - dy - pad)))
    y1 = int(min(H, math.ceil(y + dy + pad)))
    canvas = np.zeros((H, W), dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return canvas.astype(bool)
    roi_h = y1 - y0; roi_w = x1 - x0
    try:
        m = draw_one_line(
            np.zeros((roi_h, roi_w), dtype=np.uint8),
            (x - x0, y - y0), beta_deg, trail_length,
            true_value=1, line_thickness=half_psf,
        )
        canvas[y0:y1, x0:x1] = m.astype(np.uint8)
    except Exception:
        cy = int(y); cx = int(x)
        if 0 <= cy < H and 0 <= cx < W:
            canvas[cy, cx] = 1
    return canvas.astype(bool)


def evaluate_parameter_recovery(
    catalog: pd.DataFrame,
    cand_df: pd.DataFrame,
    panel_probs,
    *,
    orient_sin=None, orient_cos=None,
    diffim_panels=None,
    mf_length: bool = False,
    mf_L_min: int = 5, mf_L_max: int = 100, mf_L_step: int = 2,
    psf_width: int = 40,
) -> pd.DataFrame:
    """Match catalog injections to V2 candidates and estimate parameters.

    For each catalog row, draw the same trail mask `objectwise_confusion`
    uses, find every kept candidate whose component overlaps that mask, and
    pick the candidate with the LARGEST trail-mask overlap (i.e. the cand
    that most plausibly detected this injection). Estimate its (x, y, beta,
    length) from the component pixels.

    Returns a DataFrame with one row per catalog injection. Columns:
        panel_id, injection_idx,                      # identifiers
        x_true, y_true, beta_true_deg, length_true,   # catalog
        SNR, trail_length, mag, stack_detection,      # passed-through
        detected,                                     # bool
        x_est, y_est, beta_pca_deg, length_pca,
        eigratio, n_pix, score_rf,
        dx, dy, dpos,                                 # x_est - x_true, ...
        dbeta_deg, dlength,                           # residuals
        beta_orient_deg, or_r                         # if orient given
    """
    probs_dict   = _to_panel_dict(panel_probs)
    sin_dict     = None if orient_sin is None else _to_panel_dict(orient_sin)
    cos_dict     = None if orient_cos is None else _to_panel_dict(orient_cos)

    use_mf = bool(mf_length) and diffim_panels is not None
    diff_dict = _to_panel_dict(diffim_panels) if diffim_panels is not None else None
    sigma_cache: dict[int, float] = {}

    cat = catalog.copy().reset_index(drop=True)
    cat["injection_idx"] = cat.groupby("image_id").cumcount() + 1
    pid_sample = next(iter(probs_dict.values()))
    H, W = pid_sample.shape

    structure = ndi.generate_binary_structure(2, 2)

    # Pre-compute per-panel component labels at each cand's effective_t_low
    # (in practice effective_t_low is the same for all cands on a panel — we
    # take the median to be safe).
    panel_labels = {}
    for pid, sub in cand_df.groupby("panel_id"):
        pid = int(pid)
        if pid not in probs_dict:
            continue
        eff = float(sub["effective_t_low"].iloc[0])
        labels, _ = ndi.label(probs_dict[pid] > eff, structure=structure)
        panel_labels[pid] = (labels, eff,
                              sub[["candidate_id", "score_rf"]].assign(
                                  cid=sub["candidate_id"].astype(int)))

    rows = []
    for _, row in cat.iterrows():
        pid = int(row["image_id"])
        xt = float(row["x"]); yt = float(row["y"])
        beta_t = float(row["beta"]); Lt = float(row["trail_length"])
        out = {
            "panel_id": pid, "injection_idx": int(row["injection_idx"]),
            "x_true": xt, "y_true": yt,
            "beta_true_deg": beta_t % 180.0, "length_true": Lt,
        }
        for k in ("SNR", "mag", "stack_detection"):
            if k in cat.columns:
                out[k] = row[k]

        if pid not in panel_labels:
            out["detected"] = False
            rows.append(out); continue

        labels, eff, sub_meta = panel_labels[pid]
        trail = _trail_mask(H, W, xt, yt, beta_t, Lt, psf_width=psf_width)
        # Get label values inside the trail mask. Restrict to ones that are
        # kept candidates (in cand_df).
        kept_cids = set(int(c) for c in sub_meta["cid"].tolist())
        intersect = labels[trail]
        intersect = intersect[intersect > 0]
        if intersect.size == 0:
            out["detected"] = False
            rows.append(out); continue
        # Find the candidate (kept) with the largest overlap.
        unique, counts = np.unique(intersect, return_counts=True)
        best_cid = None; best_overlap = 0
        for u, c in zip(unique.tolist(), counts.tolist()):
            if int(u) in kept_cids and int(c) > best_overlap:
                best_cid = int(u); best_overlap = int(c)
        if best_cid is None:
            out["detected"] = False
            rows.append(out); continue

        ys_c, xs_c = np.where(labels == best_cid)
        pp = probs_dict[pid][ys_c, xs_c]
        sn = None if sin_dict is None else sin_dict[pid][ys_c, xs_c].astype(np.float32)
        cs = None if cos_dict is None else cos_dict[pid][ys_c, xs_c].astype(np.float32)
        est = estimate_component_params(ys_c, xs_c, pp, sin_vals=sn, cos_vals=cs)
        # Score of the best candidate (if present in cand_df)
        sub_row = sub_meta[sub_meta["cid"] == best_cid].iloc[0]

        out.update({
            "detected": True,
            "candidate_id": int(best_cid),
            "score_rf": float(sub_row.get("score_rf", float("nan"))),
            "overlap_px": int(best_overlap),
            "x_est": est["x"], "y_est": est["y"],
            "beta_pca_deg": est["beta_pca_deg"],
            "length_pca": est["length_pca"],
            "eigratio": est["eigratio"], "n_pix": est["n_pix"],
        })
        if "beta_orient_deg" in est:
            out["beta_orient_deg"] = est["beta_orient_deg"]
            out["or_r"] = est["or_r"]

        # Residuals
        out["dx"] = est["x"] - xt
        out["dy"] = est["y"] - yt
        out["dpos"] = math.hypot(out["dx"], out["dy"])
        out["dbeta_pca_deg"] = _wrap_beta_diff_deg(est["beta_pca_deg"], beta_t)
        if "beta_orient_deg" in est:
            out["dbeta_orient_deg"] = _wrap_beta_diff_deg(est["beta_orient_deg"], beta_t)
        out["dlength_pca"] = est["length_pca"] - Lt

        # Matched-filter length estimate (optional).
        if use_mf:
            if pid not in sigma_cache:
                sigma_cache[pid] = float(panel_mad_sigma(
                    diff_dict[pid].astype(np.float32, copy=False)))
            beta_for_mf = est.get("beta_orient_deg", est["beta_pca_deg"])
            L_best, snr_best, _ = mf_length_scan(
                diff_dict[pid], sigma_cache[pid], est["y"], est["x"], beta_for_mf,
                L_min=mf_L_min, L_max=mf_L_max, L_step=mf_L_step,
            )
            out["length_mf"] = L_best
            out["mf_length_snr"] = snr_best
            out["dlength_mf"] = L_best - Lt
        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries + plots
# ---------------------------------------------------------------------------

def summarize_residuals(matched: pd.DataFrame) -> pd.DataFrame:
    """Per-axis residual statistics over detected catalog rows.

    Returns a DataFrame with columns: median, mean, std, p10, p90, n.
    Rows: x, y, position, beta_pca, beta_orient (if present), length.
    """
    det = matched[matched["detected"] == True].copy()  # noqa: E712

    def stats(s: pd.Series) -> dict:
        s = s.dropna()
        return {
            "median": float(s.median()) if len(s) else float("nan"),
            "mean":   float(s.mean())   if len(s) else float("nan"),
            "std":    float(s.std())    if len(s) else float("nan"),
            "abs_median": float(s.abs().median()) if len(s) else float("nan"),
            "p10":    float(s.quantile(0.10)) if len(s) else float("nan"),
            "p90":    float(s.quantile(0.90)) if len(s) else float("nan"),
            "n":      int(s.notna().sum()),
        }

    rows = {
        "x [px]":              stats(det["dx"]),
        "y [px]":              stats(det["dy"]),
        "position [px]":       stats(det["dpos"]),
        "beta_pca [deg]":      stats(det["dbeta_pca_deg"]),
        "length_pca [px]":     stats(det["dlength_pca"]),
    }
    if "dbeta_orient_deg" in det.columns:
        rows["beta_orient [deg]"] = stats(det["dbeta_orient_deg"])
    if "dlength_mf" in det.columns:
        rows["length_mf [px]"] = stats(det["dlength_mf"])
    return pd.DataFrame(rows).T


def plot_residuals(matched: pd.DataFrame, *, fig=None, bins: int = 40,
                    title: str = "Parameter recovery"):
    """Histograms + scatter for the major residuals."""
    import matplotlib.pyplot as plt
    det = matched[matched["detected"] == True].copy()  # noqa: E712
    if fig is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    else:
        axes = fig.subplots(2, 3)
    fig.suptitle(f"{title}  (n_detected={len(det)})")

    axes[0, 0].hist(det["dx"], bins=bins, histtype="step", color="C0"); axes[0, 0].set_xlabel("Δx [px]"); axes[0, 0].axvline(0, ls="--", c="k", alpha=0.5)
    axes[0, 1].hist(det["dy"], bins=bins, histtype="step", color="C1"); axes[0, 1].set_xlabel("Δy [px]"); axes[0, 1].axvline(0, ls="--", c="k", alpha=0.5)
    axes[0, 2].hist(det["dpos"], bins=bins, histtype="step", color="C2"); axes[0, 2].set_xlabel("|Δ position| [px]")
    axes[1, 0].hist(det["dbeta_pca_deg"], bins=bins, histtype="step", color="C3", label="PCA")
    if "dbeta_orient_deg" in det.columns:
        axes[1, 0].hist(det["dbeta_orient_deg"], bins=bins, histtype="step", color="C4", label="orient head")
        axes[1, 0].legend()
    axes[1, 0].set_xlabel("Δβ [deg, mod 180]"); axes[1, 0].axvline(0, ls="--", c="k", alpha=0.5)
    axes[1, 1].hist(det["dlength_pca"], bins=bins, histtype="step", color="C5", label="PCA")
    if "dlength_mf" in det.columns:
        axes[1, 1].hist(det["dlength_mf"], bins=bins, histtype="step", color="C6", label="MF scan")
        axes[1, 1].legend()
    axes[1, 1].set_xlabel("Δlength [px]"); axes[1, 1].axvline(0, ls="--", c="k", alpha=0.5)

    est_col = "length_mf" if "length_mf" in det.columns else "length_pca"
    sc = axes[1, 2].scatter(det["length_true"], det[est_col],
                             c=det.get("SNR", np.zeros(len(det))),
                             s=10, cmap="viridis")
    lo = float(min(det["length_true"].min(), det[est_col].min()))
    hi = float(max(det["length_true"].max(), det[est_col].max()))
    axes[1, 2].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    axes[1, 2].set_xlabel("true length"); axes[1, 2].set_ylabel(f"est length ({est_col.split('_',1)[1]})")
    if "SNR" in det.columns:
        fig.colorbar(sc, ax=axes[1, 2], label="SNR")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
