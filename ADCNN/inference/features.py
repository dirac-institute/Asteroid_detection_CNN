"""Stage-2 RF feature extraction for the diffim detector.

For each NN candidate, ``compute_v2_features`` builds the 72-column ``RF_FEATURES_V2``
vector spanning: long matched-filter response (``_add_long_mf``), blob morphology
(``_add_morphology``), panel context (``_add_panel_context``), multi-angle matched
filter (``_add_multiangle_mf``), low-threshold PCA elongation (``_add_low_thr_pca``),
and orientation-head agreement (``_add_orient``). This feature set must stay in
lock-step with the shipped ``models/rf_postproc.pkl`` — do not reorder or drop columns.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ADCNN.inference.candidates import (
    extract_candidates,
    CandidateExtractorConfig,
)
from ADCNN.inference.matched_filter import (
    panel_mad_sigma,
    matched_filter_for_nn_candidates,
)
from ADCNN.utils.helpers import to_panel_dict



LPCA_VARIANTS = ((48, 0.10, 50), (48, 0.10, 80), (32, 0.05, 40), (64, 0.20, 60))


def _lpca_col(prefix: str, win: int, thr: float, suffix: str = "") -> str:
    return f"lpca_{prefix}_w{win}_t{int(round(thr*100)):02d}{suffix}"


RF_FEATURES_V2: tuple[str, ...] = (
    # extract_candidates geometry / prob features
    "area", "bbox_h", "bbox_w", "aspect", "elongation",
    "max_p", "mean_p", "top5_mean_p", "integrated_logit",
    "frac_real_label_overlap",
    # matched filter (PCA, footprint)
    "mf_snr", "mf_n_line", "mf_flux", "mf_length",
    # fixed long-line MF
    "lmf_snr_30", "lmf_flux_30",
    "lmf_snr_60", "lmf_flux_60",
    "lmf_snr_90", "lmf_flux_90",
    "lmf_snr_120", "lmf_flux_120",
    # diffim morphology
    "loc_med_z", "loc_std_z", "loc_max_z", "loc_min_z",
    "loc_skew", "loc_pos_frac", "loc_neg_frac", "loc_sum_z",
    "loc_dipole", "loc_npos", "loc_nneg",
    # panel-context
    "pcount", "p_med_mf_snr", "mf_snr_norm", "area_ratio",
    # multi-angle MF
    "masnr_30", "maflux_30", "meanang_30",
    "masnr_50", "maflux_50", "meanang_50",
    "masnr_80", "maflux_80", "meanang_80",
    # low-threshold PCA MF — flattened in (win, thr, L) order
    f"{_lpca_col('snr', 48, 0.10, '_L50')}", f"{_lpca_col('flux', 48, 0.10, '_L50')}",
    f"{_lpca_col('elong', 48, 0.10)}", f"{_lpca_col('L', 48, 0.10)}",
    f"{_lpca_col('snr', 48, 0.10, '_L80')}", f"{_lpca_col('flux', 48, 0.10, '_L80')}",
    f"{_lpca_col('snr', 32, 0.05, '_L40')}", f"{_lpca_col('flux', 32, 0.05, '_L40')}",
    f"{_lpca_col('elong', 32, 0.05)}", f"{_lpca_col('L', 32, 0.05)}",
    f"{_lpca_col('snr', 64, 0.20, '_L60')}", f"{_lpca_col('flux', 64, 0.20, '_L60')}",
    f"{_lpca_col('elong', 64, 0.20)}", f"{_lpca_col('L', 64, 0.20)}",
    # orient-aware
    "or_r", "or_beta", "or_n_pix",
    "or_agg_max", "or_agg_mean_loose", "or_agg_mean_tight",
    "or_snr_L30", "or_flux_L30",
    "or_snr_L50", "or_flux_L50",
    "or_snr_L80", "or_flux_L80",
)


# ---------------------------------------------------------------------------
# Feature computation (private)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------------------
# PERFORMANCE NOTE (inference throughput bottleneck for the discovery pipeline):
# the matched-filter feature family below (_add_long_mf, _add_multiangle_mf,
# _add_low_thr_pca, _add_orient) rasterises a fresh `cv2.line` into a freshly
# allocated `np.zeros((Hl,Wl))` PER CANDIDATE (and, for multiangle, per angle).
# Allocations + Python loops dominate stage-2 time. Safe, output-preserving wins:
#   1. precompute the oriented line stencils once per (length, angle) and slice
#      with a per-candidate offset instead of re-rasterising;
#   2. replace line rasterisation with scipy.ndimage.map_coordinates sampling
#      along the analytic segment (no raster, no per-call allocation);
#   3. reuse a per-panel scratch buffer (buf[:Hl,:Wl].fill(0)) instead of np.zeros.
# REQUIREMENT: any change here MUST be guarded by a bit-equivalence test — compute
# the full 72-feature matrix on a sample of test_5sigma candidates before and after
# and assert identical — because the RF (models/rf_postproc.pkl) and the verified
# 96.0% recall depend on these feature values being unchanged.
# ------------------------------------------------------------------------------


def _add_long_mf(cand_df, panel_probs, diffims, panel_sigmas, *,
                 line_length: int, line_width: int = 2):
    """Fixed-length MF: PCA from each candidate's component, draw an
    `line_length`-px line through its centroid, integrate flux.
    Adds lmf_snr_{L} and lmf_flux_{L} columns in-place."""
    structure = ndi.generate_binary_structure(2, 2)
    n = len(cand_df)
    snr_out  = np.zeros(n, dtype=np.float32)
    flux_out = np.zeros(n, dtype=np.float32)
    pid_arr = cand_df["panel_id"].to_numpy()
    cy_arr  = cand_df["y_centroid"].to_numpy()
    cx_arr  = cand_df["x_centroid"].to_numpy()
    t_arr   = cand_df["effective_t_low"].to_numpy()

    unique_panels, inverse = np.unique(pid_arr, return_inverse=True)
    for u_idx, pid in enumerate(unique_panels):
        pid = int(pid)
        if pid not in panel_probs:
            continue
        rows_idx = np.where(inverse == u_idx)[0]
        if rows_idx.size == 0:
            continue
        eff = float(np.median(t_arr[rows_idx]))
        pp = panel_probs[pid]
        diff = diffims[pid]
        sigma = panel_sigmas[pid]
        Hp, Wp = pp.shape
        labels_panel, n_lab = ndi.label(pp > eff, structure=structure)
        ys_all, xs_all = np.nonzero(labels_panel)
        if ys_all.size == 0:
            continue
        lab_at = labels_panel[ys_all, xs_all]
        order = np.argsort(lab_at, kind="stable")
        lab_sorted = lab_at[order]
        ys_sorted = ys_all[order]
        xs_sorted = xs_all[order]
        starts = np.searchsorted(lab_sorted, np.arange(1, n_lab + 2))
        for ridx in rows_idx:
            cy = float(cy_arr[ridx]); cx = float(cx_arr[ridx])
            cy_i = max(0, min(Hp - 1, int(round(cy))))
            cx_i = max(0, min(Wp - 1, int(round(cx))))
            lab_id = int(labels_panel[cy_i, cx_i])
            if lab_id == 0 or lab_id > n_lab:
                continue
            s, e = int(starts[lab_id - 1]), int(starts[lab_id])
            ys = ys_sorted[s:e]; xs = xs_sorted[s:e]
            if ys.size < 4:
                continue
            cy_p = float(ys.mean()); cx_p = float(xs.mean())
            coords = np.stack([ys.astype(np.float64) - cy_p,
                               xs.astype(np.float64) - cx_p], axis=1)
            cov = coords.T @ coords / max(len(ys) - 1, 1)
            _, V = np.linalg.eigh(cov)
            primary = V[:, -1]
            beta_rad = math.atan2(primary[0], primary[1])
            half = int(math.ceil(0.5 * line_length)) + int(line_width) + 2
            y0 = max(0, int(cy - half)); y1 = min(Hp, int(cy + half) + 1)
            x0 = max(0, int(cx - half)); x1 = min(Wp, int(cx + half) + 1)
            Hl = y1 - y0; Wl = x1 - x0
            if Hl <= 0 or Wl <= 0:
                continue
            cy_l = cy - y0; cx_l = cx - x0
            dy = math.sin(beta_rad); dx = math.cos(beta_rad)
            x1l = cx_l - 0.5 * line_length * dx
            y1l = cy_l - 0.5 * line_length * dy
            x2l = cx_l + 0.5 * line_length * dx
            y2l = cy_l + 0.5 * line_length * dy
            mask = np.zeros((Hl, Wl), dtype=np.uint8)
            cv2.line(mask, (int(round(x1l)), int(round(y1l))),
                     (int(round(x2l)), int(round(y2l))),
                     color=1, thickness=int(line_width))
            n_line = int(mask.sum())
            if n_line < 3:
                continue
            local_diff = diff[y0:y1, x0:x1]
            S = float(local_diff[mask > 0].sum())
            snr_out[ridx] = S / max(sigma * math.sqrt(n_line), 1e-6)
            flux_out[ridx] = S
    cand_df[f"lmf_snr_{line_length}"]  = snr_out
    cand_df[f"lmf_flux_{line_length}"] = flux_out


def _add_morphology(cand_df, diffims, panel_sigmas, *, crop: int = 32):
    """64x64 (= 2*crop) diffim/sigma statistics around each candidate."""
    n = len(cand_df)
    keys = ("loc_med_z", "loc_std_z", "loc_max_z", "loc_min_z",
            "loc_skew", "loc_pos_frac", "loc_neg_frac", "loc_sum_z",
            "loc_dipole", "loc_npos", "loc_nneg")
    feats = {k: np.zeros(n, dtype=np.float32) for k in keys}
    pid_arr = cand_df["panel_id"].to_numpy()
    cy_arr  = cand_df["y_centroid"].to_numpy()
    cx_arr  = cand_df["x_centroid"].to_numpy()
    for i in range(n):
        pid = int(pid_arr[i])
        if pid not in diffims:
            continue
        panel = diffims[pid]
        sig = panel_sigmas[pid]
        H, W = panel.shape
        cy = int(round(cy_arr[i])); cx = int(round(cx_arr[i]))
        y0 = max(0, cy - crop); y1 = min(H, cy + crop)
        x0 = max(0, cx - crop); x1 = min(W, cx + crop)
        if y1 <= y0 or x1 <= x0:
            continue
        sub = panel[y0:y1, x0:x1] / sig
        if sub.size < 16:
            continue
        feats["loc_med_z"][i]   = float(np.median(sub))
        feats["loc_std_z"][i]   = float(sub.std())
        feats["loc_max_z"][i]   = float(sub.max())
        feats["loc_min_z"][i]   = float(sub.min())
        feats["loc_sum_z"][i]   = float(sub.sum())
        mu = sub.mean(); std = sub.std() + 1e-6
        feats["loc_skew"][i]    = float(((sub - mu) ** 3).mean() / (std ** 3))
        feats["loc_pos_frac"][i]= float((sub > 2.0).mean())
        feats["loc_neg_frac"][i]= float((sub < -2.0).mean())
        feats["loc_dipole"][i]  = float(feats["loc_max_z"][i] + feats["loc_min_z"][i])
        feats["loc_npos"][i]    = float((sub > 2.0).sum())
        feats["loc_nneg"][i]    = float((sub < -2.0).sum())
    for k in keys:
        cand_df[k] = feats[k]


def _add_panel_context(cand_df):
    """Per-panel medians + per-candidate normalised features."""
    ps = cand_df.groupby("panel_id").agg(
        pcount=("candidate_id", "size"),
        p_med_max_p=("max_p", "median"),
        p_med_area=("area", "median"),
        p_med_mf_snr=("mf_snr", "median"),
    ).reset_index()
    merged = cand_df.merge(ps, on="panel_id", how="left", suffixes=("", "_x"))
    cand_df["pcount"]       = merged["pcount"].to_numpy()
    cand_df["p_med_max_p"]  = merged["p_med_max_p"].to_numpy()
    cand_df["p_med_area"]   = merged["p_med_area"].to_numpy()
    cand_df["p_med_mf_snr"] = merged["p_med_mf_snr"].to_numpy()
    cand_df["mf_snr_norm"]  = cand_df["mf_snr"] - cand_df["p_med_mf_snr"]
    cand_df["area_ratio"]   = cand_df["area"] / cand_df["p_med_area"].clip(lower=1)


def _add_multiangle_mf(cand_df, diffims, panel_sigmas, *,
                        lengths: Iterable[int] = (30, 50, 80),
                        n_angles: int = 12, line_width: int = 2):
    """For each candidate, integrate diffim flux along n_angles lines through
    the centroid. Records max-SNR / max-flux / mean-SNR per L."""
    n = len(cand_df)
    pid_arr = cand_df["panel_id"].to_numpy()
    cy_arr  = cand_df["y_centroid"].to_numpy()
    cx_arr  = cand_df["x_centroid"].to_numpy()
    angles = np.linspace(0, math.pi, n_angles, endpoint=False)
    angs_deg = np.degrees(angles)
    for L in lengths:
        snr_arr  = np.zeros(n, np.float32)
        flux_arr = np.zeros(n, np.float32)
        mean_arr = np.zeros(n, np.float32)
        for i in range(n):
            pid = int(pid_arr[i])
            if pid not in diffims:
                continue
            diffim = diffims[pid]; sigma = panel_sigmas[pid]
            cy = float(cy_arr[i]); cx = float(cx_arr[i])
            Hp, Wp = diffim.shape
            half = int(math.ceil(0.5 * L)) + line_width + 2
            y0 = max(0, int(cy - half)); y1 = min(Hp, int(cy + half) + 1)
            x0 = max(0, int(cx - half)); x1 = min(Wp, int(cx + half) + 1)
            Hl = y1 - y0; Wl = x1 - x0
            if Hl <= 0 or Wl <= 0:
                continue
            cy_l = cy - y0; cx_l = cx - x0
            local_diff = diffim[y0:y1, x0:x1]
            snr_vals = np.zeros(n_angles, np.float32)
            flux_vals = np.zeros(n_angles, np.float32)
            for k, ang in enumerate(angles):
                dy = math.sin(ang); dx = math.cos(ang)
                x1l = cx_l - 0.5 * L * dx
                y1l = cy_l - 0.5 * L * dy
                x2l = cx_l + 0.5 * L * dx
                y2l = cy_l + 0.5 * L * dy
                mask = np.zeros((Hl, Wl), np.uint8)
                cv2.line(mask, (int(round(x1l)), int(round(y1l))),
                         (int(round(x2l)), int(round(y2l))),
                         color=1, thickness=line_width)
                nl = int(mask.sum())
                if nl < 3:
                    continue
                S = float(local_diff[mask > 0].sum())
                snr_vals[k]  = S / max(sigma * math.sqrt(nl), 1e-6)
                flux_vals[k] = S
            snr_arr[i]  = float(snr_vals.max())
            flux_arr[i] = float(flux_vals.max())
            mean_arr[i] = float(angs_deg[int(np.argmax(snr_vals))])
        cand_df[f"masnr_{L}"]  = snr_arr
        cand_df[f"maflux_{L}"] = flux_arr
        cand_df[f"meanang_{L}"] = mean_arr


def _add_low_thr_pca(cand_df, panel_probs, diffims, panel_sigmas, *,
                      variants=LPCA_VARIANTS, line_width: int = 2):
    """PCA on pixels with prob > low_thr in a window around the centroid; MF
    along the principal axis at fixed length L."""
    n = len(cand_df)
    pid_arr = cand_df["panel_id"].to_numpy()
    cy_arr  = cand_df["y_centroid"].to_numpy()
    cx_arr  = cand_df["x_centroid"].to_numpy()
    for (win, low_thr, L) in variants:
        snr_a   = np.zeros(n, np.float32)
        flux_a  = np.zeros(n, np.float32)
        elong_a = np.zeros(n, np.float32)
        Lobs_a  = np.zeros(n, np.float32)
        for i in range(n):
            pid = int(pid_arr[i])
            if pid not in panel_probs:
                continue
            pp = panel_probs[pid]
            diffim = diffims[pid]
            sigma = panel_sigmas[pid]
            cy = float(cy_arr[i]); cx = float(cx_arr[i])
            Hp, Wp = pp.shape
            y0 = max(0, int(cy) - win); y1 = min(Hp, int(cy) + win + 1)
            x0 = max(0, int(cx) - win); x1 = min(Wp, int(cx) + win + 1)
            Hl = y1 - y0; Wl = x1 - x0
            if Hl <= 0 or Wl <= 0:
                continue
            pp_local = pp[y0:y1, x0:x1]
            diff_local = diffim[y0:y1, x0:x1]
            cy_l = cy - y0; cx_l = cx - x0
            ys, xs = np.where(pp_local > low_thr)
            if ys.size < 6:
                continue
            cy_p = float(ys.mean()); cx_p = float(xs.mean())
            coords = np.stack([ys.astype(np.float64) - cy_p,
                               xs.astype(np.float64) - cx_p], axis=1)
            cov = coords.T @ coords / max(len(ys) - 1, 1)
            w, V = np.linalg.eigh(cov)
            primary = V[:, -1]
            proj = coords @ primary
            L_obs = float(proj.max() - proj.min())
            elongation = float(w[-1] / max(w[0], 1e-6))
            beta_rad = math.atan2(primary[0], primary[1])
            dy = math.sin(beta_rad); dx = math.cos(beta_rad)
            x1l = cx_l - 0.5 * L * dx
            y1l = cy_l - 0.5 * L * dy
            x2l = cx_l + 0.5 * L * dx
            y2l = cy_l + 0.5 * L * dy
            mask = np.zeros((Hl, Wl), np.uint8)
            cv2.line(mask, (int(round(x1l)), int(round(y1l))),
                     (int(round(x2l)), int(round(y2l))),
                     color=1, thickness=line_width)
            n_line = int(mask.sum())
            Lobs_a[i] = L_obs
            elong_a[i] = elongation
            if n_line < 3:
                continue
            S = float(diff_local[mask > 0].sum())
            snr_a[i]  = S / max(sigma * math.sqrt(n_line), 1e-6)
            flux_a[i] = S
        cand_df[_lpca_col("snr", win, low_thr, f"_L{L}")]  = snr_a
        cand_df[_lpca_col("flux", win, low_thr, f"_L{L}")] = flux_a
        cand_df[_lpca_col("elong", win, low_thr)]          = elong_a
        cand_df[_lpca_col("L", win, low_thr)]              = Lobs_a


def _add_orient(cand_df, panel_probs, diffims, panel_sigmas,
                 sin_maps, cos_maps, agg_maps, *,
                 win: int = 48, low_thr: float = 0.10,
                 lengths: Iterable[int] = (30, 50, 80), line_width: int = 2,
                 orient_mode: str = "pca"):
    """Orientation features (or_beta + MF along β as or_snr_L*/or_flux_L*).

    ``orient_mode``:
      - ``"pca"`` (default): β = footprint principal axis (PCA of the prob>low_thr mask).
        Validated at ~8-10° MAD vs truth — the orientation the MF should integrate along.
      - ``"nnhead"``: β = 0.5·atan2 of the prob-weighted NN sin2β/cos2β head (the original
        behaviour; uncorrelated with truth, r≈0, ~44° MAD). Kept for A/B retraining only.

    ``or_r`` (coherence of the NN sin/cos field) is angle-agnostic and identical in both modes."""
    n = len(cand_df)
    pid_arr = cand_df["panel_id"].to_numpy()
    cy_arr  = cand_df["y_centroid"].to_numpy()
    cx_arr  = cand_df["x_centroid"].to_numpy()
    or_r       = np.zeros(n, np.float32)
    or_beta    = np.zeros(n, np.float32)
    or_n_pix   = np.zeros(n, np.float32)
    or_agg_max = np.zeros(n, np.float32)
    or_agg_lo  = np.zeros(n, np.float32)
    or_agg_ti  = np.zeros(n, np.float32)
    snrs = {L: np.zeros(n, np.float32) for L in lengths}
    flxs = {L: np.zeros(n, np.float32) for L in lengths}
    for i in range(n):
        pid = int(pid_arr[i])
        if pid not in panel_probs:
            continue
        pp = panel_probs[pid]
        sn = sin_maps[pid]
        cs = cos_maps[pid]
        ag = agg_maps[pid]
        diffim = diffims[pid]
        sigma = panel_sigmas[pid]
        cy = float(cy_arr[i]); cx = float(cx_arr[i])
        Hp, Wp = pp.shape
        y0 = max(0, int(cy) - win); y1 = min(Hp, int(cy) + win + 1)
        x0 = max(0, int(cx) - win); x1 = min(Wp, int(cx) + win + 1)
        Hl = y1 - y0; Wl = x1 - x0
        if Hl <= 0 or Wl <= 0:
            continue
        pp_w = pp[y0:y1, x0:x1]
        sn_w = sn[y0:y1, x0:x1].astype(np.float32)
        cs_w = cs[y0:y1, x0:x1].astype(np.float32)
        ag_w = ag[y0:y1, x0:x1].astype(np.float32)
        df_w = diffim[y0:y1, x0:x1]
        cy_l = cy - y0; cx_l = cx - x0
        if ag_w.size:
            or_agg_max[i] = float(_sigmoid(ag_w).max())
        loose_mask = pp_w > 0.05
        if loose_mask.any():
            or_agg_lo[i] = float(_sigmoid(ag_w[loose_mask]).mean())
        mask = pp_w > low_thr
        if mask.sum() < 4:
            continue
        w = pp_w[mask]
        sn_m = float((sn_w[mask] * w).sum() / w.sum())
        cs_m = float((cs_w[mask] * w).sum() / w.sum())
        # or_r = coherence of the NN sin2β/cos2β field (how line-like, angle-agnostic) — keep.
        or_r[i] = math.hypot(sn_m, cs_m)
        if orient_mode == "nnhead":
            # Original (broken) estimator: angle of the NN sin2β/cos2β head. r≈0 vs truth.
            beta = 0.5 * math.atan2(sn_m, cs_m)
        else:
            # Footprint principal-axis angle (PCA of the prob>low_thr mask) — ~8-10° MAD vs
            # truth. Makes or_beta true AND makes or_snr_L*/or_flux_L* integrate flux along the
            # REAL trail axis. Changes these 7 RF features -> RF must be retrained to match.
            ys_m, xs_m = np.nonzero(mask)
            ym = ys_m.astype(np.float64) - ys_m.mean()
            xm = xs_m.astype(np.float64) - xs_m.mean()
            cov = np.array([[float((ym * ym).sum()), float((ym * xm).sum())],
                            [float((ym * xm).sum()), float((xm * xm).sum())]]) / max(len(ys_m) - 1, 1)
            evec = np.linalg.eigh(cov)[1][:, -1]   # principal axis (largest eigenvalue)
            beta = math.atan2(evec[0], evec[1])    # evec[0]=y-comp, evec[1]=x-comp
        or_beta[i] = math.degrees(beta) % 180.0
        or_n_pix[i] = int(mask.sum())
        or_agg_ti[i] = float(_sigmoid(ag_w[mask]).mean())
        dy = math.sin(beta); dx = math.cos(beta)
        for L in lengths:
            x1l = cx_l - 0.5 * L * dx
            y1l = cy_l - 0.5 * L * dy
            x2l = cx_l + 0.5 * L * dx
            y2l = cy_l + 0.5 * L * dy
            lmask = np.zeros((Hl, Wl), np.uint8)
            cv2.line(lmask, (int(round(x1l)), int(round(y1l))),
                     (int(round(x2l)), int(round(y2l))),
                     color=1, thickness=line_width)
            nl = int(lmask.sum())
            if nl < 3:
                continue
            S = float(df_w[lmask > 0].sum())
            snrs[L][i] = S / max(sigma * math.sqrt(nl), 1e-6)
            flxs[L][i] = S
    cand_df["or_r"]       = or_r
    cand_df["or_beta"]    = or_beta
    cand_df["or_n_pix"]   = or_n_pix
    cand_df["or_agg_max"] = or_agg_max
    cand_df["or_agg_mean_loose"] = or_agg_lo
    cand_df["or_agg_mean_tight"] = or_agg_ti
    for L in lengths:
        cand_df[f"or_snr_L{L}"]  = snrs[L]
        cand_df[f"or_flux_L{L}"] = flxs[L]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_v2_features(
    panel_probs,
    diffim_panels,
    orient_sin,
    orient_cos,
    orient_agg,
    *,
    real_labels=None,
    adaptive: bool = True,
    t_low: float = 0.05,
    min_area: int = 4,
    line_width: int = 2,
    pad_length: int = 4,
    gate_pmax: float = 0.0,
    orient_mode: str = "pca",
    verbose: bool = False,
):
    """Extract candidates and compute the full RF feature set.

    All array-like inputs can be either an (N, H, W) array or a {pid: array}
    mapping. orient_sin / orient_cos / orient_agg should be the model's
    auxiliary outputs (see `predict_panel_overlap_3ch_full`).

    Returns (cand_df, panel_probs_dict). panel_probs_dict can be passed to
    `materialize_label_mask_v2` for fast mask reconstruction after filtering.
    """
    panel_probs = to_panel_dict(panel_probs)
    diffims     = to_panel_dict(diffim_panels)
    sin_maps    = to_panel_dict(orient_sin)
    cos_maps    = to_panel_dict(orient_cos)
    agg_maps    = to_panel_dict(orient_agg)
    if real_labels is not None:
        rl_dict = to_panel_dict(real_labels)
    else:
        rl_dict = None

    pids = sorted(panel_probs.keys())
    panel_sigmas = {pid: panel_mad_sigma(diffims[pid].astype(np.float32, copy=False))
                    for pid in pids}

    cfg = CandidateExtractorConfig(t_low=t_low, min_area=min_area,
                                   adaptive_t_low=adaptive)
    cand_dfs = []
    for pid in pids:
        panel = panel_probs[pid].astype(np.float32, copy=False)
        rl = rl_dict[pid] if rl_dict is not None else None
        cand = extract_candidates(panel, real_labels=rl, cfg=cfg, panel_id=pid)
        if len(cand):
            cand_dfs.append(cand)
        panel_probs[pid] = panel
    if not cand_dfs:
        return pd.DataFrame(), panel_probs
    cand_df = pd.concat(cand_dfs, ignore_index=True)

    # Optional cheap gate: only compute the (expensive) matched-filter/orientation features
    # for candidates whose peak NN probability clears `gate_pmax`. Candidates below it are
    # dropped (the RF would score them ~0 anyway); validate no-regression before enabling.
    if gate_pmax > 0.0:
        cand_df = cand_df[cand_df["max_p"] >= gate_pmax].reset_index(drop=True)
        if not len(cand_df):
            return pd.DataFrame(), panel_probs

    # Cast diffims to float32 once.
    diffims = {pid: arr.astype(np.float32, copy=False) for pid, arr in diffims.items()}

    if verbose:
        t = time.time()
    cand_df = matched_filter_for_nn_candidates(
        cand_df, panel_probs=panel_probs, diffim_panels=diffims,
        panel_sigmas=panel_sigmas, line_width=line_width, pad_length=pad_length,
    )
    if verbose:
        print(f"  [v2] matched_filter   {time.time()-t:.1f}s", flush=True)

    for L in (30, 60, 90, 120):
        if verbose:
            t = time.time()
        _add_long_mf(cand_df, panel_probs, diffims, panel_sigmas,
                     line_length=L, line_width=line_width)
        if verbose:
            print(f"  [v2] long_mf L={L:>3}  {time.time()-t:.1f}s", flush=True)

    if verbose:
        t = time.time()
    _add_morphology(cand_df, diffims, panel_sigmas)
    if verbose:
        print(f"  [v2] morphology      {time.time()-t:.1f}s", flush=True)

    if verbose:
        t = time.time()
    _add_panel_context(cand_df)
    if verbose:
        print(f"  [v2] panel-context   {time.time()-t:.1f}s", flush=True)

    if verbose:
        t = time.time()
    _add_multiangle_mf(cand_df, diffims, panel_sigmas)
    if verbose:
        print(f"  [v2] multi-angle MF  {time.time()-t:.1f}s", flush=True)

    if verbose:
        t = time.time()
    _add_low_thr_pca(cand_df, panel_probs, diffims, panel_sigmas)
    if verbose:
        print(f"  [v2] low-thr PCA     {time.time()-t:.1f}s", flush=True)

    if verbose:
        t = time.time()
    _add_orient(cand_df, panel_probs, diffims, panel_sigmas,
                sin_maps, cos_maps, agg_maps, orient_mode=orient_mode)
    if verbose:
        print(f"  [v2] orient          {time.time()-t:.1f}s", flush=True)

    return cand_df, panel_probs


