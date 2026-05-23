"""Symmetric matched-filter post-processor for diffim candidates.

Applied to BOTH the NN's candidate list and the LSST classical detector's
candidate list (from `real_labels`), using identical math. The point is a
fair recall-vs-FPs/panel comparison after the same filter.

Per-candidate matched filter:
  1. Estimate the candidate's footprint (set of pixels).
  2. Fit the principal axis of the footprint (PCA on pixel coords).
  3. Define a thin line of length L (footprint extent along principal axis)
     and width W (default 2 px) through the centroid at the principal angle.
  4. S = sum(diffim flux along line)
     N = panel_MAD_sigma * sqrt(n_line_pixels)
     SNR_mf = S / N
  5. Keep candidate iff SNR_mf >= threshold.

The diffim is what the matched filter "sees" — exactly the same data LSST's
detector looks at. The MAD sigma is per-panel (same convention as the diffim dataset
NN's normalization), so noise is calibrated identically for both detectors.

The filter rewards candidates that have positive integrated diffim flux
along a coherent line — what a real moving-object trail produces. It
penalises noise blobs and isolated point sources (their length is small
so n_line_pixels is small and SNR is bounded).

For the comparison to be defensible, we use the same matched-filter SNR
threshold sweep on the NN and LSST candidate sets. The resulting points
go on the same recall-vs-FPs/panel plot.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def panel_mad_sigma(arr: np.ndarray) -> float:
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(good)) + 1e-8)


def _footprint_principal_axis(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (cy, cx, beta_rad, length_px). beta_rad is angle of the
    principal axis in image coords; length_px is the extent of the footprint
    along that axis."""
    n = len(ys)
    cy = float(ys.mean()); cx = float(xs.mean())
    if n < 4:
        return cy, cx, 0.0, max(1.0, float(max(ys.max() - ys.min(), xs.max() - xs.min(), 0) + 1))
    coords = np.stack([ys.astype(np.float64) - cy, xs.astype(np.float64) - cx], axis=1)
    cov = coords.T @ coords / max(n - 1, 1)
    # Principal axis = eigenvector with largest eigenvalue.
    w, V = np.linalg.eigh(cov)
    primary = V[:, -1]
    proj = coords @ primary
    L = float(proj.max() - proj.min())
    # primary[0] is the y-component, primary[1] is the x-component.
    beta_rad = math.atan2(primary[0], primary[1])
    return cy, cx, beta_rad, max(L, 3.0)


def matched_filter_from_coords(
    diffim_panel: np.ndarray,
    panel_sigma: float,
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    line_width: int = 2,
    pad_length: int = 4,
) -> tuple[float, int, float, float]:
    """Compute matched-filter SNR from raw (ys, xs) coords (no global mask).

    Works in a local bbox to avoid panel-sized allocations.
    """
    if len(ys) < 3:
        return 0.0, 0, 0.0, 0.0
    cy, cx, beta_rad, L = _footprint_principal_axis(ys, xs)
    L_eff = float(L + pad_length)
    half = int(math.ceil(0.5 * L_eff)) + int(line_width) + 2

    H, W = diffim_panel.shape
    y0 = max(0, int(math.floor(cy - half)))
    y1 = min(H, int(math.ceil(cy + half)) + 1)
    x0 = max(0, int(math.floor(cx - half)))
    x1 = min(W, int(math.ceil(cx + half)) + 1)
    Hl = y1 - y0
    Wl = x1 - x0
    if Hl <= 0 or Wl <= 0:
        return 0.0, 0, 0.0, L_eff

    cy_l = cy - y0
    cx_l = cx - x0
    dy = math.sin(beta_rad)
    dx = math.cos(beta_rad)
    x1l = cx_l - 0.5 * L_eff * dx; y1l = cy_l - 0.5 * L_eff * dy
    x2l = cx_l + 0.5 * L_eff * dx; y2l = cy_l + 0.5 * L_eff * dy

    local_mask = np.zeros((Hl, Wl), dtype=np.uint8)
    cv2.line(local_mask,
             (int(round(x1l)), int(round(y1l))),
             (int(round(x2l)), int(round(y2l))),
             color=1, thickness=int(line_width))
    n_line = int(local_mask.sum())
    if n_line < 3:
        return 0.0, n_line, 0.0, L_eff
    local_diffim = diffim_panel[y0:y1, x0:x1]
    S = float(local_diffim[local_mask > 0].sum())
    Nnoise = panel_sigma * math.sqrt(n_line)
    return S / max(Nnoise, 1e-6), n_line, S, L_eff


def matched_filter_for_nn_candidates(
    cand_df: pd.DataFrame,
    panel_probs: dict[int, np.ndarray],
    diffim_panels: dict[int, np.ndarray],
    panel_sigmas: dict[int, float],
    *,
    line_width: int = 2,
    pad_length: int = 4,
) -> pd.DataFrame:
    """Add columns mf_snr, mf_n_line, mf_flux, mf_length to `cand_df`.

    Optimised: label the whole panel ONCE (not per candidate). All
    candidates on a panel share the same effective_t_low (the adaptive
    formula depends only on panel statistics), so a single threshold +
    label call per panel covers them all. This is ~hundreds of times
    faster than the previous per-candidate implementation.
    """
    n = len(cand_df)
    snr_out = np.zeros(n, dtype=np.float32)
    n_out = np.zeros(n, dtype=np.int32)
    flux_out = np.zeros(n, dtype=np.float32)
    len_out = np.zeros(n, dtype=np.float32)

    # Group candidates by panel.
    pid_arr = cand_df["panel_id"].to_numpy()
    cy_arr = cand_df["y_centroid"].to_numpy()
    cx_arr = cand_df["x_centroid"].to_numpy()
    t_arr = cand_df["effective_t_low"].to_numpy()

    unique_panels, inverse = np.unique(pid_arr, return_inverse=True)
    structure = ndi.generate_binary_structure(2, 2)

    for u_idx, pid in enumerate(unique_panels):
        pid = int(pid)
        if pid not in panel_probs:
            continue
        rows_idx = np.where(inverse == u_idx)[0]
        if rows_idx.size == 0:
            continue
        # All candidates on this panel share the same effective_t_low
        # (adaptive formula is panel-level). Sanity check, but otherwise
        # use the first value.
        t_low_vals = t_arr[rows_idx]
        t_low_panel = float(np.median(t_low_vals))
        pp = panel_probs[pid]
        diffim = diffim_panels[pid]
        sigma = panel_sigmas[pid]

        bin_panel = pp > t_low_panel
        if not bin_panel.any():
            continue
        labels_panel, n_lab = ndi.label(bin_panel, structure=structure)

        # Precompute footprint coords per label so each candidate just looks
        # up its label_id once. Avoids re-scanning labels_panel per candidate.
        ys_all, xs_all = np.nonzero(labels_panel)
        if ys_all.size == 0:
            continue
        lab_at = labels_panel[ys_all, xs_all]
        # Sort by label id once; index lookups via np.searchsorted.
        order = np.argsort(lab_at, kind="stable")
        lab_sorted = lab_at[order]
        ys_sorted = ys_all[order]
        xs_sorted = xs_all[order]
        starts = np.searchsorted(lab_sorted, np.arange(1, n_lab + 2))

        for ridx in rows_idx:
            cy = int(round(float(cy_arr[ridx])))
            cx = int(round(float(cx_arr[ridx])))
            cy = max(0, min(labels_panel.shape[0] - 1, cy))
            cx = max(0, min(labels_panel.shape[1] - 1, cx))
            lab_id = int(labels_panel[cy, cx])
            if lab_id == 0 or lab_id > n_lab:
                continue
            s, e = int(starts[lab_id - 1]), int(starts[lab_id])
            ys = ys_sorted[s:e]
            xs = xs_sorted[s:e]
            if ys.size < 3:
                continue
            snr, nline, flux, Leff = matched_filter_from_coords(
                diffim, sigma, ys, xs,
                line_width=line_width, pad_length=pad_length,
            )
            snr_out[ridx] = snr
            n_out[ridx] = nline
            flux_out[ridx] = flux
            len_out[ridx] = Leff

    out = cand_df.copy()
    out["mf_snr"] = snr_out
    out["mf_n_line"] = n_out
    out["mf_flux"] = flux_out
    out["mf_length"] = len_out
    return out


