"""Catalog-vs-catalog trail-overlap matching for evaluation.

The detector emits a *measured* catalog (one trail per detection: x, y, beta, length;
see ADCNN.inference.catalog). Evaluation compares it against a *truth* catalog (one trail
per injected source: x, y, beta, trail_length). Each trail is treated as a line segment
centred on (x, y), oriented at `beta` degrees (image convention, 0 = +x), of the given
length. A truth trail is a true positive if any measured trail's segment passes within
`tol_px` of it; measured trails matching no truth are false positives; truth trails matched
by none are false negatives.

`tol_px` is a fixed matching tolerance chosen in advance (≈ PSF scale) — never tuned on the
evaluation set. Matching is per `image_id` (a measured trail can only match truth on its
own panel).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def _segments(df, length_col):
    """(M, 2, 2) array of segment endpoints [[x0,y0],[x1,y1]] for each row of `df`."""
    th = np.radians(df["beta"].to_numpy(np.float64))
    h = 0.5 * df[length_col].to_numpy(np.float64)
    x = df["x"].to_numpy(np.float64); y = df["y"].to_numpy(np.float64)
    dx, dy = np.cos(th) * h, np.sin(th) * h
    p0 = np.stack([x - dx, y - dy], axis=1)
    p1 = np.stack([x + dx, y + dy], axis=1)
    return np.stack([p0, p1], axis=1)


def _pt_seg_dist(p, a, b):
    """Distance from point p to segment a-b (all length-2 arrays)."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab))))


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_seg_dist(p0, p1, q0, q1):
    """Minimum distance between segments p0-p1 and q0-q1 (0 if they intersect)."""
    if (_ccw(p0, q0, q1) * _ccw(p1, q0, q1) <= 0) and (_ccw(q0, p0, p1) * _ccw(q1, p0, p1) <= 0):
        return 0.0  # proper or touching intersection
    return min(_pt_seg_dist(p0, q0, q1), _pt_seg_dist(p1, q0, q1),
               _pt_seg_dist(q0, p0, p1), _pt_seg_dist(q1, p0, p1))


def match_trail_catalogs(measured: pd.DataFrame, truth: pd.DataFrame, *,
                         tol_px: float = 10.0,
                         truth_length_col: str = "trail_length",
                         meas_length_col: str = "length",
                         flag_col: str = "nn_detected"):
    """Trail-overlap match between a measured detection catalog and a truth catalog.

    Both frames need columns ``image_id, x, y, beta`` plus their length column
    (``trail_length`` for truth, ``length`` for measured). Returns:

      truth_out    : copy of `truth` with a bool `flag_col` (True = matched by a detection)
      measured_out : copy of `measured` with a bool ``matched`` column
      counts       : {"TP", "FP", "FN"} object-level confusion (TP = truth matched,
                     FN = truth unmatched, FP = measured matching no truth)
    """
    truth = truth.copy().reset_index(drop=True)
    measured = measured.copy().reset_index(drop=True)
    truth[flag_col] = False
    measured["matched"] = False
    if len(measured) and len(truth):
        m_by = {k: v for k, v in measured.groupby("image_id").groups.items()}
        for img_id, t_idx in truth.groupby("image_id").groups.items():
            m_idx = m_by.get(img_id)
            if m_idx is None:
                continue
            tseg = _segments(truth.loc[t_idx], truth_length_col)
            mseg = _segments(measured.loc[m_idx], meas_length_col)
            for ti, ti_lbl in enumerate(t_idx):
                for mi, mi_lbl in enumerate(m_idx):
                    if _seg_seg_dist(tseg[ti, 0], tseg[ti, 1], mseg[mi, 0], mseg[mi, 1]) <= tol_px:
                        truth.at[ti_lbl, flag_col] = True
                        measured.at[mi_lbl, "matched"] = True
    counts = {
        "TP": int(truth[flag_col].sum()),
        "FN": int((~truth[flag_col]).sum()),
        "FP": int((~measured["matched"]).sum()),
    }
    return truth, measured, counts


def evaluate_catalog(measured, truth, *, tol_px: float = 20.0,
                     flag_col: str = "nn_detected"):
    """Catalog-based evaluation entry point: match a measured detection catalog against a
    truth catalog and return object-level metrics + the flagged truth catalog.

    `measured` and `truth` may be DataFrames or paths to CSVs. Returns:
      counts : {"TP","FP","FN","recall","n_panels","fp_per_panel"} (recall over ALL truth
               trails — the all-trails denominator)
      truth_out : truth catalog with a bool `flag_col` (matched by a detection), ready for
                  the completeness/histogram plots in ADCNN.evaluation.plots.
    """
    if isinstance(measured, (str, Path)):
        measured = pd.read_csv(measured)
    if isinstance(truth, (str, Path)):
        truth = pd.read_csv(truth)
    truth_out, _, counts = match_trail_catalogs(measured, truth, tol_px=tol_px, flag_col=flag_col)
    n_panels = int(truth["image_id"].nunique()) if "image_id" in truth.columns else 0
    counts["recall"] = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    counts["n_panels"] = n_panels
    counts["fp_per_panel"] = counts["FP"] / max(n_panels, 1)
    return counts, truth_out
