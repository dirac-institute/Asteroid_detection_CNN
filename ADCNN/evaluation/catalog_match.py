"""Catalog-vs-catalog trail-overlap matching for evaluation.

The detector emits a *measured* catalog — one row per detection carrying the measured trail
geometry ``(image_id, x, y, beta, length)`` (see :mod:`ADCNN.inference.catalog`). Evaluation
compares it against a *truth* catalog — one row per injected source ``(image_id, x, y, beta,
trail_length)``.

Each trail is modelled as a line **segment** centred on ``(x, y)``, oriented at ``beta``
degrees (image convention, 0 = +x), of the given length. A truth trail is a **true positive**
if any measured trail's segment passes within ``tol_px`` of it (the "any-pixel-overlap"
criterion); a measured trail that matches no truth is a **false positive**; a truth trail
matched by none is a **false negative**. Matching is per ``image_id`` — a measured trail can
only match truth on its own panel.

``tol_px`` is a fixed matching tolerance chosen in advance (≈ PSF scale); it is never tuned
on the evaluation set. Recall uses the **all-trails** denominator (every truth row counts,
including faint ones below the detection limit). Rows with non-finite geometry never match
(NaN-safe), so they fall through to FN/FP as appropriate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

__all__ = ["match_trail_catalogs", "evaluate_catalog", "match_pairs"]

# Columns every trail catalog must provide (besides its length column).
_REQUIRED_COLUMNS = ("image_id", "x", "y", "beta")
_EPS = 1e-12

CatalogLike = Union[pd.DataFrame, str, Path]


def _require_columns(df: pd.DataFrame, name: str, length_col: str) -> None:
    missing = [c for c in (*_REQUIRED_COLUMNS, length_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{name} catalog is missing required column(s): {missing}")


def _segment_endpoints(df: pd.DataFrame, length_col: str) -> np.ndarray:
    """Endpoints of each trail segment as an ``(N, 2, 2)`` array ``[[x0, y0], [x1, y1]]``.

    The segment is centred on ``(x, y)``, oriented ``beta`` degrees from +x, of the given
    length (so it spans ``±length/2`` about the centre along that direction).
    """
    theta = np.radians(df["beta"].to_numpy(np.float64))
    half = 0.5 * df[length_col].to_numpy(np.float64)
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    dx, dy = np.cos(theta) * half, np.sin(theta) * half
    p0 = np.stack([x - dx, y - dy], axis=-1)
    p1 = np.stack([x + dx, y + dy], axis=-1)
    return np.stack([p0, p1], axis=1)


def _points_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Distance from every point to every segment.

    ``points`` is ``(K, 2)``; ``segments`` is ``(L, 2, 2)``; returns ``(K, L)``.
    """
    a = segments[:, 0, :]                                   # (L, 2)
    ab = segments[:, 1, :] - a                              # (L, 2)
    ap = points[:, None, :] - a[None, :, :]                 # (K, L, 2)
    denom = np.einsum("ld,ld->l", ab, ab)                   # (L,)
    t = np.einsum("kld,ld->kl", ap, ab) / np.where(denom > _EPS, denom, 1.0)
    t = np.clip(t, 0.0, 1.0)                                # (K, L)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]   # (K, L, 2)
    return np.linalg.norm(points[:, None, :] - proj, axis=-1)


def _orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Signed area (×2) of triangle (p, q, r); sign gives the turn direction of p→q→r."""
    return (r[..., 1] - p[..., 1]) * (q[..., 0] - p[..., 0]) \
         - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def _pairwise_segment_distance(seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Minimum distance between every segment in ``seg_a`` and every segment in ``seg_b``.

    ``seg_a`` is ``(N, 2, 2)``, ``seg_b`` is ``(M, 2, 2)``; returns ``(N, M)``. The distance
    is 0 where two segments cross, otherwise the smallest of the four endpoint-to-opposite-
    segment distances (the standard segment-to-segment distance).
    """
    a0, a1 = seg_a[:, 0, :], seg_a[:, 1, :]
    b0, b1 = seg_b[:, 0, :], seg_b[:, 1, :]
    endpoint_dist = np.minimum.reduce([
        _points_to_segments(a0, seg_b),        # (N, M)
        _points_to_segments(a1, seg_b),        # (N, M)
        _points_to_segments(b0, seg_a).T,      # (M, N) -> (N, M)
        _points_to_segments(b1, seg_a).T,
    ])
    A0, A1 = a0[:, None, :], a1[:, None, :]    # (N, 1, 2)
    B0, B1 = b0[None, :, :], b1[None, :, :]    # (1, M, 2)
    crosses = ((_orient(A0, B0, B1) * _orient(A1, B0, B1) <= 0)
               & (_orient(B0, A0, A1) * _orient(B1, A0, A1) <= 0))
    return np.where(crosses, 0.0, endpoint_dist)


def match_trail_catalogs(
    measured: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    tol_px: float = 10.0,
    truth_length_col: str = "trail_length",
    meas_length_col: str = "length",
    flag_col: str = "nn_detected",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Trail-overlap match between a measured detection catalog and a truth catalog.

    Both frames need columns ``image_id, x, y, beta`` plus their length column
    (``trail_length`` for truth, ``length`` for measured). Matching runs per ``image_id``
    and is vectorised over the candidate × truth pairs on each panel.

    Args:
        measured: detection catalog (one row per detection).
        truth: injected-source catalog (one row per trail).
        tol_px: a truth trail is matched if a measured trail's segment lies within this
            distance (pixels) of it. Fixed and pre-chosen — not tuned on the eval set.
        truth_length_col / meas_length_col: trail-length column names in each frame.
        flag_col: name of the boolean "matched" column added to the returned truth frame.

    Returns:
        ``(truth_out, measured_out, counts)`` where ``truth_out`` is ``truth`` plus a bool
        ``flag_col`` (matched by ≥1 detection), ``measured_out`` is ``measured`` plus a bool
        ``matched`` column, and ``counts`` is ``{"TP", "FP", "FN"}`` object-level confusion
        (TP = truth matched, FN = truth unmatched, FP = measured matching no truth).

    Raises:
        ValueError: if either catalog is missing a required column.
    """
    truth = truth.copy().reset_index(drop=True)
    measured = measured.copy().reset_index(drop=True)
    _require_columns(truth, "truth", truth_length_col)
    _require_columns(measured, "measured", meas_length_col)

    truth_matched = np.zeros(len(truth), dtype=bool)
    meas_matched = np.zeros(len(measured), dtype=bool)

    if len(truth) and len(measured):
        meas_by_panel = measured.groupby("image_id").indices  # {id: positional indices}
        for img_id, t_pos in truth.groupby("image_id").indices.items():
            m_pos = meas_by_panel.get(img_id)
            if m_pos is None or len(m_pos) == 0:
                continue
            t_seg = _segment_endpoints(truth.iloc[t_pos], truth_length_col)
            m_seg = _segment_endpoints(measured.iloc[m_pos], meas_length_col)
            within = _pairwise_segment_distance(t_seg, m_seg) <= tol_px  # (nt, nm); NaN -> False
            truth_matched[t_pos] = within.any(axis=1)
            meas_matched[m_pos] = within.any(axis=0)

    truth[flag_col] = truth_matched
    measured["matched"] = meas_matched
    counts = {
        "TP": int(truth_matched.sum()),
        "FN": int((~truth_matched).sum()),
        "FP": int((~meas_matched).sum()),
    }
    return truth, measured, counts


def _as_frame(catalog: CatalogLike) -> pd.DataFrame:
    return pd.read_csv(catalog) if isinstance(catalog, (str, Path)) else catalog


def evaluate_catalog(
    measured: CatalogLike,
    truth: CatalogLike,
    *,
    tol_px: float = 20.0,
    flag_col: str = "nn_detected",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Catalog-based evaluation entry point.

    Matches a measured detection catalog against a truth catalog and returns object-level
    metrics plus the flagged truth catalog (ready for the completeness / histogram plots in
    :mod:`ADCNN.evaluation.plots`, which read the ``flag_col`` and ``stack_detection`` columns).

    Args:
        measured / truth: DataFrames or paths to CSVs.
        tol_px: fixed trail-overlap tolerance (pixels), chosen in advance.
        flag_col: name of the boolean detection flag added to the truth catalog.

    Returns:
        ``(counts, truth_out)`` where ``counts`` has ``TP, FP, FN`` plus derived
        ``recall`` (over the all-trails denominator), ``n_panels`` and ``fp_per_panel``.
    """
    measured_df = _as_frame(measured)
    truth_df = _as_frame(truth)
    truth_out, _, counts = match_trail_catalogs(
        measured_df, truth_df, tol_px=tol_px, flag_col=flag_col,
    )
    n_panels = int(truth_df["image_id"].nunique()) if "image_id" in truth_df.columns else 0
    metrics: dict[str, float] = dict(counts)
    metrics["recall"] = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    metrics["n_panels"] = n_panels
    metrics["fp_per_panel"] = counts["FP"] / max(n_panels, 1)
    return metrics, truth_out


def match_pairs(measured, truth, *, tol_px: float = 20.0,
                truth_length_col: str = "trail_length", meas_length_col: str = "length"):
    """For each truth trail with a match, return the NEAREST measured detection, as a frame
    with the truth row plus ``meas_x/meas_y/meas_beta/meas_length/meas_score`` of that nearest
    detection. Used for parameter-recovery residuals (measured geometry vs truth).

    `measured`/`truth` may be DataFrames or CSV paths. Only matched truths (nearest detection
    within `tol_px`) are returned, one row each.
    """
    measured = _as_frame(measured).reset_index(drop=True)
    truth = _as_frame(truth).reset_index(drop=True)
    if not len(measured) or not len(truth):
        return truth.iloc[0:0].copy()
    meas_by_panel = measured.groupby("image_id").indices
    rows = []
    for img_id, t_pos in truth.groupby("image_id").indices.items():
        m_pos = meas_by_panel.get(img_id)
        if m_pos is None or len(m_pos) == 0:
            continue
        D = _pairwise_segment_distance(_segment_endpoints(truth.iloc[t_pos], truth_length_col),
                                       _segment_endpoints(measured.iloc[m_pos], meas_length_col))
        for ti, t_lbl in enumerate(t_pos):
            j = int(D[ti].argmin())
            if D[ti, j] > tol_px:
                continue
            md = measured.iloc[m_pos[j]]
            row = truth.iloc[t_lbl].to_dict()
            row.update(meas_x=md["x"], meas_y=md["y"], meas_beta=md["beta"],
                       meas_length=md[meas_length_col], meas_score=md.get("score_rf", float("nan")))
            rows.append(row)
    return pd.DataFrame(rows)
