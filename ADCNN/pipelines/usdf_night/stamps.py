"""Pixel measurements for the Rubin stack sources of a night (CNN env).

Each panel with stack sources is opened once with detect_night.read_fits_panel, so the image
preprocessing, header WCS and FITS mask-plane values are the ones the network rows got. Per source:
- morphology at (x, y): catalog._attach_dipole_morphology (is_dipole, m_elong, m_dipole,
  m_neg_ratio, m_neg_blob);
- mask-plane fractions (mask_planes.fractions) along the Naive trail segment: centre = segment
  midpoint, beta = Naive_angle in degrees, length = Naive_length; the 3x3 box at (x, y) when the
  segment is not finite or shorter than 1 px;
- ra0, dec0, ra1, dec1: the Naive segment endpoints through the panel WCS (NaN without a segment).
A panel that cannot be read gives NaN for all of its sources.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ADCNN.pipelines.usdf_night.mask_planes import COLUMNS as MASK_COLUMNS, fractions

MORPH_COLUMNS = ["is_dipole", "m_elong", "m_dipole", "m_neg_ratio", "m_neg_blob"]
ENDPOINT_COLUMNS = ["ra0", "dec0", "ra1", "dec1"]
COLUMNS = ["diaSourceId"] + MORPH_COLUMNS + MASK_COLUMNS + ENDPOINT_COLUMNS
SOURCE_COLUMNS = ["diaSourceId", "x", "y", "trail_x0", "trail_y0", "trail_x1", "trail_y1",
                  "trail_length_px", "trail_angle_img"]


def empty(src: pd.DataFrame) -> pd.DataFrame:
    """All-NaN measurements for the sources of a panel that could not be read."""
    out = pd.DataFrame({c: np.full(len(src), np.nan) for c in COLUMNS})
    out["diaSourceId"] = src["diaSourceId"].to_numpy(np.int64)
    return out


def measure(img, wcs, mask, values, src: pd.DataFrame) -> pd.DataFrame:
    """Morphology, mask fractions and endpoint sky positions for the sources of one panel."""
    from ADCNN.inference.catalog import _attach_dipole_morphology
    out = empty(src)
    x, y = src["x"].to_numpy(np.float64), src["y"].to_numpy(np.float64)
    cand = pd.DataFrame({"x_centroid": x, "y_centroid": y})
    _attach_dipole_morphology(cand, img)
    for c in MORPH_COLUMNS:
        out[c] = cand[c].to_numpy(np.float64)
    x0, y0, x1, y1 = (src[c].to_numpy(np.float64) for c in ("trail_x0", "trail_y0", "trail_x1", "trail_y1"))
    length = src["trail_length_px"].to_numpy(np.float64)
    seg = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    trail = seg & (length >= 1.0)
    cx = np.where(trail, 0.5 * (x0 + x1), x)
    cy = np.where(trail, 0.5 * (y0 + y1), y)
    beta = np.degrees(src["trail_angle_img"].to_numpy(np.float64))
    for c, v in fractions(mask, values, cx, cy, beta, np.where(trail, length, np.nan)).items():
        out[c] = v
    if seg.any():
        p0 = wcs.all_pix2world(np.stack([x0[seg], y0[seg]], 1), 0)
        p1 = wcs.all_pix2world(np.stack([x1[seg], y1[seg]], 1), 0)
        for c, v in zip(ENDPOINT_COLUMNS, (p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1])):
            col = np.full(len(src), np.nan)
            col[seg] = v
            out[c] = col
    return out


def _panel(task):
    """(fits_path, sources) -> (measurements, error message or None)."""
    from ADCNN.inference.detect_night import read_fits_panel
    path, src = task
    try:
        img, wcs, _mjd, mask, values = read_fits_panel(path)
    except Exception as e:
        return empty(src), f"{type(e).__name__}: {e}"
    return measure(img, wcs, mask, values, src), None


def run(stack: pd.DataFrame, manifest: pd.DataFrame, workers: int):
    """Measurements for every row of `stack` (COLUMNS, keyed by diaSourceId) and the list of
    unreadable panels [[visit, detector, error], ...]."""
    paths = {(int(v), int(d)): p for v, d, p in
             manifest[["visit", "detector", "fits_path"]].itertuples(index=False)}
    groups = stack.groupby(["visit", "detector"], sort=True).indices
    keys = [(int(v), int(d)) for v, d in groups]
    tasks = [(paths[k], stack.iloc[idx][SOURCE_COLUMNS]) for k, idx in zip(keys, groups.values())]
    parts, bad, t0 = [], [], time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks))),
                                 mp_context=mp.get_context("spawn")) as ex:
            for i, (k, (df, err)) in enumerate(zip(keys, ex.map(_panel, tasks)), 1):
                parts.append(df)
                if err:
                    bad.append([k[0], k[1], err])
                    print(f"[stamps] {k[0]} {k[1]} unreadable: {err} -> NaN", flush=True)
                if i % 500 == 0 or i == len(tasks):
                    print(f"[stamps] {i}/{len(tasks)} panels, {time.time() - t0:.0f} s", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else empty(stack.iloc[:0])
    return out, bad
