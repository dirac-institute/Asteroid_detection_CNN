from __future__ import annotations
import os
import math
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Tuple, Iterable, Dict, Any, Optional

# Rubin Butler
from lsst.daf.butler import Butler
import lsst.geom as geom

# ---------- Generic ----------
@contextmanager
def suppress_stdout():
    import sys, io
    old_stdout = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_csv_rows(path: str | Path, rows: Iterable[Dict[str, Any]]):
    path = Path(path)
    rows = list(rows)
    if not rows:
        return
    header = list(rows[0].keys())
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------- Geometry / ephemerides ----------
def vsky_and_pa(ra_rate_cosdec_deg_day: float, dec_rate_deg_day: float) -> Tuple[float, float]:
    """
    From Sorcha-like rates (east=x, north=y):
      vsky  [deg/day] = sqrt(x^2 + y^2)
      PA    [deg E of N] = atan2(x, y) in [0,360)
    """
    x = float(ra_rate_cosdec_deg_day)
    y = float(dec_rate_deg_day)
    vsky = math.hypot(x, y)
    pa = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return vsky, pa

def detectors_covering_point(butler: Butler, visit: int, ra_deg: float, dec_deg: float):
    where = (
        f"instrument='LSSTCam' AND visit={int(visit)} "
        f"AND visit_detector_region.region OVERLAPS POINT({ra_deg:.9f}, {dec_deg:.9f})"
    )
    return list(butler.registry.queryDatasets("calexp", where=where, findFirst=True))

def sky_to_pixel(calexp, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    sp = geom.SpherePoint(geom.Angle(ra_deg, geom.degrees), geom.Angle(dec_deg, geom.degrees))
    x, y = calexp.wcs.skyToPixel(sp)
    return float(x), float(y)

# ---------- Simple line rasterizer (fallback) ----------
def draw_one_line(mask: np.ndarray,
                  origin: Tuple[float, float],
                  angle: float,
                  length: float,
                  true_value: int = 1,
                  line_thickness: int = 5) -> np.ndarray:
    """
    Fallback ‘draw line’ that does not depend on OpenCV. Draws small disks along the line.
    origin = (x0, y0) in pixel coords; angle in deg E of N; length in pixels.
    """
    h, w = mask.shape
    x0, y0 = origin
    # Convert PA (E of N) to image dx,dy (x right, y down)
    theta = math.radians(90.0 - angle)  # image coords
    dx = math.cos(theta)
    dy = math.sin(theta)

    n_steps = max(2, int(length))
    rr = max(1, int(line_thickness // 2))
    for t in np.linspace(0, length, n_steps):
        xc = int(round(x0 + t * dx))
        yc = int(round(y0 - t * dy))
        if 0 <= xc < w and 0 <= yc < h:
            # draw a small disk
            x_min, x_max = max(0, xc - rr), min(w - 1, xc + rr)
            y_min, y_max = max(0, yc - rr), min(h - 1, yc + rr)
            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    if (xx - xc) ** 2 + (yy - yc) ** 2 <= rr ** 2:
                        mask[yy, xx] = true_value
    return mask
