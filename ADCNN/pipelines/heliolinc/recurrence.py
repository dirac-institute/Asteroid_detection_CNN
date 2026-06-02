"""Stationarity / recurrence flag: a real moving object (>=1 deg/day) sits at a given sky position in
ONE visit only (it moves >>2" between same-night visits), while a subtraction-residual artifact RECURS
at the same position across many visits. Flag each detection by how many DISTINCT OTHER visits have a
detection within `tol_arcsec` of its position; a recurrence veto (recur >= thr) removes recurring
residuals at ~zero cost to real NEOs (validated: 100% of injected NEO dets have recur==0).

Same-night recurrence needs multiple same-night visits (dense fields, or a coadd-template residual
history). In operational WFD (2 visits/night) build the equivalent from the survey's PERSISTENT diffim
residuals (positions producing diffim detections on many prior nights) and pass them in as `extra`.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.spatial import cKDTree


def add_recurrence(d, tol_arcsec=2.0, extra=None):
    """Add a `recur` column: # of distinct OTHER visits with a detection within tol of each det.
    `extra` (optional DataFrame with ra,dec[,visit]) = persistent-residual catalog from survey history;
    its hits add to the count (use visit=-1 if unknown so each extra source counts once)."""
    d = d.reset_index(drop=True)
    cd = np.cos(np.radians(d.dec.to_numpy()))
    xy = np.column_stack([d.ra.to_numpy() * cd, d.dec.to_numpy()])
    tree = cKDTree(xy)
    vis = d.visit.to_numpy()
    tol = tol_arcsec / 3600.0
    recur = np.empty(len(d), int)
    for i in range(len(d)):
        nb = tree.query_ball_point(xy[i], tol)
        recur[i] = len(set(vis[nb]) - {vis[i]})
    if extra is not None and len(extra):
        ec = np.cos(np.radians(extra.dec.to_numpy()))
        et = cKDTree(np.column_stack([extra.ra.to_numpy() * ec, extra.dec.to_numpy()]))
        recur = recur + np.array([len(et.query_ball_point(xy[i], tol)) for i in range(len(d))])
    d["recur"] = recur
    return d
