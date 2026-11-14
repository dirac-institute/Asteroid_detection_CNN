"""Finish stage (CNN env): merge the network detections with Rubin's stack sources, flag, write.

  python -m ADCNN.pipelines.usdf_night.join <N> [--workers 32]

Reads N/work/{manifest.csv, stack_sources.parquet, refcat.parquet, adcnn_dets.csv, prep_info.json}
and writes N/detections.csv, N/hl_dets.csv, N/colformat.txt and N/run_info.json (last). Only panels
that detect logged done are joined (all manifest panels once the detect merge removed its .done logs).

Merge, per (visit, detector) in pixels: a network trail is the segment centred on (x, y) at beta
(deg from +x) of length `length`; a stack source is its Naive segment, or the point (x, y) when the
segment is not finite. A positive stack source within MATCH_PX of one or more trails goes to the
nearest one; negative sources never match. A trail keeps the network geometry and score and takes
the stack columns of one representative source (passes the HL rule and in APDB > passes and
reliable > passes > in APDB > reliable > nearest); dia_source_ids lists every absorbed source.
Unabsorbed stack sources are rows of their own (det_id = diaSourceId); their ra/dec is the trail
centre, or the PSF position where Rubin has no trail fit.

hl_dets.csv has one row per physical detection that passes the HL rule (1.25" < length < 12.5",
not negative, stack trail flux > 0): a trail with exactly one passing source writes that source's
values, a trail with two or more passing sources (Rubin split the trail) writes the network values,
a trail with none writes the network values if they pass, a stack-only source its own values.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.pipelines.usdf_night import stamps
from ADCNN.pipelines.usdf_night.mask_planes import COLUMNS as MASK_COLUMNS

MATCH_PX = 10.0
HL_MIN_ARCSEC, HL_MAX_ARCSEC = 1.25, 12.5
RING_ARCSEC, RING_GMAG = 2.5, 21.0
STATIC_ARCSEC, STATIC_GMAG = 3.0, 20.0
NEAR_ARCSEC = 10.0
ZP_NJY = 31.4
OBSCODE = "X05"
MS_PER_DAY = 86400000.0

FLAGS = ["adcnn_detected", "stack_detected", "stack_reliable", "stack_in_apdb"]
STACK_FLOAT = ["reliability", "psf_ra", "psf_dec", "ra_err", "dec_err", "ra_dec_cov",
               "psf_flux_njy", "psf_flux_err_njy", "psf_mag", "trail_flux_njy", "trail_flux_err_njy",
               "trail_mag", "trail_length_err_arcsec", "trail_pa_err_deg"]
GEOMETRY = ["ra", "dec", "x", "y", "length_px", "length_arcsec", "beta_img_deg", "pa_sky_deg",
            "ra0", "dec0", "ra1", "dec1"]
VETOES = ["ring_flag", "refcat_sep_arcsec", "refcat_gmag", "static_flag"]
MORPH = ["is_dipole", "m_elong", "m_dipole", "m_neg_ratio", "m_neg_blob"]
HL_COLUMNS = ["#stringID", "MJD", "RA", "DEC", "Mag", "band", "Obscode", "trailpa", "traillength",
              "RA_err", "DEC_err", "RA_DEC_cov", "sigmag", "trailpa_err", "traillength_err", "visit",
              "detector", "det_id", "dia_source_ids", "adcnn_detected", "stack_detected",
              "stack_reliable", "stack_in_apdb", "score", "reliability", "ring_flag", "static_flag",
              "is_dipole", "rubin_is_dipole", "art_frac", "n_stack", "dia_object_id", "ss_object_id"]
COLFORMAT = ("IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"
             "TRAILPACOL 8\nTRAILLENCOL 9\nSIGMAGCOL 13\nSIGALONGCOL 15\n")
DONE_TAG = "adcnn_dets"


def detection_columns(pixel_flags) -> list[str]:
    return (["det_id", "visit", "detector", "band", "mjd_tai"] + FLAGS + ["is_negative"] + GEOMETRY
            + ["score", "mf_snr", "nn_pmax", "stack_dia_source_id", "n_stack", "match_sep_px",
               "dia_source_ids"]
            + STACK_FLOAT + ["rubin_is_dipole", "glint_trail", "trail_flag_edge"] + list(pixel_flags)
            + VETOES + MORPH + MASK_COLUMNS)


def nullable_flags(pixel_flags) -> list[str]:
    """0/1 columns that are NaN for some row types (written as integers or nan)."""
    return (["is_negative", "rubin_is_dipole", "glint_trail", "trail_flag_edge"] + list(pixel_flags)
            + ["ring_flag", "static_flag", "is_dipole"])


# ------------------------------------------------------------------ geometry

_EPS = 1e-12


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


def segment_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum distance (N, M) between the segments a (N, 2, 2) and b (M, 2, 2); a zero-length
    segment is a point. Endpoint-to-segment distances plus a strict crossing test (a `<= 0` test
    would also call two points, or collinear disjoint segments, crossing)."""
    a0, a1, b0, b1 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    d = np.minimum.reduce([_points_to_segments(a0, b), _points_to_segments(a1, b),
                           _points_to_segments(b0, a).T, _points_to_segments(b1, a).T])
    A0, A1, B0, B1 = a0[:, None], a1[:, None], b0[None], b1[None]
    cross = ((_orient(A0, B0, B1) * _orient(A1, B0, B1) < 0)
             & (_orient(B0, A0, A1) * _orient(B1, A0, A1) < 0))
    return np.where(cross, 0.0, d)


def network_segments(net: pd.DataFrame) -> np.ndarray:
    """(N, 2, 2) segments centred on (x, y) along beta (deg) of length `length`; a point where the
    length or angle is not finite."""
    L = net["length"].to_numpy(np.float64)
    b = np.radians(net["beta"].to_numpy(np.float64))
    ok = np.isfinite(L) & np.isfinite(b)
    hx, hy = np.where(ok, 0.5 * L * np.cos(b), 0.0), np.where(ok, 0.5 * L * np.sin(b), 0.0)
    x, y = net["x"].to_numpy(np.float64), net["y"].to_numpy(np.float64)
    return np.stack([np.stack([x - hx, y - hy], -1), np.stack([x + hx, y + hy], -1)], 1)


def stack_segments(st: pd.DataFrame) -> np.ndarray:
    """(M, 2, 2) Naive segments, or the point (x, y) where the segment is not finite."""
    p = np.stack([st[c].to_numpy(np.float64) for c in ("trail_x0", "trail_y0", "trail_x1", "trail_y1")])
    ok = np.isfinite(p).all(axis=0)
    x, y = st["x"].to_numpy(np.float64), st["y"].to_numpy(np.float64)
    e0 = np.stack([np.where(ok, p[0], x), np.where(ok, p[1], y)], -1)
    e1 = np.stack([np.where(ok, p[2], x), np.where(ok, p[3], y)], -1)
    return np.stack([e0, e1], 1)


def sky_sep_arcsec(ra0, dec0, ra1, dec1):
    a0, d0, a1, d1 = (np.radians(np.asarray(v, np.float64)) for v in (ra0, dec0, ra1, dec1))
    h = np.sin(0.5 * (d1 - d0)) ** 2 + np.cos(d0) * np.cos(d1) * np.sin(0.5 * (a1 - a0)) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))) * 3600.0


def bearing_deg(ra0, dec0, ra1, dec1):
    """Position angle (deg E of N) of point 1 seen from point 0, in (-180, 180]."""
    a0, d0, a1, d1 = (np.radians(np.asarray(v, np.float64)) for v in (ra0, dec0, ra1, dec1))
    da = a1 - a0
    b = np.degrees(np.arctan2(np.sin(da) * np.cos(d1),
                              np.cos(d0) * np.sin(d1) - np.sin(d0) * np.cos(d1) * np.cos(da)))
    return np.where(b <= -180.0, b + 360.0, b)


def fold_pa(pa):
    """Trail PA folded into [90, 270) (a trail has no direction)."""
    return np.mod(np.asarray(pa, np.float64) - 90.0, 180.0) + 90.0


def mag(flux_njy):
    """AB mag from nJy; NaN for flux <= 0 or NaN."""
    f = np.asarray(flux_njy, np.float64)
    pos = f > 0
    return np.where(pos, ZP_NJY - 2.5 * np.log10(np.where(pos, f, 1.0)), np.nan)


def passes_hl(length_arcsec, is_negative, flux_njy=None):
    """HL rule: finite length in (1.25", 12.5"), not negative, and (stack values) trail flux > 0."""
    L = np.asarray(length_arcsec, np.float64)
    ok = np.isfinite(L) & (L > HL_MIN_ARCSEC) & (L < HL_MAX_ARCSEC) & (np.asarray(is_negative) != 1)
    if flux_njy is not None:
        ok &= np.asarray(flux_njy, np.float64) > 0
    return ok


# ------------------------------------------------------------------ merge

def network_ids(visit, detector, x, y, score) -> np.ndarray:
    """det_id = (visit % 10**11) * 10**8 + detector * 10**5 + index (int64), index = ordinal of the
    row within its panel after a stable sort by (y, x, score)."""
    visit, detector = np.asarray(visit, np.int64), np.asarray(detector, np.int64)
    order = np.lexsort((np.asarray(score, np.float64), np.asarray(x, np.float64),
                        np.asarray(y, np.float64), detector, visit))
    v, d, n = visit[order], detector[order], len(order)
    new = np.ones(n, bool)
    new[1:] = (v[1:] != v[:-1]) | (d[1:] != d[:-1])
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    index = np.empty(n, np.int64)
    index[order] = np.arange(n) - start
    assert (index < 100000).all(), "more than 100000 network rows in one panel"
    assert ((detector >= 0) & (detector < 1000)).all(), "detector outside 0..999"
    return (visit % 10**11) * 10**8 + detector * 10**5 + index


def match(net: pd.DataFrame, st: pd.DataFrame, tol=MATCH_PX):
    """(trail, sep) per stack row: positional index of the nearest network trail within `tol` px on
    the same panel (-1 = none) and the segment distance to it. Negative sources never match."""
    trail, sep = np.full(len(st), -1, np.int64), np.full(len(st), np.nan)
    if not len(net) or not len(st):
        return trail, sep
    nseg, sseg = network_segments(net), stack_segments(st)
    positive = st["is_negative"].to_numpy() != 1
    panels = net.groupby(["visit", "detector"]).indices
    for key, si in st.groupby(["visit", "detector"]).indices.items():
        ni = panels.get(key)
        si = si[positive[si]]
        if ni is None or not len(si):
            continue
        d = segment_distance(sseg[si], nseg[ni])
        d = np.where(d <= tol, d, np.inf)
        j = np.argmin(d, axis=1)
        best = d[np.arange(len(si)), j]
        ok = np.isfinite(best)
        trail[si[ok]], sep[si[ok]] = ni[j[ok]], best[ok]
    return trail, sep


def absorb(trail, sep, ids, passes, in_detector, in_apdb, n_net):
    """Per network trail: representative stack row (-1 = none), n_stack, n_pass, any reliable,
    any in APDB and the ';'-joined absorbed diaSourceIds ('0' = none)."""
    a = np.flatnonzero(trail >= 0)
    t = trail[a]
    n_stack = np.bincount(t, minlength=n_net).astype(np.int64)
    n_pass = np.bincount(t, weights=passes[a], minlength=n_net).astype(np.int64)
    reliable = np.bincount(t, weights=in_detector[a], minlength=n_net) > 0
    apdb = np.bincount(t, weights=in_apdb[a], minlength=n_net) > 0
    tier = np.select([passes & (in_apdb == 1), passes & (in_detector == 1), passes,
                      in_apdb == 1, in_detector == 1], [0, 1, 2, 3, 4], 5)
    order = a[np.lexsort((ids[a], sep[a], tier[a], t))]
    first = np.ones(len(order), bool)
    first[1:] = trail[order][1:] != trail[order][:-1]
    rep = np.full(n_net, -1, np.int64)
    rep[trail[order[first]]] = order[first]
    joined = np.full(n_net, "0", dtype=object)
    if len(a):
        s = pd.Series(ids[a]).groupby(t).agg(lambda v: ";".join(map(str, np.sort(v.to_numpy()))))
        joined[s.index.to_numpy()] = s.to_numpy()
    return rep, n_stack, n_pass, reliable, apdb, joined


def _take(values, idx):
    """values[idx] as float64, NaN where idx < 0."""
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = np.asarray(values, np.float64)[idx[ok]]
    return out


def vetoes(ra, dec, refcat: pd.DataFrame) -> dict:
    """Bright-star ring (G < 21 within 2.5"), nearest G < 21 source within 10" (sep, gmag) and
    static source (G < 20 within 3") for each position; NaN where the position is not finite."""
    from scipy.spatial import cKDTree
    ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
    n, ok = len(ra), np.isfinite(ra) & np.isfinite(dec)
    out = {"ring_flag": np.full(n, np.nan), "refcat_sep_arcsec": np.full(n, np.nan),
           "refcat_gmag": np.full(n, np.nan), "static_flag": np.full(n, np.nan)}

    def unit(r, d):
        r, d = np.radians(r), np.radians(d)
        return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)

    def nearest(cat, radius):
        """(sep arcsec, index) of the nearest source of `cat` within radius; inf where none."""
        if not len(cat):
            return np.full(ok.sum(), np.inf), np.zeros(ok.sum(), np.int64)
        chord = 2.0 * np.sin(np.radians(radius / 3600.0) / 2.0)
        d, i = cKDTree(unit(cat["ra"].to_numpy(), cat["dec"].to_numpy())).query(
            unit(ra[ok], dec[ok]), k=1, distance_upper_bound=chord)
        return np.where(np.isfinite(d), np.degrees(2.0 * np.arcsin(np.minimum(d, 2.0) / 2.0)) * 3600.0,
                        np.inf), np.minimum(i, len(cat) - 1)

    g = refcat["gmag"].to_numpy(np.float64)
    bright = refcat[np.isfinite(g) & (g < RING_GMAG)].reset_index(drop=True)
    sep, i = nearest(bright, NEAR_ARCSEC)
    found = np.isfinite(sep)
    out["ring_flag"][ok] = sep < RING_ARCSEC
    out["refcat_sep_arcsec"][ok] = np.where(found, sep, np.nan)
    out["refcat_gmag"][ok] = np.where(found, bright["gmag"].to_numpy(np.float64)[i] if len(bright)
                                      else np.nan, np.nan)
    sep20, _ = nearest(refcat[np.isfinite(g) & (g < STATIC_GMAG)].reset_index(drop=True), STATIC_ARCSEC)
    out["static_flag"][ok] = sep20 < STATIC_ARCSEC
    return out


def build(man: pd.DataFrame, net: pd.DataFrame, stack: pd.DataFrame, meas: pd.DataFrame,
          refcat: pd.DataFrame):
    """(detections, hl rows, counts) for the covered panels. `meas` = stamps measurements of the
    stack rows (stamps.COLUMNS, keyed by diaSourceId)."""
    pixel_flags = [c for c in stack.columns if c.startswith("pixelFlags_")]
    panel = man[["visit", "detector", "band", "mjd_tai"]]
    net = net.reset_index(drop=True).merge(panel, on=["visit", "detector"], how="left",
                                           validate="many_to_one", suffixes=("_net", ""))
    st = stack.reset_index(drop=True).merge(panel, on=["visit", "detector"], how="left",
                                            validate="many_to_one", suffixes=("_stack", ""))
    assert net["mjd_tai"].notna().all() and st["mjd_tai"].notna().all(), "row outside the manifest"
    dt_net = np.abs(net["mjd"] - net["mjd_tai"]).to_numpy(np.float64) * MS_PER_DAY
    dt_st = np.abs(st["mjd_tai_stack"] - st["mjd_tai"]).to_numpy(np.float64) * MS_PER_DAY
    assert (dt_net < 1.0).all(), "network mjd differs from the visit midpoint by >= 1 ms"
    assert (dt_st < 1.0).all(), "stack mjd_tai differs from the manifest by >= 1 ms"
    ids = st["diaSourceId"].to_numpy(np.int64)
    st["psf_mag"], st["trail_mag"] = mag(st["psf_flux_njy"]), mag(st["trail_flux_njy"])
    st_pass = passes_hl(st["trail_length_arcsec"], st["is_negative"], st["trail_flux_njy"])
    in_det, in_apdb = st["in_detector"].to_numpy(np.int64), st["in_apdb"].to_numpy(np.int64)
    meas = meas.set_index("diaSourceId").reindex(ids)

    net["det_id"] = network_ids(net["visit"], net["detector"], net["x"], net["y"], net["score"])
    trail, sep = match(net, st)
    rep, n_stack, n_pass, reliable, apdb, joined = absorb(trail, sep, ids, st_pass, in_det, in_apdb,
                                                          len(net))
    rep_id = np.zeros(len(net), np.int64)
    rep_id[rep >= 0] = ids[rep[rep >= 0]]
    L_net = sky_sep_arcsec(net["ra0"], net["dec0"], net["ra1"], net["dec1"])
    net_pass = passes_hl(L_net, np.zeros(len(net)))
    nr = {"det_id": net["det_id"].to_numpy(np.int64), "visit": net["visit"].to_numpy(np.int64),
          "detector": net["detector"].to_numpy(np.int64), "band": net["band"].to_numpy(object),
          "mjd_tai": net["mjd_tai"].to_numpy(np.float64),
          "adcnn_detected": np.ones(len(net), np.int8), "stack_detected": (n_stack > 0).astype(np.int8),
          "stack_reliable": reliable.astype(np.int8), "stack_in_apdb": apdb.astype(np.int8),
          "is_negative": _take(st["is_negative"], rep),
          "ra": net["ra"], "dec": net["dec"], "x": net["x"], "y": net["y"], "length_px": net["length"],
          "length_arcsec": L_net, "beta_img_deg": net["beta"],
          "pa_sky_deg": bearing_deg(net["ra0"], net["dec0"], net["ra1"], net["dec1"]),
          "ra0": net["ra0"], "dec0": net["dec0"], "ra1": net["ra1"], "dec1": net["dec1"],
          "score": net["score"], "mf_snr": net["mf_snr"], "nn_pmax": net["nn_pmax"],
          "stack_dia_source_id": rep_id, "n_stack": n_stack, "match_sep_px": _take(sep, rep), "dia_source_ids": joined,
          "_src": rep, "_hl": np.select([n_pass == 1, n_pass >= 2, net_pass], [1, 2, 2], 0),
          "_split": n_pass >= 2}
    for c in STACK_FLOAT + ["rubin_is_dipole", "glint_trail", "trail_flag_edge"] + pixel_flags:
        nr[c] = _take(st[c], rep)
    for c in MORPH + MASK_COLUMNS:
        nr[c] = net[c].to_numpy(np.float64)

    o = np.flatnonzero(trail < 0)
    s = st.iloc[o]
    has_trail = np.isfinite(s["trail_ra"].to_numpy()) & np.isfinite(s["trail_dec"].to_numpy())
    sr = {"det_id": ids[o], "visit": s["visit"].to_numpy(np.int64), "detector": s["detector"].to_numpy(np.int64),
          "band": s["band"].to_numpy(object), "mjd_tai": s["mjd_tai"].to_numpy(np.float64),
          "adcnn_detected": np.zeros(len(o), np.int8), "stack_detected": np.ones(len(o), np.int8),
          "stack_reliable": in_det[o].astype(np.int8), "stack_in_apdb": in_apdb[o].astype(np.int8),
          "is_negative": s["is_negative"].to_numpy(np.float64),
          "ra": np.where(has_trail, s["trail_ra"], s["psf_ra"]),
          "dec": np.where(has_trail, s["trail_dec"], s["psf_dec"]),
          "x": s["x"], "y": s["y"], "length_px": s["trail_length_px"],
          "length_arcsec": s["trail_length_arcsec"],
          "beta_img_deg": np.degrees(s["trail_angle_img"].to_numpy(np.float64)),
          "pa_sky_deg": s["trail_pa_deg"],
          "score": np.full(len(o), np.nan), "mf_snr": np.full(len(o), np.nan),
          "nn_pmax": np.full(len(o), np.nan), "stack_dia_source_id": ids[o],
          "n_stack": np.ones(len(o), np.int64), "match_sep_px": np.full(len(o), np.nan),
          "dia_source_ids": ids[o].astype(str).astype(object),
          "_src": o, "_hl": np.where(st_pass[o], 1, 0), "_split": np.zeros(len(o), bool)}
    for c in stamps.ENDPOINT_COLUMNS + MORPH + MASK_COLUMNS:
        sr[c] = meas[c].to_numpy(np.float64)[o]
    for c in STACK_FLOAT + ["rubin_is_dipole", "glint_trail", "trail_flag_edge"] + pixel_flags:
        sr[c] = s[c].to_numpy(np.float64)

    cols = detection_columns(pixel_flags)
    frames = [pd.DataFrame({k: np.asarray(v) for k, v in d.items()}) for d in (nr, sr)]
    det = pd.concat(frames, ignore_index=True)
    for c, v in vetoes(det["ra"], det["dec"], refcat).items():
        det[c] = v
    for c in nullable_flags(pixel_flags):
        det[c] = det[c].astype("Int8")
    for c in ["det_id", "visit", "detector", "stack_dia_source_id", "n_stack"]:
        det[c] = det[c].astype(np.int64)
    det = det.sort_values(["mjd_tai", "visit", "detector", "det_id"], kind="stable").reset_index(drop=True)
    assert det["det_id"].is_unique, "duplicate det_id"

    hl = hl_rows(det, st)
    counts = {"rows": len(det), "network_rows": len(net), "stack_only_rows": len(o),
              "network_mjd_max_diff_ms": float(dt_net.max()) if len(dt_net) else None,
              "stack_mjd_max_diff_ms": float(dt_st.max()) if len(dt_st) else None,
              "stack_only_negative": int((s["is_negative"] == 1).sum()),
              "stack_only_psf_position": int((~has_trail).sum()),
              "matched_trails": int((n_stack > 0).sum()), "absorbed_sources": int((trail >= 0).sum()),
              "split_trails": int((n_pass >= 2).sum()),
              "split_trails_network_fails_hl": int(((n_pass >= 2) & ~net_pass).sum()),
              "rows_by_flags": [dict(zip(FLAGS, map(int, k)), rows=int(v)) for k, v in
                                det.groupby(FLAGS).size().items()],
              "hl_rows": len(hl),
              "hl_trail_stack_values": int(((det["_hl"] == 1) & (det["adcnn_detected"] == 1)).sum()),
              "hl_split_network_values": int(det["_split"].sum()),
              "hl_trail_network_values": int(((det["_hl"] == 2) & ~det["_split"]).sum()),
              "hl_stack_only": int(((det["_hl"] == 1) & (det["adcnn_detected"] == 0)).sum())}
    return det[cols + ["_hl", "_src"]], hl, counts


def hl_rows(det: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Ian's 17 columns + ours for the detections marked for hl (_hl 1 = stack values of row _src,
    2 = network values), sorted by MJD then #stringID."""
    row = ["visit", "detector", "det_id", "dia_source_ids"] + FLAGS + [
        "score", "reliability", "ring_flag", "static_flag", "is_dipole", "rubin_is_dipole", "art_frac",
        "n_stack"]
    a = det[det["_hl"] == 1]
    s = st.iloc[a["_src"].to_numpy()]
    sv = pd.DataFrame({"#stringID": s["diaSourceId"].to_numpy(np.int64), "MJD": a["mjd_tai"].to_numpy(),
                       "RA": s["trail_ra"].to_numpy(), "DEC": s["trail_dec"].to_numpy(),
                       "Mag": s["trail_mag"].to_numpy(), "band": a["band"].to_numpy(), "Obscode": OBSCODE,
                       "trailpa": fold_pa(s["trail_pa_deg"]), "traillength": s["trail_length_arcsec"].to_numpy(),
                       "RA_err": s["ra_err"].to_numpy(), "DEC_err": s["dec_err"].to_numpy(),
                       "RA_DEC_cov": s["ra_dec_cov"].to_numpy(),
                       "sigmag": 1.0857 * s["trail_flux_err_njy"].to_numpy() / s["trail_flux_njy"].to_numpy(),
                       "trailpa_err": s["trail_pa_err_deg"].to_numpy(),
                       "traillength_err": s["trail_length_err_arcsec"].to_numpy(),
                       **{c: a[c].to_numpy() for c in row},
                       "dia_object_id": s["dia_object_id"].to_numpy(np.int64),
                       "ss_object_id": s["ss_object_id"].to_numpy(np.int64)})
    b = det[det["_hl"] == 2]
    nan = np.full(len(b), np.nan)
    nv = pd.DataFrame({"#stringID": b["det_id"].to_numpy(np.int64), "MJD": b["mjd_tai"].to_numpy(),
                       "RA": b["ra"].to_numpy(), "DEC": b["dec"].to_numpy(), "Mag": nan,
                       "band": b["band"].to_numpy(), "Obscode": OBSCODE, "trailpa": fold_pa(b["pa_sky_deg"]),
                       "traillength": b["length_arcsec"].to_numpy(), "RA_err": nan, "DEC_err": nan,
                       "RA_DEC_cov": nan, "sigmag": nan, "trailpa_err": nan, "traillength_err": nan,
                       **{c: b[c].to_numpy() for c in row},
                       "dia_object_id": np.zeros(len(b), np.int64), "ss_object_id": np.zeros(len(b), np.int64)})
    hl = pd.concat([sv, nv], ignore_index=True)[HL_COLUMNS]
    for c in ["ring_flag", "static_flag", "is_dipole", "rubin_is_dipole"]:
        hl[c] = hl[c].astype("Int8")
    for c in FLAGS:
        hl[c] = hl[c].astype(np.int8)
    hl = hl.sort_values(["MJD", "#stringID"], kind="stable").reset_index(drop=True)
    assert np.isfinite(hl["MJD"].to_numpy(np.float64)).all(), "hl row without MJD"
    assert hl["#stringID"].is_unique, "duplicate hl #stringID"
    assert (hl["#stringID"].astype(str).str.len() <= 19).all(), "hl #stringID longer than 19 chars"
    return hl


# ------------------------------------------------------------------ inputs, outputs

def covered_panels(work: Path, man: pd.DataFrame) -> set:
    """Manifest panels detect logged done; all of them when no .done log is left (100% merge)."""
    from ADCNN.inference.detect_night import _done_panels
    expected = set(zip(man["visit"].astype(np.int64), man["detector"].astype(np.int64)))
    if not list(work.glob(f"_shard_{DONE_TAG}_*.csv.done")):
        return expected
    return expected & _done_panels(work, DONE_TAG)


def unreadable(work: Path, missing: set) -> list:
    """Last logged error of each missing panel from detect's <out>.unreadable log."""
    path, last = work / f"{DONE_TAG}.csv.unreadable", {}
    if path.exists():
        for r in csv.reader(open(path, newline="")):
            if len(r) >= 3 and (int(r[0]), int(r[1])) in missing:
                last[(int(r[0]), int(r[1]))] = r[2]
    return [[v, d, e] for (v, d), e in sorted(last.items())]


def write_csv(df: pd.DataFrame, path: Path, check=None) -> None:
    """Atomic CSV write; `check(tmp_path)` must pass before the file takes its final name."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, na_rep="nan")
    if check:
        check(tmp)
    os.replace(tmp, path)


def write_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def check_hl_text(path: Path) -> None:
    """Header exactly HL_COLUMNS, every line newline-terminated, no empty field."""
    text = path.read_text()
    lines = text.split("\n")
    assert text.endswith("\n") and lines[0] == ",".join(HL_COLUMNS), "hl header"
    assert all("" not in ln.split(",") for ln in lines[:-1]), "hl row with an empty field"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def commit(repo: Path) -> str:
    r = subprocess.run(["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "uncommitted"


def job_record(job_id) -> dict:
    """sacct summary of one Slurm job (state, elapsed, allocation, nodes)."""
    rec = {"job_id": str(job_id)}
    try:
        r = subprocess.run(["sacct", "-X", "-n", "-P", "-j", str(job_id),
                            "--format=JobID,State,Elapsed,AllocTRES,NodeList"],
                           capture_output=True, text=True, timeout=60)
        f = r.stdout.strip().splitlines()[0].split("|")
    except (OSError, subprocess.SubprocessError, IndexError):
        return rec
    rec.update(zip(["job_id", "state", "elapsed", "alloc_tres", "nodes"], f))
    return rec


def stage_summary(stages: dict):
    """(GPU {type, count} of the last detect job that recorded its nvidia-smi names, or None;
    {stage: wall seconds summed over its finished jobs}) from state.json 'stages'. The S3DF sacct
    AllocTRES carries no GPU type (gres/gpu=N)."""
    names = [r["gpus"] for r in stages.get("detect", []) if r.get("gpus")]
    gpu = {"type": ",".join(sorted(set(names[-1]))), "count": len(names[-1])} if names else None
    wall = {}
    for stage, recs in stages.items():
        secs = [r["seconds"] for r in recs if r.get("seconds") is not None]
        if secs:
            wall[stage] = sum(secs)
    return gpu, wall


def run_info(N: Path, man, covered, prep, state, stamps_bad, counts, seconds) -> dict:
    from ADCNN.config import ACTIVE
    from ADCNN.inference.mf_trail_length import MF_K
    repo = Path(__file__).resolve().parents[3]
    expected = set(zip(man["visit"].astype(np.int64), man["detector"].astype(np.int64)))
    missing = expected - covered
    pm, st, rc = prep.get("manifest", {}), prep.get("stack_tables", {}), prep.get("refcat", {})
    jobs = {k: [job_record(j) for j in (v if isinstance(v, list) else [v])]
            for k, v in state.get("jobs", {}).items() if v}
    gpu, wall = stage_summary(state.get("stages") or {})
    return {"night": state.get("night", N.name[:8]), "status": "partial" if missing else "complete",
            "commit": commit(repo), "user": state.get("user", getpass.getuser()),
            "models": {"seg": {"path": str(ACTIVE.seg_model), "md5": md5(ACTIVE.seg_model)},
                       "cnn": {"path": str(ACTIVE.cnn_model), "md5": md5(ACTIVE.cnn_model)}},
            "floor": state.get("floor"), "mf_k": MF_K,
            **{k: pm.get(k) for k in ("repo", "collection", "dataset_type", "runs", "lsst_weekly")},
            "refcat": rc.get("refcat"), "panels_expected": len(expected), "panels_done": len(covered),
            "panels_missing": [[int(v), int(d)] for v, d in sorted(missing)],
            "panels_unreadable": unreadable(N / "work", missing),
            "panels_without_stack_tables": st.get("panels_without_unfiltered"),
            "stack_validation_failed_columns": st.get("validation_failed_columns"),
            "panels_stamps_unreadable": stamps_bad, "gpu": gpu,
            "jobs": jobs, "wall_seconds": {**wall, "join": round(seconds, 1)}, "counts": counts}


def load(N: Path):
    """(covered manifest, network, stack, refcat, prep_info, covered panels, full manifest); the
    network and stack rows are those of the covered panels."""
    work = N / "work"
    man = pd.read_csv(work / "manifest.csv")
    net = pd.read_csv(work / "adcnn_dets.csv", low_memory=False)
    stack = pd.read_parquet(work / "stack_sources.parquet")
    refcat = pd.read_parquet(work / "refcat.parquet")
    prep = json.loads((work / "prep_info.json").read_text())
    for df in (man, net, stack):
        df["visit"], df["detector"] = df["visit"].astype(np.int64), df["detector"].astype(np.int64)
    for c in net.columns.drop(["detid", "visit", "detector", "band", "obscode", "is_dipole"]):
        net[c] = net[c].astype(np.float64)
    net["is_dipole"] = net["is_dipole"].astype(bool).astype(np.float64)
    covered = covered_panels(work, man)

    def keep(df):
        k = np.fromiter(((v, d) in covered for v, d in zip(df["visit"], df["detector"])), bool, len(df))
        return df[k].reset_index(drop=True)
    assert len(keep(net)) == len(net), "network rows on panels that are not done"
    return keep(man), net, keep(stack), refcat, prep, covered, man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="join", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("night_dir", type=Path)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    t0, N = time.time(), a.night_dir
    man, net, stack, refcat, prep, covered, man_all = load(N)
    state_path = N / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    print(f"[join] {len(covered)}/{len(man_all)} panels, {len(net):,} network rows, "
          f"{len(stack):,} stack sources", flush=True)
    meas, stamps_bad = stamps.run(stack, man, a.workers)
    det, hl, counts = build(man, net, stack, meas, refcat)
    write_csv(det.drop(columns=["_hl", "_src"]), N / "detections.csv")
    write_csv(hl, N / "hl_dets.csv", check=check_hl_text)
    write_text(COLFORMAT, N / "colformat.txt")
    info = run_info(N, man_all, covered, prep, state, stamps_bad, counts, time.time() - t0)
    write_text(json.dumps(info, indent=1) + "\n", N / "run_info.json")
    print(f"[join] {counts['rows']:,} detections ({counts['network_rows']:,} network, "
          f"{counts['stack_only_rows']:,} stack-only), {counts['split_trails']:,} split trails, "
          f"{counts['hl_rows']:,} hl rows -> {N} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
