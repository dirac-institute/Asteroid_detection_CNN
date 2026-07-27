#!/usr/bin/env python3
"""Extract per-alert diffim cutouts ONCE, panel-ordered, into a compact cache (npz).

Why this exists: `alert_report.py` renders rank-ordered, reading panels through a 4-panel LRU.
That is fine for a dozen alerts and catastrophic for a 10k-alert nightly stream -- consecutive
ranks land on unrelated panels, so the LRU thrashes and every alert re-reads a ~200 MB FITS
(over S3 for embargo nights). Here the loop is inverted: iterate PANELS, and for each panel cut
every alert epoch that lands on it. FITS I/O becomes O(panels) (~hundreds/night) instead of
O(alerts) (~10k), and all downstream rendering (contact sheets, per-alert pair figures, re-sorts
after a threshold change) reads the small cache instead of pixels.

Two cutout kinds per alert:
  ZOOM  -- one K x K stamp per epoch, centred on that epoch's own detection. Shows the source.
  WIDE  -- ONE downsampled stamp per alert, cut from the first epoch's panel and sized to contain
           EVERY member position plus margin, so the actual motion is visible in one frame. At
           1-8 deg/day over a ~20 min gap the members are 250-2000 px apart, far outside a zoom
           stamp, so the connection between epochs can only be seen at this scale.
           Member pixel positions within the wide stamp are stored alongside it, so the renderer
           can outline each detection and draw the line joining them without touching pixels again.

Cache layout (npz, float16 stamps):
  stamps (Nz,K,K) / alert (Nz,) / epoch (Nz,) / visit / detector / ok    -- the zooms
  wide (Nw,KW,KW) / wide_alert (Nw,) / wide_pos (Nw,MAXEP,2) / wide_apx (Nw,) / wide_ok (Nw,)
      wide_pos = member (x,y) in wide-stamp pixels (NaN = that member is outside the frame)
      wide_apx = arcsec per wide-stamp pixel, for a scale bar

Usage:
  python -m ADCNN.qa.alert_cutouts --alerts alerts.jsonl --dets adcnn_dets_masked.csv \
      --out report/cutouts.npz [--stamp-px 96] [--wide-px 220] [--workers 8]
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd

CLIP_SIGMA = 20.0
MAXEP = 4                 # member slots stored per alert in the wide frame
PIXSCALE = 0.2            # arcsec/px (LSSTCam)


def _mad_sigma(a):
    from ADCNN.data.preprocessing import diffim_mad_sigma
    return float(diffim_mad_sigma(a))


def _cut(img, x, y, k):
    """k x k cutout centred on (x, y), zero-padded at the panel edge."""
    H, W = img.shape
    h = k // 2
    x0, y0 = int(round(x)) - h, int(round(y)) - h
    out = np.zeros((k, k), np.float32)
    sx0, sx1 = max(x0, 0), min(x0 + k, W)
    sy0, sy1 = max(y0, 0), min(y0 + k, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def _match_endpoints(alerts, dets_path):
    """Attach each alert epoch's MEASURED trail endpoints (ra0/dec0, ra1/dec1) from the dets
    catalog. The alert packet only carries beta, which is an angle in that epoch's OWN panel
    frame -- not comparable across visits whose rotator differs, and useless for drawing the
    trail into another epoch's frame. The sky endpoints are frame-independent, so both trails
    can be projected exactly into the wide view and compared against the motion direction."""
    from scipy.spatial import cKDTree
    cols = ["visit", "detector", "ra", "dec", "ra0", "dec0", "ra1", "dec1"]
    d = pd.read_csv(dets_path, usecols=lambda c: c in cols)
    if not {"ra0", "dec0", "ra1", "dec1"}.issubset(d.columns):
        print("[cutouts] WARN dets lack ra0/dec0/ra1/dec1 -- trail outlines unavailable", flush=True)
        return {}
    trees = {}
    for (v, det), g in d.groupby(["visit", "detector"]):
        cd = np.cos(np.radians(g.dec.to_numpy()))
        trees[(int(v), int(det))] = (cKDTree(np.column_stack([g.ra.to_numpy() * cd,
                                                             g.dec.to_numpy()])), g)
    tol = 1.0 / 3600.0
    ends, miss = {}, 0
    for ai, al in enumerate(alerts):
        for ei, ep in enumerate(al["epochs"]):
            t = trees.get((int(ep["visit"]), int(ep["detector"])))
            if t is None:
                miss += 1; continue
            tree, g = t
            dec = float(ep["dec"])
            dist, idx = tree.query([float(ep["ra"]) * np.cos(np.radians(dec)), dec],
                                   distance_upper_bound=tol)
            if not np.isfinite(dist):
                miss += 1; continue
            r = g.iloc[int(idx)]
            ends[(ai, ei)] = (float(r.ra0), float(r.dec0), float(r.ra1), float(r.dec1))
    if miss:
        print(f"[cutouts] {miss} epochs without a matched detection (no trail outline)", flush=True)
    return ends


def _wide_cut(img, xy, out_px, margin_px=60):
    """Box containing every (x,y) in `xy` + margin, block-averaged to out_px x out_px.
    Returns (stamp, positions-in-stamp-coords, source-px-per-stamp-px)."""
    xs = np.array([p[0] for p in xy], float); ys = np.array([p[1] for p in xy], float)
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) + 2 * margin_px
    span = float(max(span, 4 * margin_px))
    half = span / 2.0
    x0, y0 = cx - half, cy - half
    H, W = img.shape
    # sample the box on an out_px grid (nearest source pixel); handles off-panel as zeros
    gx = np.clip(np.round(x0 + (np.arange(out_px) + 0.5) * span / out_px).astype(int), -1, W)
    gy = np.clip(np.round(y0 + (np.arange(out_px) + 0.5) * span / out_px).astype(int), -1, H)
    inx = (gx >= 0) & (gx < W); iny = (gy >= 0) & (gy < H)
    stamp = np.zeros((out_px, out_px), np.float32)
    if inx.any() and iny.any():
        stamp[np.ix_(iny, inx)] = img[np.ix_(gy[iny], gx[inx])]
    pos = np.full((MAXEP, 2), np.nan, np.float32)
    for i, (px, py) in enumerate(xy[:MAXEP]):
        pos[i] = ((px - x0) * out_px / span, (py - y0) * out_px / span)
    return stamp, pos, span / out_px, (x0, y0, span)


def _panel_job(args):
    """One panel: all zoom cuts on it, plus any wide cuts anchored to it."""
    path, zooms, wides, k, kw = args
    try:
        from astropy.wcs import WCS
        from ADCNN.inference.diffim_io import open_diffim
        with open_diffim(path, memmap=False) as h:
            img = np.nan_to_num(h[1].data.astype(np.float32))
            wcs = WCS(h[1].header)
        sig = _mad_sigma(img) or 1.0
        zout, wout = [], []
        if zooms:
            xy = wcs.all_world2pix(np.array([[z[2], z[3]] for z in zooms], float), 0)
            for (ai, ei, _r, _d, ends), (x, y) in zip(zooms, xy):
                st = np.clip(_cut(img, x, y, k) / sig, -CLIP_SIGMA, CLIP_SIGMA)
                e = np.full((2, 2), np.nan, np.float32)
                if ends is not None:
                    ee = wcs.all_world2pix(np.array([[ends[0], ends[1]], [ends[2], ends[3]]], float), 0)
                    h = k // 2
                    e[0] = (ee[0][0] - (round(x) - h), ee[0][1] - (round(y) - h))
                    e[1] = (ee[1][0] - (round(x) - h), ee[1][1] - (round(y) - h))
                zout.append((ai, ei, st.astype(np.float16),
                             bool(0 <= x < img.shape[1] and 0 <= y < img.shape[0]), e))
        for (ai, sky, allends) in wides:
            xy = wcs.all_world2pix(np.array(sky, float), 0)
            stamp, pos, apx, _wide_origin = _wide_cut(img, xy, kw)
            # project EVERY member's measured trail endpoints into THIS (the wide) frame, so the
            # two trails and the motion vector are all expressed in one common frame and their
            # orientations can honestly be compared by eye.
            wends = np.full((MAXEP, 2, 2), np.nan, np.float32)
            for i, en in enumerate(allends[:MAXEP]):
                if en is None:
                    continue
                ee = wcs.all_world2pix(np.array([[en[0], en[1]], [en[2], en[3]]], float), 0)
                for j in (0, 1):
                    wends[i, j] = ((ee[j][0] - _wide_origin[0]) * kw / _wide_origin[2],
                                   (ee[j][1] - _wide_origin[1]) * kw / _wide_origin[2])
            wout.append((ai, np.clip(stamp / sig, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float16),
                         pos, np.float32(apx * PIXSCALE), True, wends))
        return zout, wout
    except Exception as e:
        print(f"[cutouts] WARN panel unreadable ({e}): {path}", flush=True)
        return ([(ai, ei, np.zeros((k, k), np.float16), False,
                  np.full((2, 2), np.nan, np.float32)) for (ai, ei, _r, _d, _e) in zooms],
                [(ai, np.zeros((kw, kw), np.float16), np.full((MAXEP, 2), np.nan, np.float32),
                  np.float32(np.nan), False, np.full((MAXEP, 2, 2), np.nan, np.float32))
                 for (ai, _s, _e) in wides])


def build(alerts_path, dets_path, out_npz, stamp_px=96, wide_px=220, workers=8, limit=None):
    alerts = [json.loads(l) for l in open(alerts_path)] if os.path.getsize(alerts_path) else []
    if limit:
        alerts = alerts[:limit]
    if not alerts:
        print("[cutouts] 0 alerts -- nothing to do", flush=True)
        return None

    pmap = pd.read_csv(dets_path, usecols=["visit", "detector", "fits_path"]).drop_duplicates(
        ["visit", "detector"])
    panel_of = {(int(v), int(d)): p for v, d, p in zip(pmap.visit, pmap.detector, pmap.fits_path)}

    ends = _match_endpoints(alerts, dets_path)
    zoom_by_panel, wide_by_panel = defaultdict(list), defaultdict(list)
    n_missing = 0
    for ai, al in enumerate(alerts):
        eps = al["epochs"]
        for ei, ep in enumerate(eps):
            p = panel_of.get((int(ep["visit"]), int(ep["detector"])))
            if p is None:
                n_missing += 1
                continue
            zoom_by_panel[p].append((ai, ei, float(ep["ra"]), float(ep["dec"]),
                                     ends.get((ai, ei))))
        p0 = panel_of.get((int(eps[0]["visit"]), int(eps[0]["detector"])))
        if p0 is not None:
            wide_by_panel[p0].append((ai, [(float(e["ra"]), float(e["dec"])) for e in eps],
                                      [ends.get((ai, j)) for j in range(len(eps))]))

    panels = sorted(set(zoom_by_panel) | set(wide_by_panel))
    jobs = [(p, zoom_by_panel.get(p, []), wide_by_panel.get(p, []), stamp_px, wide_px)
            for p in panels]
    print(f"[cutouts] {len(alerts)} alerts -> {sum(len(v) for v in zoom_by_panel.values())} zooms "
          f"+ {sum(len(v) for v in wide_by_panel.values())} wide, over {len(panels)} panels "
          f"({n_missing} epochs with no panel path)", flush=True)

    zres, wres = [], []
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for n, (zo, wo) in enumerate(ex.map(_panel_job, jobs), 1):
                zres.extend(zo); wres.extend(wo)
                if n % 25 == 0 or n == len(jobs):
                    print(f"[cutouts] {n}/{len(jobs)} panels", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            zo, wo = _panel_job(j)
            zres.extend(zo); wres.extend(wo)

    zres.sort(key=lambda r: (r[0], r[1]))
    wres.sort(key=lambda r: r[0])
    K, KW = stamp_px, wide_px
    stamps = np.stack([r[2] for r in zres]) if zres else np.zeros((0, K, K), np.float16)
    wide = np.stack([r[1] for r in wres]) if wres else np.zeros((0, KW, KW), np.float16)
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        stamps=stamps,
        alert=np.array([r[0] for r in zres], np.int32),
        epoch=np.array([r[1] for r in zres], np.int8),
        ok=np.array([r[3] for r in zres], bool),
        zoom_ends=(np.stack([r[4] for r in zres]) if zres else np.zeros((0, 2, 2), np.float32)),
        visit=np.array([int(alerts[r[0]]["epochs"][r[1]]["visit"]) for r in zres], np.int64),
        detector=np.array([int(alerts[r[0]]["epochs"][r[1]]["detector"]) for r in zres], np.int64),
        wide=wide,
        wide_alert=np.array([r[0] for r in wres], np.int32),
        wide_pos=(np.stack([r[2] for r in wres]) if wres else np.zeros((0, MAXEP, 2), np.float32)),
        wide_apx=np.array([r[3] for r in wres], np.float32),
        wide_ok=np.array([r[4] for r in wres], bool),
        wide_ends=(np.stack([r[5] for r in wres]) if wres
                   else np.zeros((0, MAXEP, 2, 2), np.float32)))
    with open(os.path.splitext(out_npz)[0] + "_meta.json", "w") as f:
        json.dump(dict(n_alerts=len(alerts), n_zoom=len(zres), n_wide=len(wres), stamp_px=K,
                       wide_px=KW, clip_sigma=CLIP_SIGMA, alerts=os.path.abspath(alerts_path),
                       dets=os.path.abspath(dets_path), n_panels=len(panels),
                       n_missing_panel=n_missing), f, indent=2)
    print(f"[cutouts] wrote {out_npz} ({os.path.getsize(out_npz)/1e6:.1f} MB, "
          f"{len(zres)} zoom + {len(wres)} wide)", flush=True)
    return out_npz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--dets", required=True, help="masked dets CSV (must carry fits_path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp-px", type=int, default=96, help="zoom cutout size (px); 96 = 19.2 arcsec")
    ap.add_argument("--wide-px", type=int, default=220, help="downsampled wide-view size (px)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N alerts (rank order)")
    a = ap.parse_args(argv)
    build(a.alerts, a.dets, a.out, stamp_px=a.stamp_px, wide_px=a.wide_px, workers=a.workers,
          limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
