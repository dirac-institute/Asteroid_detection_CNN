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
import argparse, hashlib, json, os, re, sys
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


_DET_TOK_RE = re.compile(r"_(R\d\d_S\d\d)_LSSTCam_runs")


def _det_token(path):
    m = _DET_TOK_RE.search(path or "")
    return m.group(1) if m else None


def _grid_of(token):
    """R<c><r>_S<sc><sr> -> a 2-D coordinate that is LINEAR in focal-plane position (raft*3+sensor
    along each axis). Used only to fit an affine grid->sky map, so its absolute convention is
    irrelevant as long as it is linear in the real layout."""
    return (int(token[1]) * 3 + int(token[5]), int(token[2]) * 3 + int(token[6]))


def _tangent_grid(cra, cdec, span_arcsec, out_px):
    """out_px x out_px sky grid on a local gnomonic tangent plane centred at (cra,cdec). Returns
    (radec (N,2), apx_arcsec). The grid is defined in SKY, not in any one detector's pixels, so no
    detector WCS is ever extrapolated beyond its own footprint (that produced a garbage checkerboard)."""
    apx = span_arcsec / out_px
    cd = np.cos(np.radians(cdec))
    off = (np.arange(out_px) - (out_px - 1) / 2.0) * apx
    dx, dy = np.meshgrid(off, off)
    ra = cra + dx / (3600.0 * cd)
    dec = cdec + dy / 3600.0
    return np.column_stack([ra.ravel(), dec.ravel()]), apx


def _radec_to_grid(ra, dec, cra, cdec, apx, out_px):
    cd = np.cos(np.radians(cdec))
    ix = (ra - cra) * cd * 3600.0 / apx + (out_px - 1) / 2.0
    iy = (dec - cdec) * 3600.0 / apx + (out_px - 1) / 2.0
    return ix, iy


def _panel_corrupt(vals, sig):
    """A neighbour is refused if the values it would contribute are mostly saturated -- the mfsnr-
    blowup diffim class paints a black/white checkerboard that would wreck the mosaic."""
    if len(vals) == 0:
        return False
    return float((np.abs(vals) > 8.0 * sig).mean()) > 0.5


def _wide_mosaic_one(members, endpoints, out_px, cand_paths, load_fn, affine, cra_cdec,
                     margin_as=12.0):
    """Build ONE wide stamp by mosaicking every detector that overlaps the box.

    members       list of (ra,dec) per epoch
    endpoints     list per epoch of (ra0,dec0,ra1,dec1) measured trail ends (or None)
    cand_paths    ordered list of (det_id, path) candidate detectors of this visit (nearest first)
    load_fn       det path -> (img, wcs, sig), memoised by the caller (per-visit panel cache)
    Fills residual chip-gap pixels with sky-matched noise (unit sigma) so the frame is continuous.
    Returns (stamp float16, pos (MAXEP,2), apx_arcsec, wends (MAXEP,2,2), filled_frac)."""
    ras = np.array([m[0] for m in members]); decs = np.array([m[1] for m in members])
    cra, cdec = float(ras.mean()), float(decs.mean()); cd = np.cos(np.radians(cdec))
    sep = 0.0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            sep = max(sep, np.hypot((ras[i] - ras[j]) * cd, decs[i] - decs[j]) * 3600)
    span = float(max(sep + 2 * margin_as, 4 * margin_as))
    radec, apx = _tangent_grid(cra, cdec, span, out_px)
    n = out_px * out_px
    stamp = np.zeros(n, np.float32); filled = np.zeros(n, bool)
    for _det, path in cand_paths:
        if filled.all():
            break
        try:
            img, wcs, sig = load_fn(path)
        except Exception:
            continue
        idx = np.nonzero(~filled)[0]
        pix = wcs.all_world2pix(radec[idx], 0)
        rx = np.round(pix[:, 0]).astype(int); ry = np.round(pix[:, 1]).astype(int)
        H, W = img.shape
        good = (rx >= 0) & (rx < W) & (ry >= 0) & (ry < H)
        if not good.any():
            continue
        vals = img[ry[good], rx[good]]
        ok = vals != 0.0                                   # skip this detector's own masked pixels
        vals, sel = vals[ok], idx[good][ok]
        if len(vals) and not _panel_corrupt(vals, sig):
            stamp[sel] = vals / sig                        # normalise by each detector's own sigma
            filled[sel] = True
    filled_frac = float(filled.mean())
    # Gaps are filled with SYNTHETIC noise so chip boundaries do not read as hard edges -- but the RNG
    # was unseeded, so two identical calls produced different pixels and a re-render never reproduced
    # the delivered image. Seed it on the mosaic centre: deterministic, and still decorrelated between
    # alerts. Measured fabrication: 0.3-1.2% of a median-rate alert's wide view, 8-13% for the fastest
    # (whose mosaic reaches past the focal plane), and 100% when every candidate panel is unreadable.
    # hash(float('nan')) is id-derived in Python 3.10+, so a NaN centre would silently un-seed this
    # again -- the exact non-reproducibility the seed exists to remove. Fall back to a fixed seed.
    _key = (round(float(cra), 6), round(float(cdec), 6), int(out_px))
    _rng = np.random.default_rng(abs(hash(_key)) % (2**32)
                                 if np.isfinite(cra) and np.isfinite(cdec) else 0)
    stamp[~filled] = _rng.normal(0.0, 1.0, int((~filled).sum())).astype(np.float32)  # chip gaps
    stamp = np.clip(stamp, -CLIP_SIGMA, CLIP_SIGMA).reshape(out_px, out_px).astype(np.float16)
    pos = np.full((MAXEP, 2), np.nan, np.float32)
    for i, (mra, mdec) in enumerate(members[:MAXEP]):
        pos[i] = _radec_to_grid(mra, mdec, cra, cdec, apx, out_px)
    wends = np.full((MAXEP, 2, 2), np.nan, np.float32)
    for i, en in enumerate(endpoints[:MAXEP]):
        if en is None:
            continue
        for j, (ra, dec) in enumerate([(en[0], en[1]), (en[2], en[3])]):
            wends[i, j] = _radec_to_grid(ra, dec, cra, cdec, apx, out_px)
    return stamp, pos, np.float32(apx), wends, filled_frac


WIDE_MIN_FILLED = 0.5   # below this the mosaic is mostly fabricated noise, not pixels

_PANEL_CACHE = {}                # per-worker LRU of loaded panels (mosaic neighbours are re-read
_PANEL_CACHE_MAX = 12            # across anchors of the same visit; a small cache kills that cost)


def _load_panel(path):
    """(img, wcs, sig) for a diffim panel, memoised per worker (bounded)."""
    hit = _PANEL_CACHE.pop(path, None)
    if hit is not None:
        _PANEL_CACHE[path] = hit                       # move to MRU
        return hit
    from astropy.wcs import WCS
    from ADCNN.inference.diffim_io import open_diffim
    with open_diffim(path, memmap=False) as h:
        img = np.nan_to_num(h[1].data.astype(np.float32))
        wcs = WCS(h[1].header)
    val = (img, wcs, _mad_sigma(img) or 1.0)
    _PANEL_CACHE[path] = val
    if len(_PANEL_CACHE) > _PANEL_CACHE_MAX:
        _PANEL_CACHE.pop(next(iter(_PANEL_CACHE)))     # evict LRU
    return val


def _panel_job(args):
    """One anchor panel: all ZOOM cuts on it. Wide cuts are handled separately, grouped by visit,
    because they mosaic across many detectors and must not re-read a panel per alert."""
    path, zooms, k = args
    try:
        img, wcs, sig = _load_panel(path)
        zout = []
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
        return zout
    except Exception as e:
        print(f"[cutouts] WARN panel unreadable ({e}): {path}", flush=True)
        return [(ai, ei, np.zeros((k, k), np.float16), False,
                 np.full((2, 2), np.nan, np.float32)) for (ai, ei, _r, _d, _e) in zooms]


def _anchor_det(entry):
    """Anchor detector of a wide entry, or None for older 3-tuples (kept backward-compatible so a
    half-written cache from a previous version still loads)."""
    return entry[3] if len(entry) > 3 else None


def _wide_visit_job(args):
    """All wide cuts of ONE visit, mosaicked across its detectors. Each visit's panels are loaded
    at most once (local cache), so cost is O(panels of the visit) not O(alerts). Undetected
    detectors that overlap a box are pulled too: their S3 path is built from a detected sibling's
    path by swapping the R##_S## token, and which detectors overlap is decided by an affine fit of
    (raft/sensor grid coord -> sky centre) over the detected detectors -- no FITS opened for
    geometry, no camera-geometry / Butler dependency."""
    visit, wides, det_paths, det_centers, det_tokens, kw, radius_deg = args
    out = []
    try:
        cache = {}

        def load_fn(p):
            v = cache.get(p)
            if v is None:
                from astropy.wcs import WCS
                from ADCNN.inference.diffim_io import open_diffim
                with open_diffim(p, memmap=False) as h:
                    img = np.nan_to_num(h[1].data.astype(np.float32))
                    wcs = WCS(h[1].header)
                v = (img, wcs, _mad_sigma(img) or 1.0)
                if len(cache) >= 60:                       # cap RAM; visits rarely need more
                    cache.pop(next(iter(cache)))
                cache[p] = v
            return v

        # affine: focal-plane grid coord -> (ra*cos dec, dec), fit on this visit's detected panels
        kk = [d for d in det_paths if d in det_centers and d in det_tokens]
        pred_center = {}
        if len(kk) >= 3:
            mdec = np.mean([det_centers[d][1] for d in kk]); cdk = np.cos(np.radians(mdec))
            G = np.array([_grid_of(det_tokens[d]) for d in kk], float)
            S = np.array([[det_centers[d][0] * cdk, det_centers[d][1]] for d in kk])
            A, *_ = np.linalg.lstsq(np.column_stack([G, np.ones(len(G))]), S, rcond=None)
            for d, tok in det_tokens.items():
                p = np.array([*_grid_of(tok), 1.0]) @ A
                pred_center[d] = (p[0] / cdk, p[1])
        else:
            pred_center = dict(det_centers)                # fall back to detected only
        tmpl = next(iter(det_paths.values())); tmpl_tok = _det_token(tmpl)

        for (ai, members, endpoints, _det) in wides:
            bcra = float(np.mean([m[0] for m in members])); bcdec = float(np.mean([m[1] for m in members]))
            cd = np.cos(np.radians(bcdec))
            cands = []
            for d, (pra, pdec) in pred_center.items():
                dd = np.hypot((pra - bcra) * cd, pdec - bcdec)
                if dd <= radius_deg:
                    path = det_paths.get(d) or (tmpl.replace(tmpl_tok, det_tokens[d]) if d in det_tokens else None)
                    if path:
                        cands.append((dd, d, path))
            cands.sort()
            cand_paths = [(d, p) for _dd, d, p in cands]
            stamp, pos, apx, wends, ff = _wide_mosaic_one(members, endpoints, kw, cand_paths, load_fn,
                                                          None, (bcra, bcdec))
            # `ff` was computed, returned, and then DISCARDED for a hardcoded True. The total-failure
            # path (no candidate panel readable) yields filled_frac 0.0 -- a stamp of pure fabricated
            # noise, std 0.997 with >4-sigma pixels against a VMAX of 6, so it renders as a source in
            # an empty field. wide_ok now carries what was actually measured.
            out.append((ai, stamp, pos, apx, bool(ff >= WIDE_MIN_FILLED), wends))
    except Exception as e:
        print(f"[cutouts] WARN wide visit {visit} failed ({e})", flush=True)
        for (ai, _m, _e, _d) in wides:
            out.append((ai, np.zeros((kw, kw), np.float16), np.full((MAXEP, 2), np.nan, np.float32),
                        np.float32(np.nan), False, np.full((MAXEP, 2, 2), np.nan, np.float32)))
    return out


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

    # Global det_id -> R##_S## token (to construct paths of UNDETECTED overlapping detectors) and
    # per-visit detected panel paths + sky centres (to fit the mosaic's grid->sky geometry).
    det_tokens, det_paths_by_visit = {}, defaultdict(dict)
    for (v, d), p in panel_of.items():
        det_paths_by_visit[v][d] = p
        t = _det_token(p)
        if t:
            det_tokens.setdefault(d, t)
    cen = pd.read_csv(dets_path, usecols=["visit", "detector", "ra", "dec"]).groupby(
        ["visit", "detector"])[["ra", "dec"]].mean()
    det_centers_by_visit = defaultdict(dict)
    for (v, d), (r, dc) in cen.iterrows():
        det_centers_by_visit[int(v)][int(d)] = (float(r), float(dc))

    ends = _match_endpoints(alerts, dets_path)
    zoom_by_panel, wide_by_visit = defaultdict(list), defaultdict(list)
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
        v0 = int(eps[0]["visit"])
        if v0 in det_paths_by_visit:                       # anchor visit has readable panels
            # carry the anchor detector so the jobs below can be chunked on it
            _d0 = eps[0].get("detector")
            wide_by_visit[v0].append((ai, [(float(e["ra"]), float(e["dec"])) for e in eps],
                                      [ends.get((ai, j)) for j in range(len(eps))],
                                      int(_d0) if _d0 is not None else None))

    zoom_jobs = [(p, zoom_by_panel[p], stamp_px) for p in sorted(zoom_by_panel)]
    # WIDE JOBS ARE CHUNKED BY (visit, anchor detector), NOT BY VISIT.
    #
    # One job per visit made the stage serial on its biggest visit: MEASURED on 20260630, one visit
    # held 8,998 of 13,406 wide mosaics (67%) while the other ten held 1,450 down to 1 -- and with
    # the pool capped at 4 workers, three sat idle after ~2 minutes of CPU while one accumulated 66
    # minutes and had still not emitted a single progress line after 70 wall-clock minutes. That one
    # visit is the floor: no amount of extra workers could touch it.
    #
    # Chunking on the anchor DETECTOR is what makes this safe to split. The per-worker panel cache
    # exists because a mosaic pulls its neighbouring detectors, so splitting a visit arbitrarily
    # would re-read those panels in every chunk. Alerts sharing a detector share almost exactly the
    # same neighbour set, so a detector-chunk touches the same panels a visit-chunk would have, just
    # fewer times -- the cache hit rate is preserved and peak RAM per worker DROPS (a job now holds
    # one detector's neighbourhood, not a whole focal plane).
    #
    # 20260630: 11 jobs (largest 8,998) -> 302 jobs (largest 126), a 71x lower serial floor.
    wide_jobs = []
    for v in sorted(wide_by_visit):
        by_det = defaultdict(list)
        for entry in wide_by_visit[v]:
            by_det[_anchor_det(entry)].append(entry)
        for det in sorted(by_det, key=lambda d: (d is None, d)):
            wide_jobs.append((v, by_det[det], det_paths_by_visit[v],
                              det_centers_by_visit.get(v, {}), det_tokens, wide_px, 0.5))
    print(f"[cutouts] {len(alerts)} alerts -> {sum(len(v) for v in zoom_by_panel.values())} zooms "
          f"over {len(zoom_jobs)} panels + {sum(len(v) for v in wide_by_visit.values())} wide "
          f"over {len(wide_jobs)} detector-chunks in {len(wide_by_visit)} visits "
          f"({n_missing} epochs with no panel path)", flush=True)

    zres, wres = [], []
    if workers > 1 and (len(zoom_jobs) > 1 or len(wide_jobs) > 1):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for n, zo in enumerate(ex.map(_panel_job, zoom_jobs), 1):
                zres.extend(zo)
                if n % 25 == 0 or n == len(zoom_jobs):
                    print(f"[cutouts] zoom {n}/{len(zoom_jobs)} panels", flush=True)
        # The cap of 4 was there because a per-VISIT job held many ~200MB panels at once. A
        # per-DETECTOR job holds only that detector's neighbourhood, so the cap can go: the binding
        # constraint is now the same worker count the zoom stage already runs at.
        ww = max(1, workers)
        with ProcessPoolExecutor(max_workers=ww, mp_context=ctx) as ex:
            for n, wo in enumerate(ex.map(_wide_visit_job, wide_jobs), 1):
                wres.extend(wo)
                if n % 25 == 0 or n == len(wide_jobs):
                    print(f"[cutouts] wide {n}/{len(wide_jobs)} detector-chunks", flush=True)
    else:
        for j in zoom_jobs:
            zres.extend(_panel_job(j))
        for j in wide_jobs:
            wres.extend(_wide_visit_job(j))

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
    # IDENTITY FINGERPRINT of the alert sequence this cache was built against. A count cannot see a
    # PERMUTATION, and that is exactly how six delivered nights shipped sheets captioned with the
    # wrong alert: the cache is keyed by alert position and rerank_alerts rewrote alerts.jsonl in
    # place after the cut. Any consumer -- renderers, night_status -- can now decide in O(1) whether
    # the file in front of it is the one the pixels were cut from, without loading the npz.
    _fp = hashlib.sha256()
    for _a in alerts:
        for _e in (_a.get("epochs") or []):
            _fp.update(f"{_e.get('visit',-1)}:{_e.get('detector',-1)};".encode())
        _fp.update(b"|")
    with open(os.path.splitext(out_npz)[0] + "_meta.json", "w") as f:
        json.dump(dict(alerts_fingerprint=_fp.hexdigest(),
                       n_alerts=len(alerts), n_zoom=len(zres), n_wide=len(wres), stamp_px=K,
                       wide_px=KW, clip_sigma=CLIP_SIGMA, alerts=os.path.abspath(alerts_path),
                       dets=os.path.abspath(dets_path), n_panels=len(zoom_jobs),
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
