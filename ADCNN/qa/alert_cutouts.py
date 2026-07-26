#!/usr/bin/env python3
"""Extract per-alert diffim cutouts ONCE, panel-ordered, into a compact cache (npz).

Why this exists: `alert_report.py` renders rank-ordered, reading panels through a 4-panel LRU.
That is fine for a dozen alerts and catastrophic for a 10k-alert nightly stream -- consecutive
ranks land on unrelated panels, so the LRU thrashes and every alert re-reads a ~200 MB FITS
(over S3 for embargo nights). Here the loop is inverted: iterate PANELS, and for each panel cut
every alert epoch that lands on it. FITS I/O becomes O(panels) (~hundreds/night) instead of
O(alerts) (~10k), and all downstream rendering (contact sheets, per-alert stamps, re-sorts after
a threshold change) reads the small cache instead of pixels.

Cache layout (npz, float16 stamps):
  stamps  (N_epoch, K, K) float16   diffim cutout / panel MAD sigma, clipped +-20
  alert   (N_epoch,) int32          index into the alert list (rank order as given)
  epoch   (N_epoch,) int8           which epoch of that alert (0/1/...)
  visit, detector (N_epoch,) int64  provenance
  ok      (N_epoch,) bool           False = panel unreadable / off-image (stamp is zeros)
plus meta.json alongside (K, pixscale, n_alerts, night, source alerts file, missing panels).

Usage:
  python -m ADCNN.qa.alert_cutouts --alerts alerts.jsonl --dets adcnn_dets_masked.csv \
      --out report/cutouts.npz [--stamp-px 64] [--workers 8]
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd

CLIP_SIGMA = 20.0


def _mad_sigma(a):
    from ADCNN.data.preprocessing import diffim_mad_sigma
    return float(diffim_mad_sigma(a))


def _cut(img, x, y, k):
    """k x k cutout centred on (x, y), zero-padded at the panel edge."""
    H, W = img.shape
    h = k // 2
    x0, x1 = int(round(x)) - h, int(round(x)) - h + k
    y0, y1 = int(round(y)) - h, int(round(y)) - h + k
    out = np.zeros((k, k), np.float32)
    sx0, sx1 = max(x0, 0), min(x1, W)
    sy0, sy1 = max(y0, 0), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def _panel_job(args):
    """Cut every requested epoch on ONE panel. Alerts carry sky coords, so the panel's own WCS
    does the sky->pixel conversion here (the FITS is open anyway). Returns per-epoch tuples."""
    path, items, k = args           # items: [(alert_idx, epoch_idx, ra, dec), ...]
    try:
        from astropy.wcs import WCS
        from ADCNN.inference.diffim_io import open_diffim
        with open_diffim(path, memmap=False) as h:
            img = np.nan_to_num(h[1].data.astype(np.float32))
            wcs = WCS(h[1].header)
        sig = _mad_sigma(img) or 1.0
        sky = np.array([[it[2], it[3]] for it in items], float)
        xy = wcs.all_world2pix(sky, 0)
        out = []
        for (ai, ei, _ra, _dec), (x, y) in zip(items, xy):
            s = np.clip(_cut(img, x, y, k) / sig, -CLIP_SIGMA, CLIP_SIGMA)
            out.append((ai, ei, s.astype(np.float16), True))
        return out
    except Exception as e:                      # unreadable panel: emit zero stamps, flag not-ok
        print(f"[cutouts] WARN panel unreadable ({e}): {path}", flush=True)
        return [(ai, ei, np.zeros((k, k), np.float16), False) for (ai, ei, _r, _d) in items]


def build(alerts_path, dets_path, out_npz, stamp_px=64, workers=8, limit=None):
    alerts = [json.loads(l) for l in open(alerts_path)] if os.path.getsize(alerts_path) else []
    if limit:
        alerts = alerts[:limit]
    if not alerts:
        print("[cutouts] 0 alerts -- nothing to do", flush=True)
        return None

    # panel path per (visit, detector) straight from the dets catalog -- no manifest/Butler needed.
    pmap = pd.read_csv(dets_path, usecols=["visit", "detector", "fits_path"]).drop_duplicates(
        ["visit", "detector"])
    panel_of = {(int(v), int(d)): p for v, d, p in zip(pmap.visit, pmap.detector, pmap.fits_path)}

    by_panel = defaultdict(list)
    n_missing_panel = 0
    for ai, al in enumerate(alerts):
        for ei, ep in enumerate(al["epochs"]):
            key = (int(ep["visit"]), int(ep["detector"]))
            p = panel_of.get(key)
            if p is None:
                n_missing_panel += 1
                continue
            by_panel[p].append((ai, ei, float(ep["ra"]), float(ep["dec"])))
    print(f"[cutouts] {len(alerts)} alerts, {sum(len(v) for v in by_panel.values())} epochs "
          f"over {len(by_panel)} panels ({n_missing_panel} epochs with no panel path)", flush=True)

    jobs = [(p, items, stamp_px) for p, items in by_panel.items()]
    results = []
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for n, r in enumerate(ex.map(_panel_job, jobs), 1):
                results.extend(r)
                if n % 25 == 0 or n == len(jobs):
                    print(f"[cutouts] {n}/{len(jobs)} panels", flush=True)
    else:
        for n, j in enumerate(jobs, 1):
            results.extend(_panel_job(j))
            if n % 25 == 0 or n == len(jobs):
                print(f"[cutouts] {n}/{len(jobs)} panels", flush=True)

    results.sort(key=lambda r: (r[0], r[1]))
    stamps = np.stack([r[2] for r in results]) if results else np.zeros((0, stamp_px, stamp_px), np.float16)
    alert_ix = np.array([r[0] for r in results], np.int32)
    epoch_ix = np.array([r[1] for r in results], np.int8)
    ok = np.array([r[3] for r in results], bool)
    vis = np.array([int(alerts[r[0]]["epochs"][r[1]]["visit"]) for r in results], np.int64)
    det = np.array([int(alerts[r[0]]["epochs"][r[1]]["detector"]) for r in results], np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
    np.savez_compressed(out_npz, stamps=stamps, alert=alert_ix, epoch=epoch_ix,
                        visit=vis, detector=det, ok=ok)
    meta = dict(n_alerts=len(alerts), n_epochs=int(len(results)), stamp_px=stamp_px,
                clip_sigma=CLIP_SIGMA, alerts=os.path.abspath(alerts_path),
                dets=os.path.abspath(dets_path), n_panels=len(by_panel),
                n_missing_panel=n_missing_panel, n_bad_panel=int((~ok).sum()))
    with open(os.path.splitext(out_npz)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    mb = os.path.getsize(out_npz) / 1e6
    print(f"[cutouts] wrote {out_npz} ({mb:.1f} MB, {len(results)} stamps) ", flush=True)
    return out_npz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--dets", required=True, help="masked dets CSV (must carry fits_path)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--stamp-px", type=int, default=64, help="cutout size (px); 64 = 12.8 arcsec")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N alerts (rank order)")
    a = ap.parse_args(argv)
    build(a.alerts, a.dets, a.out, stamp_px=a.stamp_px, workers=a.workers, limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
