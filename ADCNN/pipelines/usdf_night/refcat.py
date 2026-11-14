"""Gaia G < 21 reference sources over the night's panels -> work/refcat.parquet (ra, dec, gmag).

Footprint = the registry visit_detector_region polygon of every manifest panel grown by 10 arcsec
(each edge pushed out by 10 arcsec). Sources come from the the_monster_20250219 htm7 shards
(repo embargo, collection refcats) that the grown panels touch, cut to the grown polygons.
gmag = 31.4 - 2.5 log10(phot_g_mean_flux / nJy).

  python -m ADCNN.pipelines.usdf_night.refcat --manifest M --out O [--workers 16]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.pipelines.usdf_night.night import (CredentialError, day_where, open_butler, update_info,
                                              write_atomic)

REFCAT = "the_monster_20250219"
REFCAT_REPO = "embargo"
REFCAT_COLLECTION = "refcats"
FLUX = "phot_g_mean_flux"
MAG_MAX = 21.0
MARGIN_ARCSEC = 10.0
ZP = 31.4
HTM_LEVEL = 7


def g_mag(flux_njy):
    """AB magnitude of a flux in nJy; NaN for flux <= 0 or NaN."""
    f = np.asarray(flux_njy, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, ZP - 2.5 * np.log10(f), np.nan)


def unit_vectors(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def cap(vertices, margin_rad):
    """(centre unit vector, angular radius) of a cap holding the polygon grown by margin_rad."""
    c = vertices.sum(axis=0)
    c /= np.linalg.norm(c)
    return c, float(np.arccos(np.clip(vertices @ c, -1.0, 1.0)).max()) + margin_rad


def inside(vertices, points, margin_rad):
    """Points (n, 3) within the convex spherical polygon `vertices` (k, 3) grown by margin_rad."""
    n = np.cross(vertices, np.roll(vertices, -1, axis=0))
    n /= np.linalg.norm(n, axis=1)[:, None]
    n *= np.sign(n @ vertices.sum(axis=0))[:, None]
    return np.all(points @ n.T >= -np.sin(margin_rad), axis=1)


def footprint_mask(points, polygons, margin_rad):
    from scipy.spatial import cKDTree
    keep = np.zeros(len(points), bool)
    if not len(points):
        return keep
    tree = cKDTree(points)
    for verts in polygons:
        c, r = cap(verts, margin_rad)
        idx = np.asarray(tree.query_ball_point(c, 2.0 * np.sin(r / 2.0)), np.int64)
        if idx.size:
            keep[idx[inside(verts, points[idx], margin_rad)]] = True
    return keep


def htm_shards(polygons, margin_rad, level=HTM_LEVEL):
    import lsst.sphgeom as sg
    pix = sg.HtmPixelization(level)
    out = set()
    for verts in polygons:
        c, r = cap(verts, margin_rad)
        for lo, hi in pix.envelope(sg.Circle(sg.UnitVector3d(*c), sg.Angle(r))):
            out.update(range(lo, hi))
    return sorted(out)


def panel_polygons(butler, day, keys):
    """{(visit, detector): vertices (k, 3)} from visit_detector_region for the given panels."""
    keys = set(keys)
    out = {}
    for rec in butler.query_dimension_records("visit_detector_region", where=day_where(day),
                                              limit=None, explain=False):
        key = (int(rec.visit), int(rec.detector))
        if key in keys:
            out[key] = np.array([[u.x(), u.y(), u.z()] for u in rec.region.getVertices()])
    return out


_BUTLER = None


def _init(repo):
    global _BUTLER
    _BUTLER = open_butler(repo)


def _shard(ref):
    cat = _BUTLER.get(ref)
    if not cat.isContiguous():
        cat = cat.copy(deep=True)
    mag = g_mag(cat[FLUX])
    keep = mag < MAG_MAX
    return np.degrees(cat["coord_ra"][keep]), np.degrees(cat["coord_dec"][keep]), mag[keep]


def build(manifest, out, workers=16) -> dict:
    t0 = time.time()
    m = pd.read_csv(manifest)
    margin = np.radians(MARGIN_ARCSEC / 3600.0)
    info = {"refcat": REFCAT, "repo": REFCAT_REPO, "collection": REFCAT_COLLECTION, "flux_column": FLUX,
            "mag_max": MAG_MAX, "margin_arcsec": MARGIN_ARCSEC,
            "footprint": "visit_detector_region polygons grown by margin_arcsec", "n_panels": len(m),
            "panels_without_region": []}
    cat = pd.DataFrame({"ra": [], "dec": [], "gmag": []})
    if len(m):
        repos, days = m.repo.unique(), m.day_obs.unique()
        if len(repos) != 1 or len(days) != 1:
            raise RuntimeError(f"manifest must hold one repo and one day_obs: {list(repos)} {list(days)}")
        keys = list(zip(m.visit.astype(int), m.detector.astype(int)))
        polys = panel_polygons(open_butler(repos[0]), str(days[0]), keys)
        info["panels_without_region"] = [list(k) for k in keys if k not in polys]
        polygons = [polys[k] for k in keys if k in polys]
        shards = htm_shards(polygons, margin)
        rb = open_butler(REFCAT_REPO)
        refs = rb.query_datasets(REFCAT, collections=REFCAT_COLLECTION,
                                 where=f"htm7 IN ({','.join(map(str, shards))})",
                                 find_first=True, limit=None, explain=False) if shards else []
        refs = sorted(refs, key=lambda r: int(r.dataId["htm7"]))
        info.update(n_shards=len(shards), n_shard_refs=len(refs))
        print(f"[refcat] {len(polygons)} panels -> {len(shards)} htm7 shards, {len(refs)} in {REFCAT} "
              f"({time.time() - t0:.1f} s)", flush=True)
        parts = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init, initargs=(REFCAT_REPO,)) as ex:
            for i, part in enumerate(ex.map(_shard, refs, chunksize=4), 1):
                parts.append(part)
                if i % 500 == 0 or i == len(refs):
                    print(f"[refcat]   {i}/{len(refs)} shards, {time.time() - t0:.0f} s", flush=True)
        ra, dec, mag = (np.concatenate([p[j] for p in parts]) if parts else np.zeros(0) for j in range(3))
        keep = footprint_mask(unit_vectors(ra, dec), polygons, margin)
        info["n_sources_shards"] = int(len(ra))
        cat = pd.DataFrame({"ra": ra[keep], "dec": dec[keep], "gmag": mag[keep]})
    write_atomic(cat, out)
    info.update(n_sources=len(cat), seconds=round(time.time() - t0, 1))
    update_info(Path(out).parent / "prep_info.json", "refcat", info)
    print(f"[refcat] {len(cat):,} sources with G < {MAG_MAX} -> {out} ({info['seconds']} s)", flush=True)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="refcat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    try:
        build(a.manifest, a.out, a.workers)
    except CredentialError as e:
        print(f"refcat: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
