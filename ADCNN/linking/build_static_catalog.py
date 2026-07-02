"""Build the bright-static template-footprint catalog for the link_2visit --static-catalog veto.

WHY (measured 2026-07-02 on embargo night 20260630, run_embargo_0630/expt_staticveto/RESULTS.md):
~90% of the dense-field 2-visit FP background is subtraction residuals living in the 2-3" WINGS of
bright (mag<20) deep-coadd statics -- structured, NOT chance (95/107 floor-0.5 alerts were
static-static pairs; exactly the clean ones survived their removal). The linker excludes
static-static seed pairs against this catalog and FLAG-demotes single-static alerts.

WHAT: for every skymap tract touched by the link-input detections, load the DRP `object` table
(new naming; NOT the purged dp2_prep `objectTable`), keep detect_isPrimary rows, and save
(ra, dec, mag) where mag = the brightest griz cModel magnitude (nJy, ZP 31.4). The catalog is
saved FULL DEPTH -- the linker applies its own --static-mag-max at read, so one catalog serves
any cut. Tracts with no `object` coverage are reported LOUDLY: dets there can never be flagged.

ENVIRONMENT: needs the LSST stack (lsst.daf.butler / lsst.geom), NOT the asteroid_cnn conda env:
    source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/<weekly>/loadLSST.sh && setup lsst_distrib
Registry auth gotcha: the shared repos point at `...-db-tx-ro` read-only replicas that may be
missing from ~/.lsst/postgres-credentials.txt -- append the -ro alias lines (same creds) to a
chmod-600 COPY and export PGPASSFILE=<copy>; never edit the original.

Usage:
    python -m ADCNN.linking.build_static_catalog \
        --dets run_embargo_0630/adcnn_dets_masked.csv \
        --out  run_embargo_0630/expt_staticveto/static_catalog.parquet
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import pandas as pd

BANDS = "griz"
AB_ZP_NJY = 31.4          # DRP fluxes are nJy; m = -2.5 log10(flux) + 31.4
OBJECT_COLUMNS = ["coord_ra", "coord_dec", "detect_isPrimary", "refExtendedness"] + \
                 [f"{b}_cModelFlux" for b in BANDS]


def dets_tracts(dets, skymap):
    """Skymap tracts touched by the detection positions. Positions are decimated to a 0.1 deg grid
    first (a tract is ~1.7 deg -- the grid cannot skip one), so findTract runs on ~10^2 unique
    points instead of ~10^5 dets. Returns (sorted tract ids, per-det tract Series)."""
    import lsst.geom as geom
    key = dets.ra.round(1).astype(str) + "_" + dets.dec.round(1).astype(str)
    tr_of = {}
    for k in sorted(set(key)):
        ra_s, dec_s = k.split("_")
        tr_of[k] = skymap.findTract(geom.SpherePoint(float(ra_s), float(dec_s),
                                                     geom.degrees)).getId()
    per_det = key.map(tr_of)
    return sorted(set(tr_of.values())), per_det


def load_tract_objects(butler, ref):
    """One tract's (ra, dec, mag) bright-static rows from a DRP `object` dataset ref: column-
    restricted read (the full table is GBs), detect_isPrimary only, mag = brightest griz cModel."""
    try:
        tab = butler.get(ref, parameters={"columns": OBJECT_COLUMNS})
    except Exception:
        tab = butler.get(ref)                      # schema variant: fall back to the full read
    tab = tab.to_pandas() if hasattr(tab, "to_pandas") else tab
    if "detect_isPrimary" in tab.columns:
        tab = tab[tab["detect_isPrimary"].fillna(False).astype(bool)]
    mags = []
    for band in BANDS:
        c = f"{band}_cModelFlux"
        if c in tab.columns:
            f = np.asarray(tab[c], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mags.append(np.where(f > 0, -2.5 * np.log10(f) + AB_ZP_NJY, np.inf))
    if not mags:
        raise SystemExit(f"tract {ref.dataId['tract']}: no {'/'.join(BANDS)}_cModelFlux columns -- "
                         f"unexpected `object` schema in run {ref.run}")
    out = pd.DataFrame({"ra": np.asarray(tab["coord_ra"], float),
                        "dec": np.asarray(tab["coord_dec"], float),
                        "mag": np.min(np.vstack(mags), axis=0)})
    return out[np.isfinite(out.mag)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True,
                    help="link-input dets CSV (needs ra,dec) -- defines the tracts the catalog must cover")
    ap.add_argument("--out", required=True, help="output path (.parquet, or .csv/.csv.gz)")
    ap.add_argument("--repo", default="main",
                    help="butler repo holding the coadd `object` tables (default: main -- the embargo "
                    "repo has only prompt/DIA types, dp2_prep is purged)")
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--collections", default="LSSTCam/runs/DRP/*",
                    help="collection glob searched for `object` datasets; the NEWEST run wins per tract")
    ap.add_argument("--dataset-type", default="object")
    ap.add_argument("--mag-max", type=float, default=None,
                    help="optional pre-cut on the SAVED catalog. Default: save full depth (the linker "
                    "applies its own --static-mag-max at read)")
    a = ap.parse_args()

    from lsst.daf.butler import Butler
    b = Butler(a.repo)
    sm = b.get("skyMap", skymap=a.skymap, collections="skymaps")
    d = pd.read_csv(a.dets, usecols=["ra", "dec"])
    need, per_det = dets_tracts(d, sm)
    print(f"[static-catalog] {len(d)} dets -> {len(need)} tracts (skymap {a.skymap})", flush=True)

    refs = list(b.registry.queryDatasets(a.dataset_type, collections=a.collections,
                                         findFirst=False))
    by_tract = {}
    for r in refs:
        if r.dataId["skymap"] == a.skymap and r.dataId["tract"] in set(need):
            by_tract.setdefault(r.dataId["tract"], []).append(r)
    missing = sorted(set(need) - set(by_tract))
    if missing:
        n_uncov = int(per_det.isin(set(missing)).sum())
        print(f"[static-catalog] WARNING: {len(missing)} tract(s) have NO `{a.dataset_type}` "
              f"coverage in {a.repo}:{a.collections} -- {n_uncov}/{len(d)} dets "
              f"({n_uncov / max(len(d), 1):.1%}) can never be static-flagged: {missing}", flush=True)
    if not by_tract:
        raise SystemExit(f"[static-catalog] no `{a.dataset_type}` datasets cover ANY needed tract -- "
                         f"wrong --repo/--collections/--skymap?")

    parts = []
    for t in sorted(by_tract):
        ref = sorted(by_tract[t], key=lambda r: r.run, reverse=True)[0]   # newest run wins
        part = load_tract_objects(b, ref)
        print(f"  tract {t} [{ref.run}]: {len(part)} primary objs", flush=True)
        parts.append(part)
    cat = pd.concat(parts, ignore_index=True)
    if a.mag_max is not None:
        cat = cat[cat.mag < a.mag_max].reset_index(drop=True)
    if str(a.out).endswith((".parquet", ".pq")):
        cat.to_parquet(a.out)
    else:
        cat.to_csv(a.out, index=False)
    print(f"[static-catalog] {len(cat)} objects ({int((cat.mag < 20).sum())} mag<20) -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
