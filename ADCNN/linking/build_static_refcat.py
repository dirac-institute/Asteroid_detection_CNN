#!/usr/bin/env python3
"""Build the static-source catalogue from an ALL-SKY reference catalogue -- coverage everywhere.

The shipped build_static_catalog reads DRP coadd `object` tables, which exist only where DRP has
processed coadds. Off that footprint the static veto is a SILENT no-op: on 20260629 just 0.15% of
alerts had any catalogue to check against, 8 were flagged instead of the ~19% a covered night
sees, and the dominant false class (bright template residuals) went entirely unvetoed while the
class counts looked clean.

Reference catalogues do not have that problem. The embargo repo carries gaia_dr2_20200414 (all
sky, complete to G~21), ps1_pv3_3pi_20170110 (Dec > -30) and the_monster_20250219. The veto only
needs bright stationary sources -- its cut is mag < 20 -- which is exactly what these provide,
everywhere the telescope points.

Trade-off worth stating: a refcat lists STARS, while DRP coadd objects also include galaxies. A
bright galaxy that leaves a subtraction residual is caught by the coadd catalogue and missed here.
So prefer the DRP catalogue where coverage exists and use this to fill the gaps -- or pass both
(the veto takes a concatenation). This tool reports the overlap so that choice is informed.

Output schema matches build_static_catalog (ra, dec, mag) and drops straight into
`link_2visit --static-catalog`.

Usage (stack env):
  python -m ADCNN.linking.build_static_refcat --dets adcnn_dets_masked.csv \
      --out static_refcat.parquet [--refcat gaia_dr2_20200414] [--mag-max 21]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

_REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

# refcat flux columns are nJy; mag = 31.4 - 2.5log10(flux_nJy) is the AB convention used elsewhere
# in this package (inject_trails.ZP).
ZP = 31.4


def build(dets_path, out_path, refcat="the_monster_20250219", repo="embargo", mag_max=21.0,
          collection="refcats"):
    from lsst.daf.butler import Butler
    import lsst.sphgeom as sphgeom

    d = pd.read_csv(dets_path, usecols=["ra", "dec"])
    ra, dec = d.ra.to_numpy(), d.dec.to_numpy()
    print(f"[refcat] night footprint from {len(d):,} detections: "
          f"ra {ra.min():.2f}-{ra.max():.2f} dec {dec.min():.2f}-{dec.max():.2f}", flush=True)

    b = Butler(repo)
    # htm7 shards covering the detections (the refcat is sharded on htm7)
    pix = sphgeom.HtmPixelization(7)
    shards = set()
    for r, dd in zip(ra, dec):
        shards.add(pix.index(sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(float(r), float(dd)))))
    print(f"[refcat] {len(shards)} htm7 shards to fetch", flush=True)

    rows = []
    got = miss = 0
    for i, sh in enumerate(sorted(shards), 1):
        try:
            cat = b.get(refcat, htm7=int(sh), collections=[collection])
        except Exception:
            miss += 1
            continue
        got += 1
        t = cat.asAstropy()
        fcol = next((c for c in t.colnames if c.endswith("_flux") and "Err" not in c), None)
        if fcol is None:
            continue
        f = np.asarray(t[fcol], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = ZP - 2.5 * np.log10(np.where(f > 0, f, np.nan))
        keep = np.isfinite(mag) & (mag < mag_max)
        if keep.any():
            rows.append(pd.DataFrame({"ra": np.degrees(np.asarray(t["coord_ra"], float))[keep],
                                      "dec": np.degrees(np.asarray(t["coord_dec"], float))[keep],
                                      "mag": mag[keep]}))
        if i % 50 == 0:
            print(f"[refcat]   {i}/{len(shards)} shards, {sum(len(x) for x in rows):,} sources",
                  flush=True)

    cat = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ra", "dec", "mag"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if str(out_path).endswith(".parquet"):
        cat.to_parquet(out_path, index=False)
    else:
        cat.to_csv(out_path, index=False)
    print(f"[refcat] {len(cat):,} sources brighter than {mag_max} from {refcat} "
          f"({got} shards read, {miss} absent) -> {out_path}", flush=True)
    return cat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--collection", default="refcats",
                    help="Butler collection holding the refcat (NOT the dataset type: passing the\n                         type name silently returns zero shards)")
    ap.add_argument("--refcat", default="the_monster_20250219",
                    help="all-sky reference catalogue dataset type (gaia_dr2_20200414, "
                         "the_monster_20250219, ps1_pv3_3pi_20170110 [Dec > -30 only])")
    ap.add_argument("--repo", default="embargo")
    ap.add_argument("--mag-max", type=float, default=21.0,
                    help="keep sources brighter than this; the veto cuts at 20, so 21 leaves a "
                         "magnitude of headroom")
    a = ap.parse_args(argv)
    build(a.dets, a.out, refcat=a.refcat, repo=a.repo, mag_max=a.mag_max,
          collection=a.collection)


if __name__ == "__main__":
    sys.exit(main())
