#!/usr/bin/env python3
"""Build the static-source catalogue from the NIGHT'S OWN detections -- no DRP coadds needed.

Why: the shipped static veto matches alerts against DRP coadd `object` tables, which only exist
where DRP has processed coadds. Off that footprint the veto is a SILENT no-op -- on 20260629 only
0.15% of alerts had any catalogue to check against, so 8 were flagged instead of the ~19% seen on
a covered night, and the dominant false class (bright template residuals) went entirely unvetoed
while the class counts looked clean.

The physical argument needs no external catalogue: a real mover is at a given sky position in ONE
visit; a static source is there in EVERY visit that covers it. So detections that recur at the
same position across distinct visits ARE the night's static sources. Coverage is complete by
construction, and the catalogue is matched to what actually leaves residuals in THESE difference
images -- galaxies, detector features and artifacts included, which no star catalogue knows about.

Output schema is identical to build_static_catalog (ra, dec, mag), so it drops straight into
`link_2visit --static-catalog`. `mag` is synthetic: selection here is by RECURRENCE, not
brightness (the GPU detector emits no PhotoCalib magnitude), so every row is written below the
veto's mag cut and the recurrence threshold does the selecting.

Usage:
  python -m ADCNN.linking.build_static_selfcal --dets adcnn_dets_masked.csv \
      --out static_selfcal.parquet [--min-visits 3] [--radius-arcsec 1.5]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

_REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd


def _unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def build(dets_path, out_path, min_visits=3, radius_arcsec=1.5, mag_value=18.0, chunk=400000):
    from scipy.spatial import cKDTree
    cols = ["ra", "dec", "visit"]
    d = pd.read_csv(dets_path, usecols=lambda c: c in cols + ["score"])
    n0 = len(d)
    print(f"[selfcal] {n0:,} detections over {d.visit.nunique()} visits", flush=True)

    xyz = _unit(d.ra.to_numpy(), d.dec.to_numpy())
    tree = cKDTree(xyz)
    rad = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
    vis = d.visit.to_numpy()
    ra = d.ra.to_numpy(); dec = d.dec.to_numpy()

    # For each detection, how many DISTINCT visits have a detection within `radius`? A mover
    # contributes 1 (its own visit); a static source contributes one per covering visit.
    keep_ra, keep_dec, nvis_all = [], [], []
    seen = np.zeros(len(d), bool)
    for lo in range(0, len(d), chunk):
        hi = min(lo + chunk, len(d))
        nbrs = tree.query_ball_point(xyz[lo:hi], r=rad)
        for k, nb in enumerate(nbrs):
            i = lo + k
            if seen[i] or not nb:
                continue
            nb = np.asarray(nb)
            nv = len(set(vis[nb].tolist()))
            if nv >= min_visits:
                seen[nb] = True                    # collapse the cluster to one catalogue entry
                keep_ra.append(float(np.median(ra[nb])))
                keep_dec.append(float(np.median(dec[nb])))
                nvis_all.append(nv)
        print(f"[selfcal]   {hi:,}/{len(d):,} scanned, {len(keep_ra):,} static sources", flush=True)

    cat = pd.DataFrame({"ra": keep_ra, "dec": keep_dec, "mag": mag_value,
                        "n_visits": nvis_all})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if str(out_path).endswith(".parquet"):
        cat.to_parquet(out_path, index=False)
    else:
        cat.to_csv(out_path, index=False)
    print(f"[selfcal] {len(cat):,} static sources (>= {min_visits} visits within "
          f"{radius_arcsec}\") -> {out_path}", flush=True)
    if len(cat):
        print(f"[selfcal] recurrence: median {np.median(nvis_all):.0f} visits, "
              f"max {max(nvis_all)}", flush=True)
    return cat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="the night's masked detections CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-visits", type=int, default=2,
                    help="distinct visits a position must recur in AT THE SAME COORDINATES to "
                         "count as static. 2 is correct and safe: a genuine mover's two detections "
                         "are 50-400 arcsec apart (1-8 deg/day over the visit gap), hundreds of "
                         "times the match radius, so it can never self-match. Requiring 3 breaks "
                         "on the common cadence where each pointing is visited exactly twice -- "
                         "it found only 272 statics on such a night vs thousands at 2")
    ap.add_argument("--radius-arcsec", type=float, default=1.5,
                    help="match radius; must exceed astrometric scatter but stay under the "
                         "smallest inter-epoch motion (1 deg/day over 20 min = 50 arcsec)")
    ap.add_argument("--mag", type=float, default=18.0,
                    help="synthetic magnitude written for every row (selection is by recurrence, "
                         "not brightness); keep below the veto's --static-mag-max")
    a = ap.parse_args(argv)
    build(a.dets, a.out, min_visits=a.min_visits, radius_arcsec=a.radius_arcsec, mag_value=a.mag)


if __name__ == "__main__":
    sys.exit(main())
