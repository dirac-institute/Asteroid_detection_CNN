"""Build a Sorcha pointing database (rubin_sim 'observations' table, sqlite) from the REAL DP2 visit
records over an ecliptic region, so Sorcha propagates the Granvik NEOs through the genuine LSST cadence
(-> realistic per-object observable-apparition counts, incl. objects in-FOV only twice).

Boresight (fieldRA/Dec) + MJD + filter + exposure are REAL (from the visit dimension records). Columns
Sorcha needs but that don't affect the in-FOV apparition set are filled permissively: fiveSigmaDepth deep
(=30, so the fading function keeps even faint apparitions -- ADCNN is the real detector), seeing ~0.8",
rotSkyPos 0 (we keep the footprint orientation neutral). The ephemeris/FOV stage uses ar_ang_fov.
"""
from __future__ import annotations
import argparse, sqlite3
import numpy as np, pandas as pd
from lsst.daf.butler import Butler
from lsst.sphgeom import LonLat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra-min", type=float, default=340.0); ap.add_argument("--ra-max", type=float, default=346.0)
    ap.add_argument("--dec-min", type=float, default=-8.0); ap.add_argument("--dec-max", type=float, default=-2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = Butler("dp2_prep")
    rows = []
    for r in b.registry.queryDimensionRecords("visit", where="instrument='LSSTCam'"):
        try:
            c = r.region.getBoundingCircle().getCenter(); ll = LonLat(c)
            ra = ll.getLon().asDegrees(); dec = ll.getLat().asDegrees()
            if not (a.ra_min <= ra <= a.ra_max and a.dec_min <= dec <= a.dec_max):
                continue
            band = str(r.physical_filter).split("_")[0][:1]
            mjd = r.timespan.begin.mjd
            rows.append((int(r.id), float(mjd), float(r.exposure_time), float(r.exposure_time), band,
                         0.8, 0.8, 30.0, float(ra), float(dec), 0.0))
        except Exception:
            continue
    cols = ["observationId", "observationStartMJD", "visitTime", "visitExposureTime", "filter",
            "seeingFwhmGeom", "seeingFwhmEff", "fiveSigmaDepth", "fieldRA", "fieldDec", "rotSkyPos"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("observationId").sort_values("observationStartMJD")
    con = sqlite3.connect(a.out); df.to_sql("observations", con, if_exists="replace", index=False); con.close()
    nights = df.observationStartMJD.apply(lambda m: int(np.floor(m - 0.5))).nunique()
    print(f"[pointing] {len(df)} visits in ra[{a.ra_min},{a.ra_max}] dec[{a.dec_min},{a.dec_max}] over {nights} nights "
          f"| filters {sorted(df.filter.unique())} -> {a.out}")


if __name__ == "__main__":
    main()
