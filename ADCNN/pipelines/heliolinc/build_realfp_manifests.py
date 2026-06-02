"""Build per-(tract,night) diffim manifests for OFF-ECLIPTIC dense field-nights, to measure the REAL
2-visit false-link rate directly (no injection, no Monte-Carlo permutation). Off-ecliptic => ~zero real
asteroids => every surviving 2-track is a genuine false link. One dense field-night gives ~(n_visits-1)
same-night pairs from one tract footprint (the run_test2 unit), so a few fields accumulate thousands of
pairs. Picks the densest |ecl_lat|>20 deg fields from cadence.csv, resolves each to its skymap tract,
and writes run_realfp/manifest_<k>.csv + a pairs summary."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from lsst.daf.butler import Butler
from lsst.geom import SpherePoint, degrees

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", default=str(REPO / "ADCNN/pipelines/heliolinc/cadence.csv"))
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--n-fields", type=int, default=6)
    ap.add_argument("--min-ecl-lat", type=float, default=20.0)
    ap.add_argument("--min-visits", type=int, default=20)
    ap.add_argument("--out-dir", default=str(REPO / "ADCNN/pipelines/heliolinc/run_realfp"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    c = pd.read_csv(a.cadence)
    sc = SkyCoord(ra=c.ra.values * u.deg, dec=c.dec.values * u.deg)
    c["ecl_lat"] = sc.barycentrictrueecliptic.lat.deg
    off = c[(c.ecl_lat.abs() > a.min_ecl_lat) & (c.n_visits >= a.min_visits)].sort_values("n_visits", ascending=False)

    STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
    b = Butler("dp2_prep")
    skymap = b.get("skyMap", skymap=a.skymap, collections="skymaps")
    rows = []
    used_tracts = set()
    k = 0
    for _, fld in off.iterrows():
        if k >= a.n_fields:
            break
        ra, dec, night = float(fld.ra), float(fld.dec), int(fld.night)
        tract = skymap.findTract(SpherePoint(ra * degrees, dec * degrees)).getId()
        if tract in used_tracts:           # distinct tracts -> independent FP fields
            continue
        used_tracts.add(tract)
        # one efficient query: all diffim panels for this tract on this night (day_obs == night)
        refs = list(b.registry.queryDatasets("difference_image", collections=STAGE4, findFirst=True,
                    where=(f"instrument='LSSTCam' AND skymap='{a.skymap}' AND tract={tract} "
                           f"AND visit.day_obs={night}")))
        man = [dict(visit=int(r.dataId["visit"]), detector=int(r.dataId["detector"]),
                    band=r.dataId.get("band", ""), fits_path=b.getURI(r).ospath) for r in refs]
        mdf = pd.DataFrame(man).drop_duplicates(["visit", "detector"])
        nv = mdf.visit.nunique() if len(mdf) else 0
        if nv < a.min_visits:
            continue
        mdf.insert(0, "image_id", range(len(mdf)))
        mdf.to_csv(out / f"manifest_{k}.csv", index=False)
        rows.append(dict(field=k, tract=tract, night=night, ra=round(ra, 2), dec=round(dec, 2),
                         ecl_lat=round(float(fld.ecl_lat), 1), n_visits=nv, n_panels=len(mdf), pairs=nv - 1))
        print(f"[realfp] field {k}: tract {tract} night {night} ecllat {fld.ecl_lat:.0f} | "
              f"{nv} visits, {len(mdf)} panels, ~{nv-1} pairs -> manifest_{k}.csv", flush=True)
        k += 1
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "fields.csv", index=False)
    print(f"[realfp] {len(summ)} fields | total {summ.pairs.sum()} pairs | {summ.n_panels.sum()} panels -> {out}/fields.csv", flush=True)


if __name__ == "__main__":
    main()
