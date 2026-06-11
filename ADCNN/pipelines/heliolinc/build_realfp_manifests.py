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
    ap.add_argument("--max-per-tract", type=int, default=1,
                    help="max field-nights reused per tract (1 = distinct tracts, the old behavior; "
                         ">1 allows the same tract on different nights to reach more fields at scale)")
    ap.add_argument("--exclude-visits-from", default=None,
                    help="comma-separated globs of csvs with a 'visit' column; those visits are dropped "
                         "from every manifest (LEAKAGE: pass the train/train2 csvs)")
    ap.add_argument("--from-diffim-cadence", default=None,
                    help="use cadence_diffim.csv (tract,night,n_visits) as the source instead of the "
                         "pointing cadence.csv -- reaches the full off-ecliptic tract-night pool (~412)")
    ap.add_argument("--start-index", type=int, default=0, help="first manifest_<k> index (append to an existing run)")
    ap.add_argument("--exclude-fields-from", default=None,
                    help="csv with tract,night columns (a validation run's fields.csv): excluded per "
                         "--exclude-mode -- the blind-test disjointness rule (EVALUATION_CONTRACT.md)")
    ap.add_argument("--exclude-mode", default="tract", choices=["tract", "tract-night"],
                    help="tract = exclude validation TRACTS on any night (strict, tier 1); tract-night = "
                         "exclude only exact (tract,night) pairs (the contract's 'different nights where "
                         "possible' fallback tier when the strict pool is exhausted)")
    ap.add_argument("--max-ecl-lat", type=float, default=None,
                    help="ECLIPTIC mode: select |ecl_lat| <= this instead of > min-ecl-lat")
    ap.add_argument("--night-min", type=int, default=None, help="earliest night (diffim retention window)")
    ap.add_argument("--night-max", type=int, default=None, help="latest night (diffim retention window)")
    ap.add_argument("--collection", default="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4",
                    help="Butler collection serving difference_image (the DP2 stage4 default is being "
                         "decommissioned; DM-53195 d_2025_11_10 is the live replacement)")
    ap.add_argument("--out-dir", default=str(REPO / "ADCNN/pipelines/heliolinc/run_realfp"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    excl = set()
    if a.exclude_visits_from:
        import glob as _glob
        for pat in a.exclude_visits_from.split(","):
            for f in _glob.glob(pat.strip()):
                try:
                    excl |= set(pd.read_csv(f, usecols=["visit"]).visit.astype(int).unique())
                except Exception as e:
                    print(f"[realfp] WARN could not read {f}: {e}", flush=True)
        print(f"[realfp] excluding {len(excl)} leakage visits from {a.exclude_visits_from}", flush=True)

    STAGE4 = a.collection
    b = Butler("dp2_prep")
    skymap = b.get("skyMap", skymap=a.skymap, collections="skymaps")

    if a.from_diffim_cadence:
        # source = diffim-available (tract, night) pool; tract centre -> ra/dec/ecl_lat
        c = pd.read_csv(a.from_diffim_cadence)
        ctr = {int(t): skymap[int(t)].getCtrCoord() for t in c.tract.unique()}
        c["ra"] = c.tract.map(lambda t: ctr[int(t)].getRa().asDegrees())
        c["dec"] = c.tract.map(lambda t: ctr[int(t)].getDec().asDegrees())
        c["has_tract"] = True
    else:
        c = pd.read_csv(a.cadence); c["has_tract"] = False
    sc = SkyCoord(ra=c.ra.values * u.deg, dec=c.dec.values * u.deg)
    c["ecl_lat"] = sc.barycentrictrueecliptic.lat.deg
    lat_sel = (c.ecl_lat.abs() <= a.max_ecl_lat) if a.max_ecl_lat is not None else (c.ecl_lat.abs() > a.min_ecl_lat)
    off = c[lat_sel & (c.n_visits >= a.min_visits)].sort_values("n_visits", ascending=False)
    if a.night_min is not None:
        off = off[off.night >= a.night_min]
    if a.night_max is not None:
        off = off[off.night <= a.night_max]
    if a.exclude_fields_from:
        ex = pd.read_csv(a.exclude_fields_from)
        n0 = len(off)
        if a.exclude_mode == "tract":
            ex_tracts = set(ex.tract.astype(int))
            off = off[~off.tract.astype(int).isin(ex_tracts)]
            print(f"[realfp] BLIND disjointness (tract tier): excluded {len(ex_tracts)} validation tracts "
                  f"({n0}->{len(off)} candidates)", flush=True)
        else:
            ex_tn = set(zip(ex.tract.astype(int), ex.night.astype(int)))
            off = off[~off.apply(lambda r: (int(r.tract), int(r.night)) in ex_tn, axis=1)]
            print(f"[realfp] BLIND disjointness (tract-night FALLBACK tier): excluded {len(ex_tn)} exact "
                  f"(tract,night) pairs ({n0}->{len(off)} candidates)", flush=True)

    # when extending, drop (tract,night) already built so the new FP fields stay independent
    fcsv0 = out / "fields.csv"
    if a.start_index > 0 and fcsv0.exists() and "tract" in off.columns:
        try:
            prev = pd.read_csv(fcsv0)
        except pd.errors.EmptyDataError:
            prev = pd.DataFrame(columns=["tract", "night"])
        seen = set(zip(prev.tract.astype(int), prev.night.astype(int)))
        off = off[~off.apply(lambda r: (int(r.tract), int(r.night)) in seen, axis=1)]
        print(f"[realfp] extend: {len(off)} candidate (tract,night) after dropping {len(seen)} already-built", flush=True)

    rows = []
    from collections import Counter
    per_tract = Counter()
    k = 0
    for _, fld in off.iterrows():
        if k >= a.n_fields:
            break
        ra, dec, night = float(fld.ra), float(fld.dec), int(fld.night)
        tract = int(fld.tract) if fld.get("has_tract") else skymap.findTract(SpherePoint(ra * degrees, dec * degrees)).getId()
        if per_tract[tract] >= a.max_per_tract:   # cap reuse of a tract -> ~independent FP fields
            continue
        # one efficient query: all diffim panels for this tract on this night (day_obs == night)
        refs = list(b.registry.queryDatasets("difference_image", collections=STAGE4, findFirst=True,
                    where=(f"instrument='LSSTCam' AND skymap='{a.skymap}' AND tract={tract} "
                           f"AND visit.day_obs={night}")))
        man = [dict(visit=int(r.dataId["visit"]), detector=int(r.dataId["detector"]),
                    band=r.dataId.get("band", ""), fits_path=b.getURI(r).ospath) for r in refs]
        mdf = pd.DataFrame(man).drop_duplicates(["visit", "detector"])
        if len(mdf) and excl:                     # drop leakage (train/train2) visits
            mdf = mdf[~mdf.visit.isin(excl)]
        nv = mdf.visit.nunique() if len(mdf) else 0
        if nv < a.min_visits:
            continue
        per_tract[tract] += 1
        kk = a.start_index + k
        mdf.insert(0, "image_id", range(len(mdf)))
        mdf.to_csv(out / f"manifest_{kk}.csv", index=False)
        rows.append(dict(field=kk, tract=tract, night=night, ra=round(ra, 2), dec=round(dec, 2),
                         ecl_lat=round(float(fld.ecl_lat), 1), n_visits=nv, n_panels=len(mdf), pairs=nv - 1))
        print(f"[realfp] field {kk}: tract {tract} night {night} ecllat {fld.ecl_lat:.0f} | "
              f"{nv} visits, {len(mdf)} panels, ~{nv-1} pairs -> manifest_{kk}.csv", flush=True)
        k += 1
    summ = pd.DataFrame(rows)
    fcsv = out / "fields.csv"
    if a.start_index > 0 and fcsv.exists():        # appending to an existing run
        try:
            prev = pd.read_csv(fcsv)
            summ = pd.concat([prev, summ], ignore_index=True).drop_duplicates("field")
        except pd.errors.EmptyDataError:
            pass
    summ.to_csv(fcsv, index=False)
    tot_pairs = summ.pairs.sum() if len(summ) else 0
    tot_pan = summ.n_panels.sum() if len(summ) else 0
    print(f"[realfp] wrote {len(rows)} new fields (total {len(summ)}) | new pairs {sum(r['pairs'] for r in rows)} | "
          f"grand total {tot_pairs} pairs, {tot_pan} panels -> {fcsv}", flush=True)


if __name__ == "__main__":
    main()
