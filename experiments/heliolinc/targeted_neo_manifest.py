"""Build a TARGETED diffim manifest covering only the panels that known NEO-rate objects cross
(their thin ephemeris arcs), so a cheap ADCNN run can test NEO recovery without processing the whole
strip (~230k panels/2wk). Uses preloaded_ss_object_visit (the same DM-53881 collection as the
diffims): identify NEO-rate objects from their ephemeris track, then for each appearance map
(ra,dec)->detector via visit_detector_region and collect the diffim FITS panel.

Outputs: <run>/manifest.csv (image_id,visit,detector,band,tract,fits_path) and
         <run>/neo_truth.csv (ObjID,mjd,ra,dec) for the NEO-rate objects (recovery crossmatch).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from lsst.daf.butler import Butler
import lsst.sphgeom as sph

STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"


def box(ra0, ra1, dec0, dec1):
    pts = [sph.UnitVector3d(sph.LonLat.fromDegrees(ra, dec))
           for ra, dec in [(ra0, dec0), (ra1, dec0), (ra1, dec1), (ra0, dec1)]]
    return sph.ConvexPolygon(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra0", type=float, default=295); ap.add_argument("--ra1", type=float, default=320)
    ap.add_argument("--dec0", type=float, default=-25); ap.add_argument("--dec1", type=float, default=-15)
    ap.add_argument("--day-start", type=int, default=20250620); ap.add_argument("--day-end", type=int, default=20250720)
    ap.add_argument("--min-rate", type=float, default=0.5, help="NEO-rate cut (deg/day)")
    ap.add_argument("--exclude", default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/train_visit_detector.csv")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    b = Butler("dp2_prep")
    strip = box(a.ra0, a.ra1, a.dec0, a.dec1)

    # visits overlapping the strip in the window
    vrecs = list(b.registry.queryDimensionRecords("visit", instrument="LSSTCam",
                 where=f"visit.day_obs>={a.day_start} AND visit.day_obs<{a.day_end} "
                       "AND visit.region OVERLAPS my_region", bind={"my_region": strip}))
    visits = sorted(set(r.id for r in vrecs))
    print(f"[targeted] {len(visits)} strip visits in [{a.day_start},{a.day_end})", flush=True)

    # pass 1: accumulate every known object's ephemeris points in these visits
    frames = []
    for i, v in enumerate(visits):
        refs = list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=STAGE4,
                    where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
        if not refs:
            continue
        t = b.get(refs[0]).to_pandas()
        t = t[(t.RA_deg >= a.ra0) & (t.RA_deg <= a.ra1) & (t.Dec_deg >= a.dec0) & (t.Dec_deg <= a.dec1)]
        if len(t):
            frames.append(pd.DataFrame({"ObjID": t.ObjID.astype(str), "visit": int(v),
                                        "mjd": t.fieldMJD_TAI.astype(float),
                                        "ra": t.RA_deg.astype(float), "dec": t.Dec_deg.astype(float)}))
        if i % 200 == 0:
            print(f"  ...{i}/{len(visits)} visits", flush=True)
    eph = pd.concat(frames, ignore_index=True)
    print(f"[targeted] {len(eph)} ephemeris points, {eph.ObjID.nunique()} known objects in strip+window", flush=True)

    # per-object on-sky rate -> NEO-rate set
    def rate(g):
        if len(g) < 2: return np.nan
        g = g.sort_values("mjd"); dt = g.mjd.diff()
        sep = np.hypot(g.ra.diff() * np.cos(np.radians(g.dec)), g.dec.diff())
        r = (sep / dt)[(dt > 0.001) & (dt < 2.0)]
        return r.median() if len(r) else np.nan
    rt = eph.groupby("ObjID").apply(rate, include_groups=False)
    nights = eph.assign(n=eph.mjd.astype(int)).groupby("ObjID").n.nunique()
    neo = sorted(rt[(rt >= a.min_rate)].index)
    neo_link = sorted(rt[(rt >= a.min_rate) & (nights >= 2)].index)
    print(f"[targeted] NEO-rate(>= {a.min_rate}) objects: {len(neo)} | with >=2 nights: {len(neo_link)}", flush=True)
    en = eph[eph.ObjID.isin(neo)].copy()

    # pass 2: for each NEO appearance, find the diffim detector containing it
    excl = set()
    if Path(a.exclude).exists():
        ex = pd.read_csv(a.exclude); excl = set(zip(ex.visit, ex.detector))
    panels = {}
    for v, gv in en.groupby("visit"):
        recs = list(b.registry.queryDimensionRecords("visit_detector_region", instrument="LSSTCam", where=f"visit={int(v)}"))
        regs = [(r.detector, r.region) for r in recs]
        for _, row in gv.iterrows():
            uv = sph.UnitVector3d(sph.LonLat.fromDegrees(row.ra, row.dec))
            for det, reg in regs:
                if reg.contains(uv):
                    if (int(v), int(det)) not in excl:
                        panels[(int(v), int(det))] = None
                    break
    print(f"[targeted] {len(panels)} unique (visit,detector) panels along NEO arcs", flush=True)

    # resolve diffim FITS path per panel
    rows = []
    for (v, det) in panels:
        for r in b.registry.queryDatasets("difference_image", collections=STAGE4,
                  where=f"instrument='LSSTCam' AND visit={v} AND detector={det}", findFirst=True):
            band = r.dataId.get("band", "r")
            rows.append((v, det, str(band), r.dataId.get("tract", -1), b.getURI(r).path))
            break
    man = pd.DataFrame(rows, columns=["visit", "detector", "band", "tract", "fits_path"]).drop_duplicates("fits_path")
    man.insert(0, "image_id", range(len(man)))
    man.to_csv(run / "manifest.csv", index=False)
    en[en.ObjID.isin(neo_link)][["ObjID", "mjd", "ra", "dec"]].to_csv(run / "neo_truth.csv", index=False)
    print(f"[targeted] manifest: {len(man)} panels (excluded {len(excl)&0} train) -> {run}/manifest.csv")
    print(f"[targeted] neo_truth: {len(neo_link)} linkable NEO-rate objects -> {run}/neo_truth.csv")


if __name__ == "__main__":
    main()
