"""Stage B (lsst_distrib, Butler): read candidates.parquet (Stage A), fetch the
visit-detector WCS + MJD from the Butler, convert candidate (x,y)->(RA,Dec), write
a HelioLinC detection catalog. --validate compares detected RA/Dec to truth RA/Dec."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
import lsst.geom as geom
from lsst.daf.butler import Butler
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE3="LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"; STAGE2="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
ap=argparse.ArgumentParser(); ap.add_argument("--cands",default=str(REPO/"experiments/heliolinc/run_adcnn/candidates.parquet"))
ap.add_argument("--out",default=str(REPO/"experiments/heliolinc/run_adcnn/adcnn_dets.csv")); ap.add_argument("--validate",action="store_true")
a=ap.parse_args()
c=pd.read_parquet(a.cands); b=Butler("dp2_prep",collections=[STAGE3,STAGE2])
truth=pd.read_csv(REPO/"experiments/explore_simreal_gap/test_real_realistic/per_sighting_forced_lsst.csv").dropna(subset=["ra","dec"]) if a.validate else None
rows=[]; wcs_cache={}
for (visit,det),grp in c.groupby(["visit","detector"]):
    try:
        pvi=b.get("preliminary_visit_image",dataId={"instrument":"LSSTCam","visit":int(visit),"detector":int(det)})
        wcs=pvi.getWcs(); xy0=pvi.getBBox().getBegin(); mjd=pvi.getInfo().getVisitInfo().getDate().get()
    except Exception as e:
        print(f"  WCS fail v={visit} d={det}: {e}",flush=True); continue
    for _,r in grp.iterrows():
        sp=wcs.pixelToSky(geom.Point2D(r.x_centroid+xy0.getX(), r.y_centroid+xy0.getY()))
        rows.append(dict(detid=len(rows),mjd=mjd,ra=sp.getRa().asDegrees(),dec=sp.getDec().asDegrees(),
                         mag=21.5,band=str(r.band)[0],obscode="I11",visit=int(visit),detector=int(det),
                         x=r.x_centroid,y=r.y_centroid,score_rf=r.score_rf))
out=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
open(str(Path(a.out).parent/"colformat.txt"),"w").write("IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
print(f"[stageB] wrote {len(out)} detections (RA/Dec) -> {a.out}",flush=True)
if a.validate and truth is not None and len(out):
    # match each detection to nearest truth sighting on same visit-detector; report WCS sep + recovery
    matched=0; seps=[]
    for (v,d),g in out.groupby(["visit","detector"]):
        ts=truth[(truth.visit==v)&(truth.detector==d)]
        for _,t in ts.iterrows():
            sep=np.hypot((g.ra-t.ra)*np.cos(np.radians(t.dec)),g.dec-t.dec)*3600
            if sep.min()<5: matched+=1; seps.append(sep.min())
    print(f"[validate] {matched} truth sightings matched a detection within 5\"; median sep {np.median(seps):.2f}\"" if seps else "no matches",flush=True)
print("STAGEB DONE",flush=True)
