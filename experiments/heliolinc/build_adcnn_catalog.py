"""END-TO-END ADCNN -> HelioLinC bridge: run reg2 v7 + reg2 RF on REAL diffim panels,
extract candidate centroids (x,y), convert to (RA,Dec) via the visit-detector WCS,
attach MJD/band/mag/obscode, and write a HelioLinC detection catalog. --validate mode
checks detected RA/Dec against the known truth RA/Dec (verifies the WCS conversion).
"""
import sys, argparse, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
import lsst.geom as geom
from lsst.daf.butler import Butler
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"experiments/explore_simreal_gap"))
sys.path.insert(0, str(REPO/"experiments/explore_rf_leakage"))
from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.rf_postproc import compute_v2_features, apply_rf_v2, load_rf, RF_FEATURES_V2

REG2 = REPO/"experiments/diffim_runs/pilot_v7_reg2/ckpts/v7_reg2_best_scripted.pt"
RF = REPO/"experiments/explore_simreal_gap/rf_postproc_v2_reg2_neg5.pkl"
H5 = REPO/"DATA_DIFFIM/test_real/test.h5"
PANELS = REPO/"DATA_DIFFIM/test_real/panels.csv"
STAGE3="LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"; STAGE2="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=20)
    ap.add_argument("--rf-thr",type=float,default=0.5); ap.add_argument("--validate",action="store_true")
    ap.add_argument("--out",default=str(REPO/"experiments/heliolinc/run_adcnn/adcnn_dets.csv")); a=ap.parse_args()
    dev=torch.device("cuda"); model=torch.jit.load(str(REG2),map_location=dev).eval(); rf=load_rf(str(RF))
    b=Butler("dp2_prep",collections=[STAGE3,STAGE2])
    pan=pd.read_csv(PANELS)
    truth=pd.read_csv(REPO/"experiments/explore_simreal_gap/test_real_realistic/per_sighting_forced_lsst.csv")
    sub=pan[pan.role=="asteroid"].head(a.limit) if a.validate else pan.head(a.limit)
    rows=[]; vchk=[]
    with h5py.File(H5,"r") as f:
        for _,p in sub.iterrows():
            idx=int(p.image_id); visit=int(p.visit); det=int(p.detector)
            img=f["images"][idx][:].astype(np.float32); rl=f["real_labels"][idx][:].astype(np.uint16)
            prob,sin,cos,agg=predict_panel_overlap_3ch_full(model,img,rl,device=dev)
            cand,_=compute_v2_features(prob[None],img[None],sin[None],cos[None],agg[None],real_labels=rl[None],verbose=False)
            if not len(cand): continue
            cand[RF_FEATURES_V2]=cand[RF_FEATURES_V2].replace([np.inf,-np.inf],np.nan)
            cand=apply_rf_v2(cand,rf); keep=cand[cand.score_rf>=a.rf_thr]
            if not len(keep): continue
            # WCS for this visit-detector (diffim shares the science PVI WCS + xy0)
            try:
                pvi=b.get("preliminary_visit_image",dataId={"instrument":"LSSTCam","visit":visit,"detector":det})
                wcs=pvi.getWcs(); xy0=pvi.getBBox().getBegin()
                mjd=pvi.getInfo().getVisitInfo().getDate().get()  # MJD
            except Exception as e:
                print(f"  panel {idx} WCS fail: {e}",flush=True); continue
            for _,c in keep.iterrows():
                sp=wcs.pixelToSky(geom.Point2D(c.x_centroid+xy0.getX(), c.y_centroid+xy0.getY()))
                ra,dec=sp.getRa().asDegrees(),sp.getDec().asDegrees()
                rows.append(dict(detid=len(rows),mjd=mjd,ra=ra,dec=dec,mag=21.5,band=str(p.band)[0],obscode="I11",
                                 image_id=idx,x=c.x_centroid,y=c.y_centroid,score_rf=c.score_rf))
            if a.validate:
                ts=truth[(truth.visit==visit)&(truth.detector==det)].dropna(subset=["ra","dec"])
                for _,t in ts.iterrows():
                    d=keep.assign(dra=(keep.x_centroid),dist=np.hypot(keep.x_centroid-t.x,keep.y_centroid-t.y))
                    near=d.sort_values("dist").iloc[0]
                    sp=wcs.pixelToSky(geom.Point2D(near.x_centroid+xy0.getX(),near.y_centroid+xy0.getY()))
                    vchk.append(dict(obj=t.ObjID,truth_ra=t.ra,truth_dec=t.dec,
                                     det_ra=sp.getRa().asDegrees(),det_dec=sp.getDec().asDegrees(),pix_dist=near.dist))
    out=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
    print(f"[adcnn] wrote {len(out)} detections from {sub.shape[0]} panels -> {a.out}",flush=True)
    if a.validate and vchk:
        v=pd.DataFrame(vchk); v["sep_arcsec"]=np.hypot((v.det_ra-v.truth_ra)*np.cos(np.radians(v.truth_dec)),v.det_dec-v.truth_dec)*3600
        nn=v[v.pix_dist<20]  # candidate within 20px of truth = a real detection match
        print(f"[validate] {len(v)} truth sightings on these panels; {len(nn)} have a candidate within 20px",flush=True)
        print(f"[validate] WCS sep (det vs truth ra/dec) for matched: median {nn.sep_arcsec.median():.2f}\" p90 {nn.sep_arcsec.quantile(.9):.2f}\"" if len(nn) else "no matches",flush=True)
    print("ADCNN CATALOG DONE",flush=True)

if __name__=="__main__": main()
