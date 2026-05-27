"""Rejecter build Stage B (lsst env): add Veres trailed-source-fit features per candidate, parallel
over panels. Veres is fit on the test.h5 INJECTED image (where v7's candidate x,y live), using the
butler PSF. Adds veres_x/y/len/theta/rchi/ok + offsets. Writes candB.parquet."""
import sys, warnings, os, numpy as np, pandas as pd, h5py
warnings.simplefilter("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
DATA=REPO/"DATA_DIFFIM/test_5sigma"; OUT=REPO/"experiments/heliolinc/rejecter_data"
H5=str(DATA/"test.h5")
STAGE4="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
_B=None
def _init():
    global _B
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
    from lsst.daf.butler import Butler; _B=Butler("dp2_prep",collections=[STAGE4])

def _panel(args):
    pid,v,det,dets=args
    import scipy.optimize as sciOpt, lsst.afw.image as afwImage, lsst.geom as geom
    from lsst.meas.extensions.trailedSources import VeresModel
    with h5py.File(H5,"r") as f: img=np.nan_to_num(f["images"][int(pid)].astype(np.float32))
    H,W=img.shape
    try:
        psf=_B.get("difference_image.psf",dataId={"instrument":"LSSTCam","visit":int(v),"detector":int(det)})
        psf_sig=float(psf.computeShape(psf.getAveragePosition()).getDeterminantRadius())
    except Exception: psf=None; psf_sig=2.0
    med=np.median(img); var_const=max(float(np.median(np.abs(img-med))*1.4826)**2,1e-3)
    exp=afwImage.ExposureF(W,H); exp.image.array[:]=img; exp.variance.array[:]=var_const
    if psf is not None: exp.setPsf(psf)
    res=[]
    for d in dets:
        x,y,cid=float(d["x"]),float(d["y"]),int(d["candidate_id"])
        rec=dict(panel_id=int(pid),candidate_id=cid,veres_x=np.nan,veres_y=np.nan,veres_len=np.nan,
                 veres_theta=np.nan,veres_rchi=np.nan,veres_ok=0)
        L0=max((float(d["mf_length"])-33.4)/0.887,2.0); half=int(L0/2+6*psf_sig+6)
        bb=geom.Box2I(geom.Point2I(int(x)-half,int(y)-half),geom.Extent2I(2*half+1,2*half+1)); bb.clip(exp.getBBox())
        if bb.getWidth()>=8 and bb.getHeight()>=8 and psf is not None:
            cut=exp.Factory(exp,bb); model=VeresModel(cut)
            seed=np.array([x,y,max(float(d["mf_snr"])*100,1000.0),L0,np.radians(float(d["mf_beta"]))])
            bnds=[(x-15,x+15),(y-15,y+15),(0.0,1e7),(1.0,300.0),(-np.pi,np.pi)]
            try:
                r=sciOpt.minimize(model,seed,method="L-BFGS-B",jac=model.gradient,bounds=bnds,options=dict(maxiter=500))
                xc,yc,flux,Lf,th=r.x; rchi=float(r.fun/max(cut.image.array.size-6,1))
                if np.isfinite(Lf) and 2.0<=Lf<=295.0:
                    rec.update(veres_x=float(xc),veres_y=float(yc),veres_len=float(abs(Lf)),
                               veres_theta=float(np.degrees(th)%180),veres_rchi=rchi,veres_ok=1)
            except Exception: pass
        res.append(rec)
    return res

def main():
    cand=pd.read_parquet(OUT/"candA_masked.parquet")   # carries the mask-bit features through
    cat=pd.read_csv(DATA/"test.csv"); pmap=cat.drop_duplicates("image_id").set_index("image_id")[["visit","detector"]].to_dict("index")
    tasks=[]
    for pid,g in cand.groupby("panel_id"):
        vd=pmap.get(int(pid));
        if vd is None: continue
        dets=g[["x","y","candidate_id","mf_length","mf_beta","mf_snr"]].to_dict("records")
        tasks.append((int(pid),int(vd["visit"]),int(vd["detector"]),dets))
    print(f"panels={len(tasks)} | candidates={len(cand)} | fitting Veres (parallel)...",flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC","48")),initializer=_init) as ex:
        for i,r in enumerate(ex.map(_panel,tasks)):
            rows.extend(r)
            if i%10==0: print(f"  {i}/{len(tasks)} panels",flush=True)
    vf=pd.DataFrame(rows)
    out=cand.merge(vf,on=["panel_id","candidate_id"],how="left")
    out["veres_dpos"]=np.hypot(out.veres_x-out.x, out.veres_y-out.y)
    out["veres_dlen"]=out.veres_len-out._len
    out.to_parquet(OUT/"candB.parquet")
    print(f"Veres ok: {int(vf.veres_ok.sum())}/{len(vf)} ({100*vf.veres_ok.mean():.0f}%) | -> {OUT}/candB.parquet ({len(out)} rows, {out.shape[1]} cols)")

if __name__=="__main__": main()
