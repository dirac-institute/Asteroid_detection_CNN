"""Rejecter Stage A.5 (lsst env): mask-filter the v7 candidates BEFORE Veres, and VALIDATE the filter
on this val set (exact injection labels). For each candidate, sample the butler diffim mask plane at
(x,y) + along the trail; flag instrumental-artifact bits (SPIKE/SAT/CR/... NOT the DETECTED 5-sigma
bit, NOT INJECTED). Report per-bit and combined FP-removed / TP-kept / sub-5sigma-kept, then write the
survivors for Veres. Butler diffim is CLEAN (injections are sim-only) so MP_INJECTED isn't in it ->
sampling at TP positions gives clean-background bits (no self-contamination)."""
import os, warnings, numpy as np, pandas as pd
warnings.simplefilter("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
DATA=REPO/"DATA_DIFFIM/test_5sigma"; OUT=REPO/"experiments/heliolinc/rejecter_data"
STAGE4="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
# TP-safe instrumental bits (exclude DETECTED=5 [5-sigma floor] and INJECTED=23 [the TP truth])
FILTER_BITS=["SAT","CR","EDGE","STREAK","SPIKE","SENSOR_EDGE","CROSSTALK","ITL_DIP","BAD","SUSPECT","DETECTED_NEGATIVE"]
_B=None;_BITS=None
def _init():
    global _B,_BITS
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
    from lsst.daf.butler import Butler; _B=Butler("dp2_prep",collections=[STAGE4])

def _panel(args):
    pid,v,det,dets=args
    try:
        exp=_B.get("difference_image",dataId={"instrument":"LSSTCam","visit":int(v),"detector":int(det)})
    except Exception:
        # diffim missing in collection -> no mask info; keep these candidates (mask_acc=0 -> not flagged)
        return [dict(panel_id=int(pid),candidate_id=int(d["candidate_id"]),mask_acc=0,
                     **{f"m_{nm}":0 for nm in FILTER_BITS}) for d in dets]
    msk=exp.mask.array; bd=exp.mask.getMaskPlaneDict(); H,W=msk.shape
    out=[]
    for d in dets:
        x,y=int(round(d["x"])),int(round(d["y"]))
        x0,x1=max(0,x-4),min(W,x+5); y0,y1=max(0,y-4),min(H,y+5)
        acc=int(np.bitwise_or.reduce(msk[y0:y1,x0:x1].ravel())) if (x1>x0 and y1>y0) else 0
        L=float(d["mf_length"]);
        if L==L and L>=4:
            th=np.radians(float(d["mf_beta"]))
            for t in np.linspace(-L/2,L/2,int(min(L,60))):
                xx,yy=int(round(x+t*np.cos(th))),int(round(y+t*np.sin(th)))
                if 0<=xx<W and 0<=yy<H: acc|=int(msk[yy,xx])
        rec=dict(panel_id=int(pid),candidate_id=int(d["candidate_id"]),mask_acc=acc)
        for nm in FILTER_BITS: rec[f"m_{nm}"]=int((acc>>bd[nm])&1) if nm in bd else 0
        out.append(rec)
    return out

def main():
    cand=pd.read_parquet(OUT/"candA.parquet")
    cat=pd.read_csv(DATA/"test.csv"); pmap=cat.drop_duplicates("image_id").set_index("image_id")[["visit","detector"]].to_dict("index")
    tasks=[]
    for pid,g in cand.groupby("panel_id"):
        vd=pmap.get(int(pid));
        if vd is None: continue
        tasks.append((int(pid),int(vd["visit"]),int(vd["detector"]),g[["x","y","candidate_id","mf_length","mf_beta"]].to_dict("records")))
    print(f"panels={len(tasks)} | candidates={len(cand)} | sampling masks (parallel)...",flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC","46")),initializer=_init) as ex:
        for i,r in enumerate(ex.map(_panel,tasks)): rows.extend(r)
    mf=pd.DataFrame(rows); d=cand.merge(mf,on=["panel_id","candidate_id"],how="left")
    # sub-5sigma TP tag (stack_detection=False) via nearest injection
    d["sub5"]=False
    for pid,g in d[d.label==1].groupby("panel_id"):
        inj=cat[cat.image_id==pid]
        if not len(inj): continue
        for idx,r in g.iterrows():
            dd=np.hypot(inj.x-r.x,inj.y-r.y)
            if dd.min()<10: d.loc[idx,"sub5"]=not bool(inj.loc[dd.idxmin(),"stack_detection"])
    ntp=int((d.label==1).sum()); nfp=int((d.label==0).sum()); nsub5=int(d[d.label==1].sub5.sum())
    print(f"\n=== mask-filter VALIDATION on val set ({ntp} TP, {nfp} FP, {nsub5} sub-5sigma TP) ===")
    print(f"{'bit':>16} {'%FP flagged':>12} {'%TP flagged':>12} {'%sub5 flagged':>14}")
    for nm in FILTER_BITS:
        col=f"m_{nm}"; on=d[col]==1
        print(f"{nm:>16} {100*on[d.label==0].mean():>12.2f} {100*on[d.label==1].mean():>12.2f} {100*on[(d.label==1)&d.sub5].mean():>14.2f}")
    d["mask_flag"]=(d[[f"m_{n}" for n in FILTER_BITS]].sum(axis=1)>0).astype(int)
    fpc=100*(d[d.label==0].mask_flag).mean(); tpc=100*(d[d.label==1].mask_flag).mean(); s5c=100*(d[(d.label==1)&d.sub5].mask_flag).mean()
    print(f"\nCOMBINED instrumental filter: removes {fpc:.1f}% of FP | costs {tpc:.2f}% of TP | {s5c:.2f}% of sub-5sigma TP")
    surv=d[d.mask_flag==0]
    print(f"survivors -> Veres: {len(surv)} of {len(d)} ({100*len(surv)/len(d):.0f}%) | TP kept {int((surv.label==1).sum())}/{ntp} | FP kept {int((surv.label==0).sum())}/{nfp}")
    d.to_parquet(OUT/"candA_masked.parquet")
    print(f"-> {OUT}/candA_masked.parquet (all cands + mask_flag)")

if __name__=="__main__": main()
