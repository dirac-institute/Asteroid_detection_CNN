"""For the top PROCESSABLE dense (tract,night) fields (cadence_diffim.csv), count recoverable FAINT
fast movers (>=3 same-night epochs, speed >= threshold, mag 21.5-24.5 = clean long-trail regime).
Picks the best field for the 3-visit discovery campaign."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from lsst.daf.butler import Butler
STAGE4="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"; SOLARDAY=86400.0

def visits_for(b, tract, night):
    vs=set()
    for did in b.registry.queryDataIds(["visit","tract"], datasets="difference_image", collections=STAGE4,
            where=f"instrument='LSSTCam' AND skymap='lsst_cells_v1' AND tract={tract} AND visit.day_obs={night}"):
        vs.add(int(did["visit"]))
    return sorted(vs)

def known(b, visits):
    fr=[]
    for v in visits:
        refs=list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=STAGE4,
            where=f"instrument='LSSTCam' AND visit={v}", findFirst=True))
        if not refs: continue
        t=b.get(refs[0]).to_pandas()
        fr.append(pd.DataFrame(dict(ObjID=t.ObjID.astype(str), mjd=t.fieldMJD_TAI.astype(float),
                  ra=t.RA_deg.astype(float), dec=t.Dec_deg.astype(float), mag=t.trailedSourceMag.astype(float))))
    return pd.concat(fr,ignore_index=True) if fr else pd.DataFrame(columns=["ObjID","mjd","ra","dec","mag"])

def stats(g):
    g=g.sort_values("mjd"); t=g.mjd.to_numpy()
    ep=1+int(np.sum(np.diff(t)*SOLARDAY>120)) if len(t)>1 else 1
    cd=np.cos(np.radians(g.dec.mean()))
    spd=float(np.nanmax(np.hypot(np.diff(g.ra.to_numpy())*cd,np.diff(g.dec.to_numpy()))/np.where(np.diff(t)>0,np.diff(t),1e9))) if len(t)>1 else 0.0
    return ep,spd,float(g.mag.mean())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cadence",default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/cadence_diffim.csv")
    ap.add_argument("--top",type=int,default=12); ap.add_argument("--cap-visits",type=int,default=30)
    ap.add_argument("--fast",type=float,default=1.5)
    ap.add_argument("--out",default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/field_proc_fastmovers.csv")
    a=ap.parse_args()
    cad=pd.read_csv(a.cadence).head(a.top)
    b=Butler("dp2_prep"); rows=[]
    for _,f in cad.iterrows():
        vs=visits_for(b,int(f.tract),int(f.night))[:a.cap_visits]
        k=known(b,vs); nf=nf3=nff3=0; ex=[]
        for o,g in k.groupby("ObjID"):
            ep,spd,mag=stats(g)
            if spd>=a.fast:
                nf+=1
                if ep>=3:
                    nf3+=1
                    if 21.5<=mag<=24.5: nff3+=1; ex.append(f"{o}({spd:.1f}d/d,{ep}ep,m{mag:.1f})")
        rows.append(dict(tract=int(f.tract),night=int(f.night),n_visits=int(f.n_visits),n_panels=int(f.n_panels),
                         n_fast3=nf3,n_faintfast3=nff3,examples="; ".join(ex[:6])))
        print(f"  tract {int(f.tract)} {int(f.night)} nv{int(f.n_visits)}: fast>=3ep={nf3} FAINTfast>=3ep={nff3}  {('; '.join(ex[:3]))}",flush=True)
    out=pd.DataFrame(rows).sort_values(["n_faintfast3","n_fast3"],ascending=False).reset_index(drop=True)
    out.to_csv(a.out,index=False); print("\n"+out.to_string(index=False),flush=True)

if __name__=="__main__": main()
