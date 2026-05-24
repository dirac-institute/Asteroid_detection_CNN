"""LEAK-FREE selection of the candidate-gate (gate_pmax) + stride on the held-out VAL
panels (shard_3 val, disjoint from test). Reports recall/FP/time vs the no-gate baseline;
we pick the fastest config that loses no true positives. Test sets are NOT touched here.
"""
import sys, time
from pathlib import Path
import pandas as pd
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.catalog import build_detection_catalog
from ADCNN.evaluation.catalog_match import match_trail_catalogs
from ADCNN.inference.rf_postproc import DEFAULT_THR

V7=REPO/"models/v7_diffim_scripted.pt"; RF=REPO/"models/rf_postproc.pkl"
VAL_H5=REPO/"DATA_DIFFIM_realistic/shard_3/train.h5"
VAL_CSV=REPO/"DATA_DIFFIM_realistic/shard_3_val.csv"
TOL=20.0

def run():
    truth=pd.read_csv(VAL_CSV)
    val_ids=sorted(truth.image_id.unique())
    npan=len(val_ids)
    configs=[("baseline",0.0,64),("gate0.05",0.05,64),("gate0.10",0.10,64),
             ("gate0.20",0.20,64),("stride96",0.0,96),("gate0.10+stride96",0.10,96)]
    base_tp=None
    print(f"VAL leak-free sweep on {npan} held-out shard_3 val panels (ids {val_ids[0]}..{val_ids[-1]})",flush=True)
    for nm,gate,stride in configs:
        t0=time.time()
        cat=build_detection_catalog(str(VAL_H5),str(V7),str(RF),rf_thr=DEFAULT_THR,
                                    device="cuda",panel_ids=val_ids,gate_pmax=gate,stride=stride)
        dt=time.time()-t0
        _,_,c=match_trail_catalogs(cat,truth,tol_px=TOL)
        rec=c["TP"]/max(c["TP"]+c["FN"],1)
        if base_tp is None: base_tp=c["TP"]
        dtp=c["TP"]-base_tp
        print(f"[{nm:18s}] det={len(cat):5d} TP={c['TP']} (dTP={dtp:+d}) FP={c['FP']:5d} "
              f"recall={rec:.3f} | {dt:.0f}s ({dt/npan:.2f}s/panel)",flush=True)
    print("VAL-SWEEP DONE",flush=True)

if __name__=="__main__":
    run()
