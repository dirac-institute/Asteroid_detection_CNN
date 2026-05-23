"""Stage A (torch/GPU, no Butler): run reg2 v7 + reg2 RF on real diffim panels,
save kept candidate centroids -> candidates.parquet. Stage B adds WCS->RA/Dec."""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"experiments/explore_simreal_gap"))
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import compute_v2_features, apply_rf_v2, load_rf, RF_FEATURES_V2
REG2=REPO/"experiments/diffim_runs/pilot_v7_reg2/ckpts/v7_reg2_best_scripted.pt"
RF=REPO/"experiments/explore_simreal_gap/rf_postproc_v2_reg2_neg5.pkl"
H5=REPO/"DATA_DIFFIM/test_real/test.h5"; PANELS=REPO/"DATA_DIFFIM/test_real/panels.csv"
ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--rf-thr",type=float,default=0.5)
ap.add_argument("--role",default="asteroid"); ap.add_argument("--out",default=str(REPO/"experiments/heliolinc/run_adcnn/candidates.parquet")); a=ap.parse_args()
dev=torch.device("cuda"); model=torch.jit.load(str(REG2),map_location=dev).eval(); rf=load_rf(str(RF))
pan=pd.read_csv(PANELS); pan=pan[pan.role==a.role] if a.role!="all" else pan
if a.limit: pan=pan.head(a.limit)
rows=[]
with h5py.File(H5,"r") as f:
    for n,(_,p) in enumerate(pan.iterrows()):
        idx=int(p.image_id)
        img=f["images"][idx][:].astype(np.float32); rl=f["real_labels"][idx][:].astype(np.uint16)
        prob,sin,cos,agg=predict_panel_overlap_3ch_full(model,img,rl,device=dev)
        cand,_=compute_v2_features(prob[None],img[None],sin[None],cos[None],agg[None],real_labels=rl[None],verbose=False)
        if not len(cand): continue
        cand[RF_FEATURES_V2]=cand[RF_FEATURES_V2].replace([np.inf,-np.inf],np.nan)
        cand=apply_rf_v2(cand,rf); keep=cand[cand.score_rf>=a.rf_thr]
        for _,c in keep.iterrows():
            rows.append(dict(image_id=idx,visit=int(p.visit),detector=int(p.detector),band=str(p.band),
                             x_centroid=float(c.x_centroid),y_centroid=float(c.y_centroid),score_rf=float(c.score_rf)))
        if (n+1)%100==0: print(f"  {n+1}/{len(pan)} panels, {len(rows)} kept cands",flush=True)
out=pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(a.out)
print(f"[stageA] {len(out)} candidates from {len(pan)} panels -> {a.out}",flush=True); print("STAGEA DONE",flush=True)
