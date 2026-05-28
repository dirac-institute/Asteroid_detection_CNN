"""Rejecter build Stage A (torch env): run v7 over the test_5sigma panels, extract candidate
detections + FEATURES_V2 + injection labels (1=injected TP, 0=FP), plus panel-clumping/isolation
context features. Writes candA.parquet for Stage B (Veres + mask features, lsst env).
Panel-disjoint train/val is done later at training time (GroupKFold on panel_id) -- leak-free."""
import sys, numpy as np, pandas as pd, h5py, torch
from pathlib import Path
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.rf_train import infer_candidate_features
from ADCNN.inference.features import FEATURES_V2

H5=REPO/"DATA_DIFFIM/test_5sigma/test.h5"
CSV=REPO/"DATA_DIFFIM/test_5sigma/test.csv"
V7=REPO/"models/v7_diffim_scripted.pt"
OUT=REPO/"experiments/heliolinc/rejecter_data"; OUT.mkdir(exist_ok=True)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

cat=pd.read_csv(CSV)
with h5py.File(H5,"r") as f: npan=f["images"].shape[0]
panel_ids=list(range(npan))
print(f"panels={npan} | injections={len(cat)}", flush=True)
model=torch.jit.load(str(V7),map_location=dev).eval()
cand,labels=infer_candidate_features(model,str(H5),panel_ids,cat,dev)
cand["label"]=labels
cand.to_parquet(OUT/"candA_raw.parquet")   # checkpoint the v7 work before ctx
print(f"candidates={len(cand)} | TP(injected)={int((labels==1).sum())} | FP={int((labels==0).sum())}", flush=True)

# centroid is x_centroid/y_centroid; length/orient seeds = mf_length/mf_beta
cand["x"]=cand["x_centroid"]; cand["y"]=cand["y_centroid"]; cand["_len"]=cand["mf_length"]
def add_ctx(g):
    n=len(g); longn=int((g._len>=15).sum())
    g=g.copy(); g["panel_nlong"]=longn; g["panel_ncand"]=n
    xy=g[["x","y"]].to_numpy()
    if n>1:
        from scipy.spatial import cKDTree
        d,_=cKDTree(xy).query(xy,k=2); g["nn_dist"]=d[:,1]
    else: g["nn_dist"]=9999.0
    return g
cand=cand.groupby("panel_id",group_keys=False).apply(add_ctx)
cand["is_long_clumped"]=((cand._len>=15)&(cand.panel_nlong>=3)).astype(int)

cand.to_parquet(OUT/"candA.parquet")
print(f"feature cols: {len(FEATURES_V2)} RF + ctx(panel_nlong,panel_ncand,nn_dist,is_long_clumped)")
print(f"-> {OUT}/candA.parquet  ({len(cand)} rows, cols={cand.shape[1]})")
