"""Regenerate the Evaluation detection catalogs using the focal-loss cutout CNN as stage-2 (instead
of the RF). Same v7 candidates + same public schema; the 'score_rf' column now holds the CNN score and
rows are kept at the CNN operating threshold. Output: Evaluation/catalogs/<set>_detections.csv (RF
version backed up to <set>_detections_rf.csv). Usage: python cnn_make_catalog.py <set> [--thr 0.0695]"""
import sys, argparse, shutil, numpy as np, pandas as pd, h5py, torch, torch.nn as nn
from pathlib import Path
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.features import compute_v2_features
from ADCNN.inference.catalog import _COLMAP
ap=argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--thr",type=float,default=0.0695); ap.add_argument("--k",type=int,default=48)
a=ap.parse_args(); K=a.k; Hh=K//2
DD=REPO/"DATA_DIFFIM"/a.set; CATS=REPO/"Evaluation/catalogs"; dev=torch.device("cuda")
class Net(nn.Module):
    def __init__(s,c=3,w=40):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(c,w),blk(w,2*w),blk(2*w,4*w),nn.AdaptiveAvgPool2d(1)); s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(4*w,1))
    def forward(s,x): return s.h(s.f(x)).squeeze(1)
cnn=Net().to(dev); cnn.load_state_dict(torch.load(REPO/"experiments/heliolinc/rejecter_data/cnn_focal_final.pt",map_location=dev)); cnn.eval()
v7=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval()
def cut(arr,x,y):
    H2,W2=arr.shape; x,y=int(round(x)),int(round(y)); o=np.zeros((K,K),np.float32)
    x0,x1,y0,y1=max(0,x-Hh),min(W2,x+Hh),max(0,y-Hh),min(H2,y+Hh); c=arr[y0:y1,x0:x1]; o[:c.shape[0],:c.shape[1]]=c; return o
rows=[]
with h5py.File(DD/"test.h5","r") as f:
    npan=f["images"].shape[0]
    for pid in range(npan):
        img=f["images"][pid].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
        prob,sn,cs,agg=predict_panel_overlap_3ch_full(v7,img,rl,device=dev); prob=prob.astype(np.float32); agg=np.asarray(agg,np.float32)
        cand,_=compute_v2_features({pid:prob},{pid:img},{pid:sn},{pid:cs},{pid:agg},real_labels={pid:rl},verbose=False)
        if not len(cand): continue
        s=float(np.median(np.abs(img-np.median(img)))*1.4826) or 1.0
        Xc=np.stack([np.stack([cut(img,r.x_centroid,r.y_centroid)/s,cut(prob,r.x_centroid,r.y_centroid),cut(agg,r.x_centroid,r.y_centroid)]) for _,r in cand.iterrows()])
        with torch.no_grad():
            sc=torch.sigmoid(torch.cat([cnn(torch.tensor(np.clip(Xc[k:k+512],-20,20)).to(dev)) for k in range(0,len(Xc),512)])).cpu().numpy()
        cand=cand.assign(score_rf=sc, image_id=pid); cand=cand[cand.score_rf>=a.thr]
        if len(cand): rows.append(cand[[c for c in _COLMAP if c in cand.columns]].rename(columns=_COLMAP))
        if pid%10==0: print(f"  panel {pid}: kept {len(cand)}",flush=True)
out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=list(_COLMAP.values()))
dst=CATS/f"{a.set}_detections.csv"; rfbak=CATS/f"{a.set}_detections_rf.csv"
if dst.exists() and not rfbak.exists(): shutil.copy(dst,rfbak)
out.to_csv(dst,index=False)
print(f"{a.set}: CNN catalog {len(out)} detections (thr {a.thr}) -> {dst} (RF backed up to {rfbak.name})")
