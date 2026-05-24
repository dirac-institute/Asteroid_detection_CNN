"""Experiment: speed up the GPU forward WITHOUT changing detections, validated on the
held-out shard_3 val panels (leak-free). Variants:
  baseline  : scripted v7, autocast (current)
  jit-opt   : torch.jit.freeze + optimize_for_inference (conv-bn fusion; numerically equal)
  fp16      : full model.half() + half inputs (more aggressive; must validate)
Reports TP/FP/recall + s/panel for each; we keep only what holds TP and doesn't add FP.
"""
import sys, time
from pathlib import Path
import numpy as np, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
import ADCNN.inference.predict as P
P._TILE_BATCH=64; P._PREP_WORKERS=8
from ADCNN.inference.predict import predict_panel_overlap_3ch_full, hann2d, _tile_starts, _PREP_WORKERS
from ADCNN.data.preprocessing import build_3channel, diffim_mad_sigma
from ADCNN.inference.rf_postproc import RF_FEATURES_V2, compute_v2_features, apply_rf_v2, load_rf, DEFAULT_THR
from ADCNN.evaluation.catalog_match import match_trail_catalogs
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

dev=torch.device("cuda"); torch.backends.cudnn.benchmark=True
RF=load_rf(str(REPO/"models/rf_postproc.pkl"))
VAL_H5=str(REPO/"DATA_DIFFIM_realistic/shard_3/train.h5")
truth=pd.read_csv(REPO/"DATA_DIFFIM_realistic/shard_3_val.csv"); VAL_IDS=sorted(truth.image_id.unique())

def infer_fp16(model, img, rl, tile=128, stride=64, clip=5.0):
    """Full-fp16 sliding-window forward (model + inputs in half)."""
    H,W=img.shape; s=min(1024,H,W); sig=diffim_mad_sigma(img[(H-s)//2:(H-s)//2+s,(W-s)//2:(W-s)//2+s])
    pa=np.zeros((H,W),np.float32);sa=pa.copy();ca=pa.copy();ga=pa.copy();wa=pa.copy();hann=hann2d(tile)
    ys=_tile_starts(H,tile,stride);xs=_tile_starts(W,tile,stride);coords=[(y,x) for y in ys for x in xs]
    def b(loc):
        y,x=loc;return build_3channel(img[y:y+tile,x:x+tile],rl[y:y+tile,x:x+tile],panel_sigma=sig,clip=clip)
    with ThreadPoolExecutor(max_workers=8) as ex: x3s=list(ex.map(b,coords))
    for i in range(0,len(coords),64):
        xb=torch.from_numpy(np.stack(x3s[i:i+64])).to(dev).half()
        sl,sn,cs,_,ag=model(xb)
        pr=torch.sigmoid(sl).float().cpu().numpy();sn=sn.float().cpu().numpy();cs=cs.float().cpu().numpy();ag=ag.float().cpu().numpy()
        for (y,x),p,s_,c_,a_ in zip(coords[i:i+64],pr[:,0],sn[:,0],cs[:,0],ag[:,0]):
            pa[y:y+tile,x:x+tile]+=p*hann;sa[y:y+tile,x:x+tile]+=s_*hann;ca[y:y+tile,x:x+tile]+=c_*hann;ga[y:y+tile,x:x+tile]+=a_*hann;wa[y:y+tile,x:x+tile]+=hann
    w=np.maximum(wa,1e-6)
    return (pa/w).astype(np.float16),(sa/w).astype(np.float16),(ca/w).astype(np.float16),(ga/w).astype(np.float16)

@torch.no_grad()
def run(label, model, fp16=False):
    t0=time.time(); parts=[]
    with h5py.File(VAL_H5,"r") as f:
        for pid in VAL_IDS:
            img=f["images"][pid][:].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
            if fp16: prob,sn,cs,ag=infer_fp16(model,img,rl)
            else:    prob,sn,cs,ag=predict_panel_overlap_3ch_full(model,img,rl,device=dev)
            cand,_=compute_v2_features(prob[None],img[None],sn[None],cs[None],ag[None],real_labels=rl[None],gate_pmax=0.0,verbose=False)
            if not len(cand): continue
            cand[list(RF_FEATURES_V2)]=cand[list(RF_FEATURES_V2)].replace([np.inf,-np.inf],np.nan)
            cand=apply_rf_v2(cand,RF); cand=cand[cand.score_rf>=DEFAULT_THR].copy(); cand["image_id"]=int(pid)
            if len(cand): parts.append(cand.rename(columns={"x_centroid":"x","y_centroid":"y","or_beta":"beta","mf_length":"length"}))
    dt=time.time()-t0
    cat=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["image_id","x","y","beta","length"])
    _,_,c=match_trail_catalogs(cat,truth,tol_px=20.0)
    print(f"[{label:10s}] TP={c['TP']} FP={c['FP']} FN={c['FN']} recall={c['TP']/max(c['TP']+c['FN'],1):.3f} | {dt:.0f}s ({dt/len(VAL_IDS):.2f}s/panel)",flush=True)
    return c

base=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval()
cb=run("baseline", base)
try:
    opt=torch.jit.optimize_for_inference(torch.jit.freeze(base))
    co=run("jit-opt", opt)
    print(f"   jit-opt vs baseline: dTP={co['TP']-cb['TP']:+d} dFP={co['FP']-cb['FP']:+d}",flush=True)
except Exception as e:
    print("jit-opt FAILED:",repr(e),flush=True)
try:
    half=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval().half()
    ch=run("fp16", half, fp16=True)
    print(f"   fp16 vs baseline:    dTP={ch['TP']-cb['TP']:+d} dFP={ch['FP']-cb['FP']:+d}",flush=True)
except Exception as e:
    print("fp16 FAILED:",repr(e),flush=True)
print("EXPERIMENT DONE",flush=True)
