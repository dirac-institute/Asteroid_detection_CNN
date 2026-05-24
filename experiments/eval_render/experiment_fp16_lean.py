"""Lean check: isolated GPU-forward speedup + prob-map equivalence on val panels.
jit-opt is operator fusion -> if its prob map == baseline bit-for-bit, ΔTP/ΔFP=0 is proven.
fp16 perturbs the map -> we quantify the perturbation + flips at the candidate threshold."""
import sys, time
from pathlib import Path
import numpy as np, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
import ADCNN.inference.predict as P
P._TILE_BATCH=64; P._PREP_WORKERS=8
from ADCNN.inference.predict import predict_panel_overlap_3ch_full, hann2d, _tile_starts
from ADCNN.data.preprocessing import build_3channel, diffim_mad_sigma
from concurrent.futures import ThreadPoolExecutor
dev=torch.device("cuda"); torch.backends.cudnn.benchmark=True
VAL_H5=str(REPO/"DATA_DIFFIM_realistic/shard_3/train.h5"); VAL_IDS=list(range(1036,1060))  # 24 panels

def infer(model, img, rl, half=False, tile=128, stride=64, clip=5.0):
    H,W=img.shape; s=min(1024,H,W); sig=diffim_mad_sigma(img[(H-s)//2:(H-s)//2+s,(W-s)//2:(W-s)//2+s])
    pa=np.zeros((H,W),np.float32); wa=pa.copy(); hann=hann2d(tile)
    ys=_tile_starts(H,tile,stride); xs=_tile_starts(W,tile,stride); coords=[(y,x) for y in ys for x in xs]
    def b(loc):
        y,x=loc; return build_3channel(img[y:y+tile,x:x+tile],rl[y:y+tile,x:x+tile],panel_sigma=sig,clip=clip)
    with ThreadPoolExecutor(max_workers=8) as ex: x3s=list(ex.map(b,coords))
    for i in range(0,len(coords),64):
        xb=torch.from_numpy(np.stack(x3s[i:i+64])).to(dev)
        if half: xb=xb.half(); sl=model(xb)[0]
        else:
            with torch.amp.autocast("cuda"): sl=model(xb)[0]
        pr=torch.sigmoid(sl).float().cpu().numpy()
        for (y,x),p in zip(coords[i:i+64],pr[:,0]): pa[y:y+tile,x:x+tile]+=p*hann; wa[y:y+tile,x:x+tile]+=hann
    return (pa/np.maximum(wa,1e-6)).astype(np.float32)

@torch.no_grad()
def timeit(model, half=False, warmup=True):
    t_tot=0.0; maps=[]
    with h5py.File(VAL_H5,"r") as f:
        ids=VAL_IDS
        if warmup:  # cudnn.benchmark autotune on first call
            img=f["images"][ids[0]][:].astype(np.float32); rl=f["real_labels"][ids[0]][:].astype(np.uint16); infer(model,img,rl,half)
        for pid in ids:
            img=f["images"][pid][:].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
            torch.cuda.synchronize(); t0=time.time(); m=infer(model,img,rl,half); torch.cuda.synchronize(); t_tot+=time.time()-t0
            maps.append(m)
    return t_tot/len(VAL_IDS), maps

base=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval()
sb,mb=timeit(base); print(f"[baseline] {sb:.2f}s/panel",flush=True)
def cmp(mv, mb):
    d=[np.abs(a.astype(np.float32)-b.astype(np.float32)) for a,b in zip(mv,mb)]
    mx=max(x.max() for x in d); me=np.mean([x.mean() for x in d])
    flips=int(sum(((a>0.05)!=(b>0.05)).sum() for a,b in zip(mv,mb)))  # candidate-threshold pixel flips
    hi=int(sum(((a>0.5)!=(b>0.5)).sum() for a,b in zip(mv,mb)))       # high-conf pixel flips
    return mx,me,flips,hi
try:
    opt=torch.jit.optimize_for_inference(torch.jit.freeze(base)); so,mo=timeit(opt)
    mx,me,fl,hi=cmp(mo,mb); print(f"[jit-opt]  {so:.2f}s/panel  speedup={sb/so:.2f}x | max|dprob|={mx:.2e} mean={me:.2e} thr-flips={fl} hi-flips={hi}",flush=True)
except Exception as e: print("jit-opt FAILED:",repr(e),flush=True)
try:
    half=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval().half(); sh,mh=timeit(half,half=True)
    mx,me,fl,hi=cmp(mh,mb); print(f"[fp16]     {sh:.2f}s/panel  speedup={sb/sh:.2f}x | max|dprob|={mx:.2e} mean={me:.2e} thr-flips={fl} hi-flips={hi}",flush=True)
except Exception as e: print("fp16 FAILED:",repr(e),flush=True)
print("LEAN EXPERIMENT DONE",flush=True)
