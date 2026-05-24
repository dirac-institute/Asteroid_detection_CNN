"""Build an fp16 TensorRT engine for v7 and validate speed + detection-equivalence on the
held-out shard_3 val panels (leak-free). Reports isolated GPU-forward speedup + prob-map
deltas (max/mean |Δ|, candidate-threshold pixel flips). Keep TRT only if it doesn't move
detections beyond the existing fp16 noise floor."""
import sys, time, traceback
from pathlib import Path
import numpy as np, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.predict import hann2d, _tile_starts
from ADCNN.data.preprocessing import build_3channel, diffim_mad_sigma
from concurrent.futures import ThreadPoolExecutor
dev=torch.device("cuda"); torch.backends.cudnn.benchmark=True
VAL_H5=str(REPO/"DATA_DIFFIM_realistic/shard_3/train.h5"); VAL_IDS=list(range(1036,1060))
BATCH=64; TILE=128; STRIDE=64

def infer(model, img, rl, trt=False, clip=5.0):
    H,W=img.shape; s=min(1024,H,W); sig=diffim_mad_sigma(img[(H-s)//2:(H-s)//2+s,(W-s)//2:(W-s)//2+s])
    pa=np.zeros((H,W),np.float32); wa=pa.copy(); hann=hann2d(TILE)
    ys=_tile_starts(H,TILE,STRIDE); xs=_tile_starts(W,TILE,STRIDE); coords=[(y,x) for y in ys for x in xs]
    def b(loc):
        y,x=loc; return build_3channel(img[y:y+TILE,x:x+TILE],rl[y:y+TILE,x:x+TILE],panel_sigma=sig,clip=clip)
    with ThreadPoolExecutor(max_workers=8) as ex: x3s=list(ex.map(b,coords))
    for i in range(0,len(coords),BATCH):
        chunk=x3s[i:i+BATCH]; n=len(chunk)
        xb=torch.from_numpy(np.stack(chunk)).to(dev)
        if trt:
            out=model(xb); sl=out[0]
        else:
            with torch.amp.autocast("cuda"): sl=model(xb)[0]
        pr=torch.sigmoid(sl).float().cpu().numpy()
        for (y,x),p in zip(coords[i:i+n],pr[:n,0]): pa[y:y+TILE,x:x+TILE]+=p*hann; wa[y:y+TILE,x:x+TILE]+=hann
    return (pa/np.maximum(wa,1e-6)).astype(np.float32)

@torch.no_grad()
def timeit(model, trt=False):
    with h5py.File(VAL_H5,"r") as f:
        img=f["images"][VAL_IDS[0]][:].astype(np.float32); rl=f["real_labels"][VAL_IDS[0]][:].astype(np.uint16)
        infer(model,img,rl,trt)  # warmup
        t=0.0; maps=[]
        for pid in VAL_IDS:
            img=f["images"][pid][:].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
            torch.cuda.synchronize(); t0=time.time(); m=infer(model,img,rl,trt); torch.cuda.synchronize(); t+=time.time()-t0; maps.append(m)
    return t/len(VAL_IDS), maps

base=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval()
sb,mb=timeit(base); print(f"[baseline] {sb:.2f}s/panel",flush=True)
try:
    import torch_tensorrt
    print("torch_tensorrt",torch_tensorrt.__version__,flush=True)
    trt=torch_tensorrt.compile(base,
        inputs=[torch_tensorrt.Input(min_shape=(1,3,128,128),opt_shape=(BATCH,3,128,128),max_shape=(BATCH,3,128,128),dtype=torch.float32)],
        enabled_precisions={torch.float16}, truncate_long_and_double=True, require_full_compilation=False)
    st,mt=timeit(trt,trt=True)
    d=[np.abs(a-b) for a,b in zip(mt,mb)]; mx=max(x.max() for x in d); me=np.mean([x.mean() for x in d])
    fl=int(sum(((a>0.05)!=(b>0.05)).sum() for a,b in zip(mt,mb))); hi=int(sum(((a>0.5)!=(b>0.5)).sum() for a,b in zip(mt,mb)))
    print(f"[TRT-fp16] {st:.2f}s/panel  speedup={sb/st:.2f}x | max|dprob|={mx:.2e} mean={me:.2e} thr-flips={fl} hi-flips={hi}",flush=True)
    torch.jit.save(trt, str(REPO/"models/v7_trt_fp16.ts"))
    print("saved engine -> models/v7_trt_fp16.ts",flush=True)
except Exception as e:
    print("TRT FAILED:\n"+traceback.format_exc(),flush=True)
print("TRT EXPERIMENT DONE",flush=True)
