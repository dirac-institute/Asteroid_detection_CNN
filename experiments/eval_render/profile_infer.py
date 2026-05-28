"""Granular per-stage timing of the sliding-window inference, to find what actually
costs wall time: CPU preprocessing (build_3channel/local-std), H2D transfer, GPU forward,
D2H, or Hann accumulation. Run on a few panels."""
import sys, time
from pathlib import Path
import numpy as np, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.data.preprocessing import build_3channel, diffim_mad_sigma
from ADCNN.inference.predict import hann2d, _tile_starts
dev=torch.device("cuda"); torch.backends.cudnn.benchmark=True
m=torch.jit.load(str(REPO/"models/segmentation_model.pt"),map_location=dev).eval()
H5=REPO/"DATA_DIFFIM/test_5sigma/test.h5"
tile,stride,BATCH,clip=128,64,64,5.0
T={k:0.0 for k in ["prep","h2d","fwd","d2h","acc"]}; ntiles=0
with h5py.File(H5,"r") as f, torch.no_grad():
    for pid in range(3):
        img=f["images"][pid][:].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
        H,W=img.shape; s=min(1024,H,W); sig=diffim_mad_sigma(img[(H-s)//2:(H-s)//2+s,(W-s)//2:(W-s)//2+s])
        hann=hann2d(tile); pa=np.zeros((H,W),np.float32); wa=np.zeros((H,W),np.float32)
        ys=_tile_starts(H,tile,stride); xs=_tile_starts(W,tile,stride)
        bx=[]; bl=[]
        def flush():
            global ntiles
            if not bx: return
            t=time.time(); xb=torch.from_numpy(np.stack(bx)).to(dev,non_blocking=True); torch.cuda.synchronize(); T["h2d"]+=time.time()-t
            t=time.time()
            with torch.amp.autocast("cuda"): seg,_,_,_,_=m(xb)
            torch.cuda.synchronize(); T["fwd"]+=time.time()-t
            t=time.time(); probs=torch.sigmoid(seg).cpu().numpy().astype(np.float32); T["d2h"]+=time.time()-t
            t=time.time()
            for (y0,x0),p in zip(bl,probs[:,0]): pa[y0:y0+tile,x0:x0+tile]+=p*hann; wa[y0:y0+tile,x0:x0+tile]+=hann
            T["acc"]+=time.time()-t; ntiles+=len(bx); bx.clear(); bl.clear()
        for y0 in ys:
            for x0 in xs:
                t=time.time(); x3=build_3channel(img[y0:y0+tile,x0:x0+tile],rl[y0:y0+tile,x0:x0+tile],panel_sigma=sig,clip=clip); T["prep"]+=time.time()-t
                bx.append(x3); bl.append((y0,x0))
                if len(bx)>=BATCH: flush()
        flush()
tot=sum(T.values())
print(f"3 panels, {ntiles} tiles, total inference {tot:.1f}s ({tot/3:.2f}s/panel)")
for k,v in T.items(): print(f"  {k:5s}: {v:6.2f}s ({100*v/tot:4.1f}%)  {1000*v/ntiles:.2f} ms/tile")
