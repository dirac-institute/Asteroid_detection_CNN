"""Verify parallel build_3channel is bit-identical to serial, and time the new inference."""
import sys, time
from pathlib import Path
import numpy as np, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
import ADCNN.inference.predict as P
P._TILE_BATCH=64
dev=torch.device("cuda"); torch.backends.cudnn.benchmark=True
m=torch.jit.load(str(REPO/"models/segmentation_model.pt"),map_location=dev).eval()
with h5py.File(REPO/"DATA_DIFFIM/test_5sigma/test.h5","r") as f:
    img=f["images"][0][:].astype(np.float32); rl=f["real_labels"][0][:].astype(np.uint16)
# bit-identical check: serial vs parallel prep
P._PREP_WORKERS=1
o1=P.predict_panel_overlap_3ch_full(m,img,rl,device=dev)
P._PREP_WORKERS=8
o2=P.predict_panel_overlap_3ch_full(m,img,rl,device=dev)
ident=all(np.array_equal(a,b) for a,b in zip(o1,o2))
print(f"parallel-prep BIT-IDENTICAL to serial: {ident}")
# timing (parallel prep, batch64) over a few panels
P._PREP_WORKERS=8
with h5py.File(REPO/"DATA_DIFFIM/test_5sigma/test.h5","r") as f:
    t0=time.time()
    for pid in range(4):
        img=f["images"][pid][:].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
        P.predict_panel_overlap_3ch_full(m,img,rl,device=dev)
    dt=time.time()-t0
print(f"new inference: {dt:.1f}s / 4 panels = {dt/4:.2f}s/panel (was 7.33s/panel single-thread prep)")
