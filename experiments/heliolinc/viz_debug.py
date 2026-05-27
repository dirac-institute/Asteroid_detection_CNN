"""Debug: can we even SEE the injected trails? Cut out the BRIGHTEST injections at their test.csv
(x,y), print value stats, and render with trail-tuned scaling + try x/y swap. If a bright injection
shows no trail at (x,y) -> coordinate/image bug; if it shows -> the noise montage was a scaling issue."""
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); DATA=REPO/"DATA_DIFFIM/test_5sigma"
OUT=REPO/"experiments/heliolinc/rejecter_data"
cat=pd.read_csv(DATA/"test.csv")
bright=cat.sort_values("mag").head(12)   # brightest injections (lowest mag)
print("brightest injections: mag",bright.mag.round(1).tolist())
print("trail_length px:",bright.trail_length.round(0).tolist())
C=30
fig,axes=plt.subplots(3,12,figsize=(24,7))
with h5py.File(DATA/"test.h5","r") as f:
    for j,(_,r) in enumerate(bright.iterrows()):
        im=f["images"][int(r.image_id)].astype(np.float32); H,W=im.shape
        x,y=int(round(r.x)),int(round(r.y))
        for ri,(xx,yy,tag) in enumerate([(x,y,"x,y"),(y,x,"y,x(swap)")]):
            x0,x1,y0,y1=max(0,xx-C),min(W,xx+C),max(0,yy-C),min(H,yy+C)
            c=im[y0:y1,x0:x1]
            ax=axes[ri,j]; ax.axis("off")
            if c.size:
                v=np.percentile(np.abs(c),98) or 1
                ax.imshow(c,vmin=-v,vmax=v,cmap="gray")
                if ri==0 and j==0: print(f"cutout stats: shape{c.shape} min{c.min():.1f} max{c.max():.1f} std{c.std():.2f}")
            ax.set_title(f"{tag} m{r.mag:.1f}",fontsize=7)
        # full-panel value range + where the max is, to sanity-check
        if j==0:
            print(f"panel {int(r.image_id)} full image: min{im.min():.1f} max{im.max():.1f} median{np.median(im):.2f} std{im.std():.2f}")
            ax=axes[2,0]; ax.imshow(im[::8,::8],vmin=-np.percentile(np.abs(im),99),vmax=np.percentile(np.abs(im),99),cmap="gray"); ax.set_title("full panel /8",fontsize=7); ax.axis("off")
for j in range(1,12): axes[2,j].axis("off")
plt.tight_layout(); plt.savefig(OUT/"debug_inj.png",dpi=80,bbox_inches="tight")
print(f"-> {OUT}/debug_inj.png  (row0: at (x,y); row1: swapped (y,x))")
