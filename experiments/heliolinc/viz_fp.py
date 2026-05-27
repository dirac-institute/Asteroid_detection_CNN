"""Look at the pixels: montage of TP (injected) vs hard-FP (high classifier score, label=0) candidate
cutouts from the test.h5 images. If FP look like trails -> genuine confusion; if noise/dipoles ->
candidate-extraction/labeling mistake."""
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
DATA=REPO/"DATA_DIFFIM/test_5sigma"; OUT=REPO/"experiments/heliolinc/rejecter_data"
cand=pd.read_parquet(OUT/"candB.parquet")
sc=pd.read_csv(OUT/"rejecter_scores.csv")
cand=cand.merge(sc[["panel_id","candidate_id","new_score","old_score","sub5"]],on=["panel_id","candidate_id"],how="left")
C=32  # half-cutout
from astropy.visualization import ZScaleInterval
_zs=ZScaleInterval()
def cut(img,x,y):
    H,W=img.shape; x,y=int(round(x)),int(round(y))
    x0,x1,y0,y1=max(0,x-C),min(W,x+C),max(0,y-C),min(H,y+C)
    c=img[y0:y1,x0:x1]; out=np.zeros((2*C,2*C),np.float32); out[:c.shape[0],:c.shape[1]]=c; return out
def grab(df,n):
    df=df.head(n); imgs=[]
    with h5py.File(DATA/"test.h5","r") as f:
        for pid,g in df.groupby("panel_id"):
            im=f["images"][int(pid)].astype(np.float32)
            for _,r in g.iterrows(): imgs.append((cut(im,r.x,r.y),r))
    return imgs
tp=grab(cand[cand.label==1].sort_values("mf_snr",ascending=False),16)         # strongest TP trails
hardfp=grab(cand[cand.label==0].sort_values("mf_snr",ascending=False),16)     # FP with HIGHEST line-coherence (the hard ones)
fig,axes=plt.subplots(4,8,figsize=(18,10))
allimgs=[("TP",x) for x in tp]+[("FP",x) for x in hardfp]
for k in range(32):
    ax=axes[k//8,k%8]; ax.axis("off")
    if k<len(allimgs):
        tag,(im,row)=allimgs[k]
        try: lo,hi=_zs.get_limits(im)
        except Exception: lo,hi=im.min(),im.max()
        ax.imshow(im,vmin=lo,vmax=hi,cmap="gray")
        ax.set_title(f"{tag} mfsnr{row.mf_snr:.1f} L{row._len:.0f}",fontsize=8,
                     color=("green" if tag=="TP" else "red"))
fig.suptitle("rows 1-2: TP (injected)   |   rows 3-4: HARD FP (label=0, highest rejecter score)",fontsize=12)
plt.tight_layout(); plt.savefig(OUT/"fp_vs_tp.png",dpi=80,bbox_inches="tight")
print(f"-> {OUT}/fp_vs_tp.png")
print(f"TP count {int((cand.label==1).sum())} | FP {int((cand.label==0).sum())}")
print(f"hard FP (label0, score>0.5): {int(((cand.label==0)&(cand.new_score>0.5)).sum())}")
print(f"FP len_db: median {cand[cand.label==0]._len.median():.1f} p90 {cand[cand.label==0]._len.quantile(.9):.0f} | TP len_db median {cand[cand.label==1]._len.median():.1f}")
