"""#1a investigation: do the diffim MASK planes (STREAK/SPIKE/CR/SAT/...) flag ADCNN's false-positive
streaks while sparing real asteroids? For a sample of NEO_large panels, read the MASK HDU and, for
each ADCNN detection, OR the mask bits over a small footprint around (x,y) AND along the trail
(beta,len_db). Split detections into REAL (match a catalogued object within 2") vs FP and report the
mask-overlap fraction per bit. A good free filter = high FP-on-mask, ~0 real-on-mask."""
import pandas as pd, numpy as np, random
from astropy.io import fits

HL="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc"
RUN=f"{HL}/NEO_large"
BITS={"SAT":1,"CR":3,"EDGE":4,"STREAK":10,"SPIKE":19,"SENSOR_EDGE":18,"CROSSTALK":12,
      "ITL_DIP":14,"BAD":0,"SUSPECT":7,"DETECTED":5,"DETECTED_NEGATIVE":6}
NPANELS=150

ad=pd.read_csv(f"{RUN}/adcnn_dets.csv"); ad["visit"]=ad.visit.astype(int); ad["detector"]=ad.detector.astype(int)
man=pd.read_csv(f"{RUN}/manifest.csv"); man["visit"]=man.visit.astype(int); man["detector"]=man.detector.astype(int)
kn=pd.read_csv(f"{RUN}/known.csv").dropna(subset=["ra","dec","mjd"])
# label each detection real/FP by match to known (2", same mjd)
import bisect; from collections import defaultdict
kby=defaultdict(list)
for m,r,d in zip(kn.mjd.values,kn.ra.values,kn.dec.values): kby[round(m,5)].append((r,d))
kms=np.array(sorted(kby.keys()))
def isreal(m,r,d):
    i=bisect.bisect_left(kms,m)
    for j in (i-1,i):
        if 0<=j<len(kms) and abs(kms[j]-m)<0.005:
            for kr,kd in kby[kms[j]]:
                if abs(kd-d)<2/3600 and abs((kr-r)*np.cos(np.radians(d)))<2/3600: return True
    return False

pmap=man.set_index(["visit","detector"]).fits_path.to_dict()
random.seed(1)
panels=random.sample(list(ad.groupby(["visit","detector"]).groups.keys()), NPANELS)
rows=[]
for (v,det) in panels:
    fp=pmap.get((v,det))
    if not fp: continue
    try: msk=fits.open(fp)[2].data
    except Exception: continue
    H,W=msk.shape
    g=ad[(ad.visit==v)&(ad.detector==det)]
    for _,r in g.iterrows():
        x,y=int(round(r.x)),int(round(r.y))
        # footprint: box +-4px AND sample along trail (beta deg, len_db px)
        x0,x1=max(0,x-4),min(W,x+5); y0,y1=max(0,y-4),min(H,y+5)
        acc=int(np.bitwise_or.reduce(msk[y0:y1,x0:x1].ravel())) if (x1>x0 and y1>y0) else 0
        L=float(r.len_db) if r.len_db==r.len_db else 0
        if L>=4:
            th=np.radians(float(r.beta) if r.beta==r.beta else 0)
            for t in np.linspace(-L/2,L/2,int(L)):
                xx,yy=int(round(x+t*np.cos(th))),int(round(y+t*np.sin(th)))
                if 0<=xx<W and 0<=yy<H: acc|=int(msk[yy,xx])
        rows.append((isreal(r.mjd,r.ra,r.dec), acc, L, f"{v}_{det}"))
df=pd.DataFrame(rows,columns=["real","mask","len_db","panel"])
# #1c clumping signal: long (>=15px) detections in panels carrying many such streaks = instrumental
longp=df[df.len_db>=15].groupby("panel").size()
clumped_panels=set(longp[longp>=3].index)
df["clump"]=(df.len_db>=15)&(df.panel.isin(clumped_panels))
print(f"sampled {len(df)} detections over {NPANELS} panels | real {df.real.sum()} | FP {(~df.real).sum()}")
print(f"\n{'bit':>16} {'%FP on bit':>11} {'%REAL on bit':>13}")
for name,b in BITS.items():
    on=(df["mask"]&(1<<b))>0
    fpf=100*on[~df.real].mean() if (~df.real).any() else 0
    rlf=100*on[df.real].mean() if df.real.any() else 0
    print(f"{name:>16} {fpf:>11.1f} {rlf:>13.1f}")
def report(name, cut):
    fp=100*cut[~df.real].mean(); tp=100*cut[df.real].mean()
    print(f"  {name:<42} removes {fp:5.1f}% FP | costs {tp:4.1f}% REAL")
print("\n=== candidate TP-safe FP filters (on labeled sample) ===")
narrow=(df["mask"]&((1<<10)|(1<<19)|(1<<3)|(1<<1)))>0
instr_bits=sum(1<<b for b in [1,3,4,10,19,18,12,14,0,7,6])  # SAT CR EDGE STREAK SPIKE SENSOR_EDGE CROSSTALK ITL_DIP BAD SUSPECT DET_NEG
instr=(df["mask"]&instr_bits)>0
report("STREAK|SPIKE|CR|SAT (narrow)", narrow)
report("all instrumental-artifact bits", instr)
report("panel-clumping (long streaks in clumped panels)", df.clump)
report("instrumental-mask OR clumping (combined)", instr|df.clump)
report("combined OR not-DETECTED", instr|df.clump|((df["mask"]&(1<<5))==0))
