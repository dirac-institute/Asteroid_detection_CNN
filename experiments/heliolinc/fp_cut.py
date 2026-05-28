"""Lower the FP noise: quantify a TP-safe line-coherence cut on the val set. The bulk FP are noise
blobs (low coherence); see how much FP a cut on the best coherence features removes at fixed TP recall,
preserving sub-5sigma TP (ADCNN's gain). Compares single-feature cuts to the trained rejecter."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from ADCNN.inference.features import FEATURES_V2
OUT=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/rejecter_data")
DATA=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA_DIFFIM/test_5sigma")
c=pd.read_parquet(OUT/"candB.parquet")
cat=pd.read_csv(DATA/"test.csv")
# sub-5sigma tag (compute inline, robust)
c["sub5"]=False
for pid,g in c[c.label==1].groupby("panel_id"):
    inj=cat[cat.image_id==pid]
    if not len(inj): continue
    for idx,r in g.iterrows():
        d=np.hypot(inj.x-r.x,inj.y-r.y)
        if d.min()<10: c.loc[idx,"sub5"]=not bool(inj.loc[d.idxmin(),"stack_detection"])
y=c.label.to_numpy(); ntp=int((y==1).sum()); nfp=int((y==0).sum()); nsub5=int(c[c.label==1].sub5.sum())
print(f"TP {ntp} | FP {nfp} | sub-5sigma TP {nsub5}")

def report(name,score):
    score=np.nan_to_num(score,nan=-1e9)
    out=[]
    for rec in [0.99,0.98,0.95,0.90]:
        thr=np.quantile(score[y==1],1-rec); keep=score>=thr
        fpk=int((keep&(y==0)).sum()); s5=(keep&(c.label==1)&c.sub5).sum()/max(nsub5,1)
        out.append(f"r{rec:.2f}:FP{100*fpk/nfp:.0f}%/s5{s5:.2f}")
    print(f"  {name:<26} "+"  ".join(out))

print("\n=== FP kept (% of all FP) at fixed TP recall / sub-5sigma recall ===")
for f in ["integrated_logit","top5_mean_p","masnr_30","mf_snr"]:
    report(f"cut: {f}", c[f].fillna(0).to_numpy().astype(float))
# trained rejecter (out-of-fold), full feature set
MASK=["m_SAT","m_CR","m_EDGE","m_STREAK","m_SPIKE","m_SENSOR_EDGE","m_CROSSTALK","m_ITL_DIP","m_BAD","m_SUSPECT","m_DETECTED_NEGATIVE"]
FE=list(FEATURES_V2)+["veres_len","veres_theta","veres_rchi","veres_ok","nn_dist","is_long_clumped"]+[m for m in MASK if m in c]
X=c[FE].fillna(0).to_numpy(np.float32); g=c.panel_id.to_numpy()
oof=np.full(len(c),np.nan)
for tr,va in GroupKFold(5).split(X,y,g):
    rf=RandomForestClassifier(n_estimators=400,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0).fit(X[tr],y[tr])
    oof[va]=rf.predict_proba(X[va])[:,1]
report("REJECTER (all features)",oof)
print("\n(FP% = fraction of all 112k FP kept; lower=better. s5 = sub-5sigma TP recall, want high.)")
