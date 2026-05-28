"""Lock the rejecter at the 95%-TP-recall operating point and evaluate leak-free on test_5sigma
(panel-disjoint out-of-fold). Reports TP recall, sub-5sigma recall, FP removed, FP/panel vs the
72-feature baseline; ships rejecter_v2_final.pkl with the threshold baked in. test_real untouched."""
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from ADCNN.inference.features import FEATURES_V2
OUT=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/rejecter_data")
DATA=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA_DIFFIM/test_5sigma")
c=pd.read_parquet(OUT/"candB.parquet"); cat=pd.read_csv(DATA/"test.csv")
c["sub5"]=False
for pid,g in c[c.label==1].groupby("panel_id"):
    inj=cat[cat.image_id==pid]
    if not len(inj): continue
    for idx,r in g.iterrows():
        d=np.hypot(inj.x-r.x,inj.y-r.y)
        if d.min()<10: c.loc[idx,"sub5"]=not bool(inj.loc[d.idxmin(),"stack_detection"])
y=c.label.to_numpy(); ntp=int((y==1).sum()); nfp=int((y==0).sum()); nsub5=int(c[c.label==1].sub5.sum()); npan=c.panel_id.nunique()
MASK=[m for m in ["m_SAT","m_CR","m_EDGE","m_STREAK","m_SPIKE","m_SENSOR_EDGE","m_CROSSTALK","m_ITL_DIP","m_BAD","m_SUSPECT","m_DETECTED_NEGATIVE"] if m in c]
FE=list(FEATURES_V2)+["veres_len","veres_theta","veres_rchi","veres_ok","nn_dist","is_long_clumped"]+MASK
HP=dict(n_estimators=500,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0)
g=c.panel_id.to_numpy()
def oof(feats):
    X=c[feats].fillna(0).to_numpy(np.float32); s=np.full(len(c),np.nan)
    for tr,va in GroupKFold(5).split(X,y,g):
        s[va]=RandomForestClassifier(**HP).fit(X[tr],y[tr]).predict_proba(X[va])[:,1]
    return s
def evalpt(name,score,rec=0.95):
    score=np.nan_to_num(score,nan=-1e9); thr=float(np.quantile(score[y==1],1-rec)); keep=score>=thr
    tpk=int((keep&(y==1)).sum()); fpk=int((keep&(y==0)).sum()); s5=(keep&(c.label==1)&c.sub5).sum()/max(nsub5,1)
    print(f"  {name:<24} thr={thr:.3f} | TP recall {tpk/ntp:.3f} | sub-5σ recall {s5:.3f} | FP kept {fpk}/{nfp} ({100*fpk/nfp:.0f}%, {fpk/npan:.0f}/panel) | FP removed {100*(1-fpk/nfp):.0f}%"); return thr
print(f"=== test_5sigma leak-free eval (out-of-fold) @ 95% TP recall | TP {ntp}, FP {nfp}, sub-5σ {nsub5}, {npan} panels ===")
evalpt("baseline 72-feat",oof(list(FEATURES_V2)))
s_new=oof(FE); thr95=evalpt("REJECTER (Veres+mask+ctx)",s_new)
# ship: final model on ALL panels + threshold
rf=RandomForestClassifier(**HP).fit(c[FE].fillna(0).to_numpy(np.float32),y)
joblib.dump({"rf":rf,"features":FE,"thr":thr95,"recall_target":0.95},OUT/"rejecter_v2_final.pkl")
print(f"\nshipped -> {OUT}/rejecter_v2_final.pkl (thr={thr95:.3f} @95% recall, {len(FE)} features)")
