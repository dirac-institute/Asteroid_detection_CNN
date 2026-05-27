"""Rejecter build Stage C: train the upgraded rejecter (RF_FEATURES_V2 + Veres + clumping/isolation)
with PANEL-DISJOINT out-of-fold CV (leak-free) and compare to the shipped RF. Reports, at matched
TP-recall, how much more FP the new rejecter removes, and confirms sub-5-sigma TP are preserved
(stack_detection=False = ADCNN's gain). test_real untouched."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.features import RF_FEATURES_V2
from ADCNN.inference.rf_postproc import load_rf
OUT=REPO/"experiments/heliolinc/rejecter_data"
DATA=REPO/"DATA_DIFFIM/test_5sigma"

cand=pd.read_parquet(OUT/"candB.parquet")
# training pool: injected TP + clean FP (not on a DIA/real mask) -- same rule as rf_train.build_pool
fp=(cand.label==0)&(cand["frac_real_label_overlap"].to_numpy()<0.5)
pool=cand[(cand.label==1)|fp].copy().reset_index(drop=True)
y=pool.label.to_numpy(); groups=pool.panel_id.to_numpy()
print(f"pool: {len(pool)} ({int(y.sum())} TP / {int((y==0).sum())} FP) over {pool.panel_id.nunique()} panels")

# sub-5sigma tag for TP: match to nearest injection in test.csv -> stack_detection
cat=pd.read_csv(DATA/"test.csv")
pool["sub5"]=False
for pid,g in pool[pool.label==1].groupby("panel_id"):
    inj=cat[cat.image_id==pid]
    if not len(inj): continue
    for idx,r in g.iterrows():
        d=np.hypot(inj.x-r.x,inj.y-r.y)
        if d.min()<10: pool.loc[idx,"sub5"]=not bool(inj.loc[d.idxmin(),"stack_detection"])
nsub5=int(pool[(pool.label==1)].sub5.sum()); ntp=int(y.sum())
print(f"TP: {ntp} | sub-5sigma TP (stack missed, ADCNN's gain): {nsub5}")

MASK_FEATS=["m_SAT","m_CR","m_EDGE","m_STREAK","m_SPIKE","m_SENSOR_EDGE","m_CROSSTALK","m_ITL_DIP",
            "m_BAD","m_SUSPECT","m_DETECTED_NEGATIVE"]
NEW_FEATS=list(RF_FEATURES_V2)+["veres_len","veres_theta","veres_rchi","veres_dpos","veres_dlen",
    "veres_ok","panel_nlong","panel_ncand","nn_dist","is_long_clumped"]+[m for m in MASK_FEATS]
def Xof(df,feats): return df[feats].fillna(0.0).to_numpy(np.float32)

# FAIR test: train BASELINE (72 feats) and AUGMENTED (72+Veres+mask+clumping) on the SAME panel-disjoint
# folds -> isolates the feature contribution (no training-data confound). Also keep shipped-RF as ref.
oldrf=load_rf(REPO/"models/rf_postproc.pkl")
pool["ship_score"]=oldrf.predict_proba(Xof(pool,list(RF_FEATURES_V2)))[:,1]
pool["old_score"]=np.nan   # baseline 72-feat RF, same folds/hyperparams as augmented
pool["new_score"]=np.nan   # augmented
Xb=Xof(pool,list(RF_FEATURES_V2)); Xn=Xof(pool,NEW_FEATS)
HP=dict(n_estimators=400,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0)
for tr,va in GroupKFold(n_splits=5).split(Xn,y,groups):
    rb=RandomForestClassifier(**HP).fit(Xb[tr],y[tr]); pool.iloc[va,pool.columns.get_loc("old_score")]=rb.predict_proba(Xb[va])[:,1]
    rn=RandomForestClassifier(**HP).fit(Xn[tr],y[tr]); pool.iloc[va,pool.columns.get_loc("new_score")]=rn.predict_proba(Xn[va])[:,1]
print("(OLD = baseline 72-feat RF, SAME folds; NEW = +Veres+mask+clumping, SAME folds -> fair feature test)")

def fp_at_recall(score,target_recall):
    """min FP kept while keeping >= target_recall of TP, and the sub-5sigma recall there."""
    order=np.argsort(-score); ys=y[order]; sub=pool.sub5.to_numpy()[order]
    tpc=np.cumsum(ys==1); fpc=np.cumsum(ys==0)
    need=int(np.ceil(target_recall*ntp)); k=np.searchsorted(tpc,need)
    if k>=len(score): return None
    thr=score[order][k]
    keep=score>=thr
    return dict(thr=round(float(thr),3),tp=int((keep&(y==1)).sum()),fp=int((keep&(y==0)).sum()),
                recall=round((keep&(y==1)).sum()/ntp,3),
                sub5_recall=round((keep&(pool.label==1)&pool.sub5).sum()/max(nsub5,1),3))
print("\n=== OLD RF vs NEW rejecter, at matched TP recall (panel-disjoint, leak-free) ===")
print(f"{'recall':>8} | {'OLD FP kept':>12} {'OLD sub5':>9} | {'NEW FP kept':>12} {'NEW sub5':>9} {'FP reduction':>13}")
for rec in [0.90,0.95,0.98,0.99]:
    o=fp_at_recall(pool.old_score.to_numpy(),rec); n=fp_at_recall(pool.new_score.to_numpy(),rec)
    if o and n:
        red=f"{100*(1-n['fp']/max(o['fp'],1)):.0f}%"
        print(f"{rec:>8.2f} | {o['fp']:>12} {o['sub5_recall']:>9.2f} | {n['fp']:>12} {n['sub5_recall']:>9.2f} {red:>13}")
# at the OLD RF's shipped operating point (thr 0.5): its recall+FP, then NEW matched to that recall
keepo=pool.old_score>=0.5
oprec=(keepo&(pool.label==1)).sum()/ntp
print(f"\nOLD RF @0.5 ship point: recall {oprec:.3f}, FP kept {int((keepo&(y==0)).sum())}, sub5 recall {(keepo&(pool.label==1)&pool.sub5).sum()/max(nsub5,1):.3f}")
nm=fp_at_recall(pool.new_score.to_numpy(),oprec)
if nm: print(f"NEW rejecter @ same recall {nm['recall']}: FP kept {nm['fp']} (reduction {100*(1-nm['fp']/max(int((keepo&(y==0)).sum()),1)):.0f}%), sub5 recall {nm['sub5_recall']}")
import joblib
# fit final new rejecter on ALL pool (for shipping) + save feature list
rf=RandomForestClassifier(n_estimators=500,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0).fit(Xn,y)
joblib.dump({"rf":rf,"features":NEW_FEATS},OUT/"rejecter_v2.pkl")
pool[["panel_id","candidate_id","label","sub5","old_score","new_score"]].to_csv(OUT/"rejecter_scores.csv",index=False)
print(f"\n-> {OUT}/rejecter_v2.pkl (features={len(NEW_FEATS)}), rejecter_scores.csv")
