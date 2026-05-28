"""Overnight FP-filter iteration harness. Loads cached train2 cutouts, fixed panel-disjoint 80/20
split, runs a set of methods, and logs FP-removed @95% recall (sub-5sigma preserved) for each to
fp_iter_results.csv. Goal: KEEP TP (>=95% recall + high sub-5sigma) and remove a good amount of FP.
Run: python fp_iterate.py --methods rf,rf_coh,cnn,ensemble,cnn_focal"""
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn, time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
R=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/rejecter_data")
DD=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA_DIFFIM")
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ap=argparse.ArgumentParser(); ap.add_argument("--methods",default="rf,rf_coh,cnn,ensemble"); ap.add_argument("--recall",type=float,default=0.95)
ap.add_argument("--epochs",type=int,default=25); a=ap.parse_args(); REC=a.recall

def load_parts(dirp):
    fs=sorted(Path(dirp).glob("part_*.npz"))
    assert fs, f"no part_*.npz in {dirp}"
    P={k:[] for k in ("X","y","panel","cid","feat","xy")}
    for fn in fs:
        z=np.load(fn,allow_pickle=True)
        for k in P: P[k].append(z[k])
    return {k:np.concatenate(v) for k,v in P.items()}
d=load_parts(R/"cut_train2")
X=np.clip(d["X"].astype(np.float32),-20,20); y=d["y"].astype(np.float32); pan=d["panel"]; xy=d["xy"]; feat=d["feat"].astype(np.float32)
cat=pd.read_csv(DD/"train2/train.csv")
s5=np.zeros(len(y),bool)
for i in range(len(y)):
    if y[i]!=1: continue
    inj=cat[cat.image_id==pan[i]]
    if len(inj):
        dd=np.hypot(inj.x.values-xy[i,0],inj.y.values-xy[i,1])
        if dd.min()<10: s5[i]=not bool(inj.iloc[dd.argmin()].stack_detection)
pans=np.unique(pan); rng=np.random.default_rng(42); rng.shuffle(pans)
te=set(pans[:max(1,len(pans)//5)].tolist()); M_te=np.isin(pan,list(te)); M_tr=~M_te
ntp=int((y[M_te]==1).sum()); n5=int((s5&M_te&(y==1)).sum())
print(f"train2: {len(y)} cand {int(y.sum())} TP {int(s5.sum())} sub5 | TRAIN {M_tr.sum()} TEST {M_te.sum()} (TPte {ntp}, sub5te {n5})",flush=True)

def coherence_feats(Xprob):  # from seg_prob cutout (channel 1): line-fit residual + elongation
    out=np.zeros((len(Xprob),5),np.float32)
    for i,p in enumerate(Xprob):
        ys,xs=np.nonzero(p>0.5); w=p[ys,xs]
        if len(ys)<4: out[i]=[0,0,0,len(ys),0]; continue
        ys=ys.astype(float); xs=xs.astype(float); cx,cy=np.average(xs,weights=w),np.average(ys,weights=w)
        cov=np.cov(np.vstack([xs-cx,ys-cy]),aweights=w); ev=np.linalg.eigvalsh(cov)
        elong=ev[1]/(ev[0]+1e-3); ang=0.5*np.arctan2(2*cov[0,1],cov[0,0]-cov[1,1])
        # residual perpendicular to principal axis
        perp=np.abs(-(xs-cx)*np.sin(ang)+(ys-cy)*np.cos(ang)); res=np.average(perp,weights=w)
        out[i]=[elong, res, ev[1], len(ys), w.mean()]
    return out

def metric(score):  # threshold@REC recall on test TP, report FP removed + sub5 recall
    s=np.nan_to_num(score,nan=-1e9); thr=np.quantile(s[(M_te)&(y==1)],1-REC); keep=(s>=thr)&M_te
    fp=int((keep&(y==0)).sum()); nfp=int((M_te&(y==0)).sum())
    return dict(recall=round((keep&(y==1)).sum()/max(ntp,1),3),fp_removed=round(1-fp/max(nfp,1),3),
                fp_kept=fp,nfp=nfp,sub5=round((keep&(y==1)&s5).sum()/max(n5,1),3),auc=round(roc_auc_score(y[M_te],s[M_te]),3))
RFHP=dict(n_estimators=500,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0)
results=[]; cnn_oof=None
def log(name,score):
    m=metric(score); m["method"]=name; results.append(m)
    print(f"  [{name}] recall {m['recall']} | FP removed {m['fp_removed']:.0%} | sub5 {m['sub5']} | AUC {m['auc']}",flush=True); return score

class Net(nn.Module):
    def __init__(s,c=3,w=56):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(c,w),blk(w,2*w),blk(2*w,4*w),nn.AdaptiveAvgPool2d(1)); s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(4*w,1))
    def forward(s,x): return s.h(s.f(x)).squeeze(1)
def train_cnn(Xa,focal=False,hardneg=False,epochs=a.epochs):
    net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
    yt=y[M_tr]; pw=torch.tensor([(yt==0).sum()/max((yt==1).sum(),1)],device=dev)
    def lossf(o,t):
        if focal:
            p=torch.sigmoid(o); ce=nn.functional.binary_cross_entropy_with_logits(o,t,reduction="none",pos_weight=pw)
            return ((1-torch.where(t==1,p,1-p))**2*ce).mean()
        return nn.functional.binary_cross_entropy_with_logits(o,t,pos_weight=pw)
    Xt=torch.tensor(Xa[M_tr]); yt_t=torch.tensor(yt); N=len(yt); bs=256
    def sc(Xb):
        net.eval(); T=torch.tensor(Xb).to(dev)
        with torch.no_grad(): return torch.sigmoid(torch.cat([net(T[k:k+512]) for k in range(0,len(T),512)])).cpu().numpy()
    w=np.ones(N)
    for ep in range(epochs):
        net.train(); pr=np.random.permutation(N) if not hardneg else np.random.choice(N,N,p=w/w.sum())
        for k in range(0,N,bs):
            b=pr[k:k+bs]; opt.zero_grad(); loss=lossf(net(Xt[b].to(dev)),yt_t[b].to(dev)); loss.backward(); opt.step()
        if hardneg and ep%5==4:
            s_tr=sc(Xa[M_tr]); w=np.where(yt==0,1+5*np.clip(s_tr,0,1),3.0)  # upweight high-score FP + all TP
    full=np.full(len(y),np.nan); full[M_tr]=sc(Xa[M_tr]); full[M_te]=sc(Xa[M_te]); return full

methods=a.methods.split(",")
if "rf" in methods:
    rf=RandomForestClassifier(**RFHP).fit(feat[M_tr],y[M_tr]); log("rf72",rf.predict_proba(feat)[:,1])
if "rf_coh" in methods:
    coh=coherence_feats(X[:,1]); fc=np.hstack([feat,coh])
    rf=RandomForestClassifier(**RFHP).fit(fc[M_tr],y[M_tr]); log("rf72+coh",rf.predict_proba(fc)[:,1])
if any(m in methods for m in ("cnn","ensemble","cnn_focal","cnn_hardneg")):
    cnn_oof=train_cnn(X);
    if "cnn" in methods: log("cnn3ch",cnn_oof)
if "cnn_focal" in methods: log("cnn_focal",train_cnn(X,focal=True))
if "cnn_hardneg" in methods: log("cnn_hardneg",train_cnn(X,hardneg=True))
if "ensemble" in methods:
    coh=coherence_feats(X[:,1]); rf=RandomForestClassifier(**RFHP).fit(np.hstack([feat,coh])[M_tr],y[M_tr])
    rfs=rf.predict_proba(np.hstack([feat,coh]))[:,1]
    meta_X=np.column_stack([cnn_oof,rfs,feat[:, :20]])  # cnn + rf + top feats
    meta=GradientBoostingClassifier(n_estimators=300,max_depth=3,random_state=0).fit(meta_X[M_tr],y[M_tr])
    log("ensemble(cnn+rf+feat)",meta.predict_proba(meta_X)[:,1])
df=pd.DataFrame(results).sort_values("fp_removed",ascending=False)
out=R/"fp_iter_results.csv"; hdr=not out.exists()
df["ts"]=time.strftime("%H:%M"); df.to_csv(out,mode="a",header=hdr,index=False)
print("\n=== ITERATION RESULTS (sorted by FP removed @%.0f%% recall) ==="%(REC*100))
print(df[["method","recall","fp_removed","sub5","auc","fp_kept","nfp"]].to_string(index=False))
best=df[df.sub5>=0.9].sort_values("fp_removed",ascending=False).head(1)
print("\nBEST (sub5>=0.9):", best[["method","fp_removed","sub5"]].to_string(index=False) if len(best) else "none meet sub5>=0.9")
