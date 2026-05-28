"""Develop the post-segmentation model FP filter on the dedicated train2 set (SNR 2-8, ~440 panels, disjoint from
segmentation model-train + all test/val sets). PANEL-DISJOINT internal split (80% train / 20% test). Trains CNN
(cutout) + RF (hand feats) on the train split, evaluates FP-removed at 95% recall (sub-5sigma
preserved) on the held-out test split. Answers: with enough data, how well can we filter FP?"""
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/rejecter_data")
DD=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA_DIFFIM")
dev=torch.device("cuda")
d=np.load(R/"cut_train2.npz",allow_pickle=True)
cat=pd.read_csv(DD/"train2/train.csv")
y=d["y"].astype(np.float32); X=np.clip(d["X"],-20,20); pan=d["panel"]; xy=d["xy"]; feat=d["feat"]
s5=np.zeros(len(y),bool)
for i in range(len(y)):
    if y[i]!=1: continue
    inj=cat[cat.image_id==pan[i]]
    if len(inj):
        dd=np.hypot(inj.x.values-xy[i,0],inj.y.values-xy[i,1])
        if dd.min()<10: s5[i]=not bool(inj.iloc[dd.argmin()].stack_detection)
pans=np.unique(pan); rng=np.random.default_rng(0); rng.shuffle(pans)
te_pan=set(pans[:max(1,len(pans)//5)].tolist()); m_te=np.isin(pan,list(te_pan)); m_tr=~m_te
print(f"train2: {len(y)} cand, {int(y.sum())} TP, {int(s5.sum())} sub5 | TRAIN panels {len(pans)-len(te_pan)} TEST panels {len(te_pan)}",flush=True)
n5te=int((s5&m_te&(y==1)).sum())
def at95(s_te,s_tr_tp):
    thr=np.quantile(s_tr_tp,0.05); keep=(s_te>=thr)
    yt=y[m_te]; s5t=s5[m_te]
    return dict(recall=float((keep&(yt==1)).sum()/max((yt==1).sum(),1)),fp=int((keep&(yt==0)).sum()),
                nfp=int((yt==0).sum()),sub5=float((keep&(yt==1)&s5t).sum()/max(n5te,1)))
# RF
rf=RandomForestClassifier(n_estimators=500,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0).fit(feat[m_tr],y[m_tr])
rtr=rf.predict_proba(feat[m_tr])[:,1]; rte=rf.predict_proba(feat[m_te])[:,1]
rf_r=at95(rte,rtr[y[m_tr]==1])
# CNN
class Net(nn.Module):
    def __init__(s,c=3):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(c,32),blk(32,64),blk(64,128),nn.AdaptiveAvgPool2d(1)); s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,1))
    def forward(s,x): return s.h(s.f(x)).squeeze(1)
def score(net,Xa):
    net.eval(); Xt=torch.tensor(Xa).to(dev)
    with torch.no_grad(): return torch.sigmoid(torch.cat([net(Xt[k:k+512]) for k in range(0,len(Xt),512)])).cpu().numpy()
net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
Xtr,ytr=X[m_tr],y[m_tr]; posw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],device=dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=posw); Xtr_t=torch.tensor(Xtr); ytr_t=torch.tensor(ytr); N=len(Xtr); bs=256
# small internal val (panels from train split) for early stop
tp2=list(te_pan); vp=set(np.unique(pan[m_tr])[:max(1,(len(pans)-len(te_pan))//6)].tolist())
mv=np.isin(pan,list(vp))&m_tr; mt2=m_tr&~mv
best=(-1,None)
for ep in range(25):
    net.train(); idx_all=np.where(mt2)[0]; perm=np.random.permutation(len(idx_all))
    Xt2=torch.tensor(X[idx_all]); yt2=torch.tensor(y[idx_all])
    for k in range(0,len(idx_all),bs):
        b=perm[k:k+bs]; opt.zero_grad(); loss=lossf(net(Xt2[b].to(dev)),yt2[b].to(dev)); loss.backward(); opt.step()
    sv=score(net,X[mv]); auc=roc_auc_score(y[mv],sv)
    if auc>best[0]: best=(auc,{k:v.cpu().clone() for k,v in net.state_dict().items()})
    print(f"ep{ep}: internal-val AUC {auc:.3f}",flush=True)
net.load_state_dict(best[1])
# threshold from train-split TP, eval on held-out test
str_=score(net,X[m_tr]); ste=score(net,X[m_te]); cnn_r=at95(ste,str_[y[m_tr]==1])
print("\n=== train2 held-out TEST split @ ~95% recall (FP filter, lots of data) ===")
for nm,r,auc,s in [("CNN (cutout)",cnn_r,roc_auc_score(y[m_te],ste),ste),("RF (hand feats)",rf_r,roc_auc_score(y[m_te],rte),rte)]:
    print(f"  {nm:<18} recall {r['recall']:.3f} | FP removed {100*(1-r['fp']/r['nfp']):.0f}% ({r['fp']}/{r['nfp']} kept) | sub5 {r['sub5']:.2f} | AUC {auc:.3f}")
torch.save(best[1],R/"cnn_train2.pt")
