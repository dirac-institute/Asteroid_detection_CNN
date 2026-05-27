"""Post-v7 cutout-CNN rejecter, LEAK-FREE. val2_snr gen crashed, so threshold on a PANEL-DISJOINT
hold-out of val_filter_snr (the split.json val panels, SNR 2-8, part of train set; v7 only early-
stopped on them). TRAIN = 40 val_filter panels, THRESH = held-out 10, TEST = test_5sigma (untouched).
Compares CNN (cutout [diffim/sigma, v7_prob, v7_agg] 48x48) vs RF (hand feats) at 95% recall, sub-5sigma."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/rejecter_data")
DD=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA_DIFFIM")
dev=torch.device("cuda")
def sub5_tag(npz,csv):
    cat=pd.read_csv(csv); y=npz["y"]; xy=npz["xy"]; pan=npz["panel"]; s=np.zeros(len(y),bool)
    for i in range(len(y)):
        if y[i]!=1: continue
        inj=cat[cat.image_id==pan[i]]
        if len(inj):
            d=np.hypot(inj.x.values-xy[i,0],inj.y.values-xy[i,1])
            if d.min()<10: s[i]=not bool(inj.iloc[d.argmin()].stack_detection)
    return s
allf=np.load(R/"cut_val_filter_snr.npz",allow_pickle=True); te=np.load(R/"cut_train.npz",allow_pickle=True)
pans=np.unique(allf["panel"]); rng=np.random.default_rng(0); rng.shuffle(pans)
thr_pan=set(pans[:max(1,len(pans)//5)].tolist())
m_thr=np.isin(allf["panel"],list(thr_pan)); m_tr=~m_thr
s5all=sub5_tag(allf,DD/"val_filter_snr/train.csv")
Xtr,ytr=np.clip(allf["X"][m_tr],-20,20),allf["y"][m_tr].astype(np.float32)
Xva,yva=np.clip(allf["X"][m_thr],-20,20),allf["y"][m_thr].astype(np.float32); s5va=s5all[m_thr]
Xte,yte=np.clip(te["X"],-20,20),te["y"].astype(np.float32); s5te=sub5_tag(te,DD/"test_5sigma/test.csv")
ftr,fva=allf["feat"][m_tr],allf["feat"][m_thr]
print(f"TRAIN {Xtr.shape} TP {int(ytr.sum())} | THRESH(holdout) {Xva.shape} TP {int(yva.sum())} sub5 {int(s5va.sum())} | TEST(test5) {Xte.shape} TP {int(yte.sum())} sub5 {int(s5te.sum())}",flush=True)

class Net(nn.Module):
    def __init__(s,c=3):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(c,32),blk(32,64),blk(64,128),nn.AdaptiveAvgPool2d(1)); s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,1))
    def forward(s,x): return s.h(s.f(x)).squeeze(1)
def score(net,X):
    net.eval(); Xt=torch.tensor(X).to(dev)
    with torch.no_grad(): return torch.sigmoid(torch.cat([net(Xt[k:k+512]) for k in range(0,len(Xt),512)])).cpu().numpy()
def final(s,thr_s,y,s5,n5):  # threshold at 95% recall on thr_s's own TP, apply to (s,y)
    thr=np.quantile(thr_s,0.05); keep=s>=thr
    return dict(recall=float((keep&(y==1)).sum()/max((y==1).sum(),1)),fp=int((keep&(y==0)).sum()),
                nfp=int((y==0).sum()),sub5=float((keep&(y==1)&s5).sum()/max(n5,1)))
net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
posw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],device=dev); lossf=nn.BCEWithLogitsLoss(pos_weight=posw)
Xtr_t=torch.tensor(Xtr); ytr_t=torch.tensor(ytr); N=len(Xtr); bs=256; best=(-1,None)
for ep in range(20):
    net.train(); perm=torch.randperm(N)
    for k in range(0,N,bs):
        idx=perm[k:k+bs]; opt.zero_grad(); loss=lossf(net(Xtr_t[idx].to(dev)),ytr_t[idx].to(dev)); loss.backward(); opt.step()
    sv=score(net,Xva); auc=roc_auc_score(yva,sv)
    if auc>best[0]: best=(auc,{k:v.cpu().clone() for k,v in net.state_dict().items()})
    print(f"ep{ep}: holdout AUC {auc:.3f}",flush=True)
net.load_state_dict(best[1])
sv=score(net,Xva); st=score(net,Xte)
cnn=final(st,sv[yva==1],yte,s5te,int(s5te.sum()))
rf=RandomForestClassifier(n_estimators=500,max_depth=16,class_weight="balanced",n_jobs=-1,random_state=0).fit(ftr,ytr)
rv=rf.predict_proba(fva)[:,1]; rt=rf.predict_proba(te["feat"])[:,1]
rff=final(rt,rv[yva==1],yte,s5te,int(s5te.sum()))
print("\n=== FINAL on TEST_5sigma (threshold from val_filter hold-out, leak-free) @ ~95% recall ===")
for nm,r,auc in [("CNN (cutout)",cnn,roc_auc_score(yte,st)),("RF (hand feats)",rff,roc_auc_score(yte,rt))]:
    print(f"  {nm:<18} recall {r['recall']:.3f} | FP kept {r['fp']}/{r['nfp']} (removed {100*(1-r['fp']/r['nfp']):.0f}%) | sub5 {r['sub5']:.2f} | test AUC {auc:.3f}")
torch.save(best[1],R/"cnn_rejecter.pt")
