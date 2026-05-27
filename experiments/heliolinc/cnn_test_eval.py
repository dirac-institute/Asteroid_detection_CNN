"""FINAL test-set eval of the best post-v7 FP filter (focal-loss cutout CNN, w40, 30ep).
TRAIN on train2 (cut_train2 parts, 90% panels), pick 95%-recall threshold on a 10% train2 holdout,
then EVALUATE on the held-out test_5sigma (cut_train.npz). Reports FP-removed/recall/sub-5sigma.
Saves cnn_focal_final.pt + a JSON summary. test_real untouched."""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
R=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/rejecter_data")
DD=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA_DIFFIM")
dev=torch.device("cuda")
def load_parts(dirp):
    P={k:[] for k in("X","y","panel","xy")}
    for fn in sorted(Path(dirp).glob("part_*.npz")):
        z=np.load(fn,allow_pickle=True)
        for k in P: P[k].append(z[k])
    return {k:np.concatenate(v) for k,v in P.items()}
def sub5(npz_xy,npz_pan,npz_y,csv):
    cat=pd.read_csv(csv); s=np.zeros(len(npz_y),bool)
    for i in range(len(npz_y)):
        if npz_y[i]!=1: continue
        inj=cat[cat.image_id==npz_pan[i]]
        if len(inj):
            d=np.hypot(inj.x.values-npz_xy[i,0],inj.y.values-npz_xy[i,1])
            if d.min()<10: s[i]=not bool(inj.iloc[d.argmin()].stack_detection)
    return s
# TRAIN data = train2 (90% panels), THRESH = 10% panel-disjoint holdout
tr=load_parts(R/"cut_train2"); Xtr=np.clip(tr["X"],-20,20); ytr=tr["y"].astype(np.float32)
pans=np.unique(tr["panel"]); rng=np.random.default_rng(7); rng.shuffle(pans)
hp=set(pans[:max(1,len(pans)//10)].tolist()); m_h=np.isin(tr["panel"],list(hp)); m_t=~m_h
# TEST data = test_5sigma
te=np.load(R/"cut_train.npz",allow_pickle=True); Xte=np.clip(te["X"],-20,20); yte=te["y"].astype(np.float32)
s5te=sub5(te["xy"],te["panel"],te["y"],DD/"test_5sigma/test.csv"); n5=int(s5te.sum())
print(f"TRAIN {m_t.sum()} (TP {int(ytr[m_t].sum())}) | THRESH {m_h.sum()} | TEST_5sigma {len(yte)} (TP {int(yte.sum())}, sub5 {n5})",flush=True)
class Net(nn.Module):
    def __init__(s,c=3,w=40):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(c,w),blk(w,2*w),blk(2*w,4*w),nn.AdaptiveAvgPool2d(1)); s.h=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(4*w,1))
    def forward(s,x): return s.h(s.f(x)).squeeze(1)
net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
pw=torch.tensor([(ytr[m_t]==0).sum()/max((ytr[m_t]==1).sum(),1)],device=dev)
def focal(o,t):
    p=torch.sigmoid(o); ce=nn.functional.binary_cross_entropy_with_logits(o,t,reduction="none",pos_weight=pw)
    return ((1-torch.where(t==1,p,1-p))**2*ce).mean()
Xt=torch.tensor(Xtr[m_t]); yt=torch.tensor(ytr[m_t]); N=len(yt); bs=256
def sc(Xa):
    net.eval(); T=torch.tensor(Xa).to(dev)
    with torch.no_grad(): return torch.sigmoid(torch.cat([net(T[k:k+512]) for k in range(0,len(T),512)])).cpu().numpy()
for ep in range(30):
    net.train(); pr=torch.randperm(N)
    for k in range(0,N,bs):
        b=pr[k:k+bs]; opt.zero_grad(); loss=focal(net(Xt[b].to(dev)),yt[b].to(dev)); loss.backward(); opt.step()
    if ep%5==4: print(f"ep{ep} done",flush=True)
# threshold @95% recall on train2 holdout
sh=sc(Xtr[m_h]); thr=float(np.quantile(sh[ytr[m_h]==1],0.05))
ste=sc(Xte); keep=ste>=thr
res=dict(threshold=round(thr,4),test_recall=round(float((keep&(yte==1)).sum()/max((yte==1).sum(),1)),3),
         test_fp_removed=round(float(1-(keep&(yte==0)).sum()/max((yte==0).sum(),1)),3),
         test_fp_kept=int((keep&(yte==0)).sum()),test_nfp=int((yte==0).sum()),
         test_sub5_recall=round(float((keep&(yte==1)&s5te).sum()/max(n5,1)),3),test_auc=round(roc_auc_score(yte,ste),3))
print("\n=== FINAL on HELD-OUT test_5sigma (focal CNN w40/30ep, thr from train2 holdout) ===")
print(json.dumps(res,indent=2))
torch.save(net.state_dict(),R/"cnn_focal_final.pt"); json.dump(res,open(R/"cnn_focal_test5_result.json","w"),indent=2)
print(f"-> saved cnn_focal_final.pt + cnn_focal_test5_result.json")
