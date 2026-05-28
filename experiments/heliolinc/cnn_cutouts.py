"""Build a cutout dataset for the post-v7 CNN rejecter. For each panel: run v7 (prob+agg), find
candidates, label by injection overlap, save multi-channel cutout [diffim/sigma, v7_prob, v7_agg].
CHUNKED + RESUMABLE: writes one part-npz per CHUNK panels into <out_dir>/, tracks done panels in
done.txt, skips them on restart -> survives ampere preemption (use --requeue). Usage:
  python cnn_cutouts.py <h5> <csv> <out_dir> [--k 48] [--fp-cap 600] [--chunk 40]"""
import sys, argparse, numpy as np, pandas as pd, h5py, torch
from pathlib import Path
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.features import compute_v2_features, FEATURES_V2
from ADCNN.inference.features import label_candidates_by_injection_overlap

ap=argparse.ArgumentParser(); ap.add_argument("h5"); ap.add_argument("csv"); ap.add_argument("out")
ap.add_argument("--k",type=int,default=48); ap.add_argument("--fp-cap",type=int,default=600); ap.add_argument("--chunk",type=int,default=40)
a=ap.parse_args(); K=a.k; H=K//2
OUT=Path(a.out); OUT.mkdir(parents=True,exist_ok=True); DONE=OUT/"done.txt"
done=set(int(x) for x in DONE.read_text().split()) if DONE.exists() else set()
V7=REPO/"models/v7_diffim_scripted.pt"; dev=torch.device("cuda")
model=torch.jit.load(str(V7),map_location=dev).eval(); cat=pd.read_csv(a.csv)
def cut(arr,x,y):
    Hh,Ww=arr.shape; x,y=int(round(x)),int(round(y)); o=np.zeros((K,K),np.float32)
    x0,x1,y0,y1=max(0,x-H),min(Ww,x+H),max(0,y-H),min(Hh,y+H)
    cc=arr[y0:y1,x0:x1]; o[:cc.shape[0],:cc.shape[1]]=cc; return o
buf=dict(X=[],y=[],panel=[],cid=[],feat=[],xy=[]); chunk_pids=[]
def flush():
    if not buf["X"]: return
    p0=chunk_pids[0]; fn=OUT/f"part_{p0:04d}.npz"
    np.savez_compressed(fn,X=np.array(buf["X"],np.float32),y=np.array(buf["y"],np.int8),
        panel=np.array(buf["panel"],np.int32),cid=np.array(buf["cid"],np.int32),
        feat=np.array(buf["feat"],np.float32),xy=np.array(buf["xy"],np.float32))
    done.update(chunk_pids); DONE.write_text(" ".join(map(str,sorted(done))))
    for k in buf: buf[k].clear()
    print(f"  [flush] part_{p0:04d}.npz ({len(done)} panels done)",flush=True); chunk_pids.clear()
with h5py.File(a.h5,"r") as f:
    npan=f["images"].shape[0]
    for pid in range(npan):
        if pid in done: continue
        img=f["images"][pid].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
        prob,sn,cs,agg=predict_panel_overlap_3ch_full(model,img,rl,device=dev)
        prob=prob.astype(np.float32); agg=np.asarray(agg,np.float32)
        cand,_=compute_v2_features({pid:prob},{pid:img},{pid:sn},{pid:cs},{pid:agg},real_labels={pid:rl},verbose=False)
        chunk_pids.append(pid)
        if len(cand):
            lab=label_candidates_by_injection_overlap(cand,cat,{pid:prob})
            s=float(np.median(np.abs(img-np.median(img)))*1.4826) or 1.0
            feat=cand[list(FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
            keep=set(range(len(cand)))
            if a.fp_cap>0:
                fp_i=[i for i in range(len(cand)) if lab[i]==0]
                if len(fp_i)>a.fp_cap: keep-=set(np.random.default_rng(pid).choice(fp_i,len(fp_i)-a.fp_cap,replace=False).tolist())
            for i,((_,r),y_) in enumerate(zip(cand.iterrows(),lab)):
                if i not in keep: continue
                xx,yy=r.x_centroid,r.y_centroid
                buf["X"].append(np.stack([cut(img,xx,yy)/s,cut(prob,xx,yy),cut(agg,xx,yy)]))
                buf["y"].append(int(y_)); buf["panel"].append(pid); buf["cid"].append(int(r.candidate_id))
                buf["feat"].append(feat[i]); buf["xy"].append((xx,yy))
            print(f"  panel {pid}: {len(cand)} cand, {int((lab==1).sum())} TP",flush=True)
        if len(chunk_pids)>=a.chunk: flush()
flush()
np.save(OUT/"feat_names.npy",np.array(list(FEATURES_V2)))
print(f"CUTOUTS DONE: {len(done)} panels -> {OUT}/ ({len(list(OUT.glob('part_*.npz')))} parts)")
