"""Profile the inference engine stages on a few panels to find the speed bottleneck."""
import sys, time
from pathlib import Path
import numpy as np, h5py, torch
REPO=Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"); sys.path.insert(0,str(REPO))
from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.candidates import extract_candidates, CandidateExtractorConfig
from ADCNN.inference.rf_postproc import RF_FEATURES_V2, compute_v2_features, apply_rf_v2, load_rf
dev=torch.device("cuda"); torch.backends.cudnn.benchmark=True
m=torch.jit.load(str(REPO/"models/v7_diffim_scripted.pt"),map_location=dev).eval()
rf=load_rf(str(REPO/"models/rf_postproc.pkl"))
h5=REPO/"DATA_DIFFIM/test_5sigma/test.h5"
import warnings; warnings.filterwarnings("ignore")
with h5py.File(h5,"r") as f:
    for pid in range(3):
        img=f["images"][pid][:].astype(np.float32); rl=f["real_labels"][pid][:].astype(np.uint16)
        torch.cuda.synchronize(); t0=time.time()
        prob,sin,cos,agg=predict_panel_overlap_3ch_full(m,img,rl,device=dev)
        torch.cuda.synchronize(); t1=time.time()
        ncand=len(extract_candidates(prob,cfg=CandidateExtractorConfig(),panel_id=pid))
        t2=time.time()
        cand,_=compute_v2_features(prob[None],img[None],sin[None],cos[None],agg[None],real_labels=rl[None],verbose=False)
        t3=time.time()
        cand[list(RF_FEATURES_V2)]=cand[list(RF_FEATURES_V2)].replace([np.inf,-np.inf],np.nan)
        cand=apply_rf_v2(cand,rf); t4=time.time()
        kept=int((cand.score_rf>=0.5).sum())
        print(f"panel{pid}: infer={t1-t0:.2f}s | extract={t2-t1:.2f}s ({ncand} cand) | "
              f"features={t3-t2:.2f}s | rf={t4-t3:.2f}s | kept={kept}/{len(cand)}",flush=True)
print("PROFILE DONE",flush=True)
