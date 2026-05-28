import sys; sys.path.insert(0,'/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN')
sys.path.insert(0,'/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/explore_simreal_gap')
import h5py, numpy as np, pandas as pd, torch
from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
import ADCNN.evaluation.detection as evals
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
dev=torch.device("cuda")
m=torch.jit.load(f"{REPO}/models/segmentation_model.pt",map_location=dev).eval()  # consolidated models/ + ADCNN code
cat=pd.read_csv(f"{REPO}/DATA_DIFFIM/test_5sigma/test.csv")
with h5py.File(f"{REPO}/DATA_DIFFIM/test_5sigma/test.h5") as f:
    probs=[]; 
    for i in range(f["images"].shape[0]):
        p,*_=predict_panel_overlap_3ch_full(m,f["images"][i][:].astype(np.float32),f["real_labels"][i][:],device=dev)
        probs.append(p.astype(np.float32))
probs=np.stack(probs)
tp,fp,fn,_=evals.objectwise_confusion(cat,(probs>=0.5).astype(np.uint8),0.5,use_threads=True,max_workers=8)
print(f"VERIFY consolidated seg_model-only objectwise recall @0.5: {tp}/{tp+fn} = {100*tp/(tp+fn):.1f}%  (reg2 baseline 96.0%)",flush=True)
print("VERIFY DONE",flush=True)
