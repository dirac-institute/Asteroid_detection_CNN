"""Compare real trail-candidate features vs the synthetic positives/negatives the
RF was trained on. Answers: which features push real trails into the RF's
'background' region? Read-only / CPU."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.inference.diffim_postproc_v2 import RF_FEATURES_V2, load_rf

OUT = REPO / "experiments/explore_simreal_gap"
SC = Path("/sdf/scratch/users/m/mrakovci/rf_leakage")
feats = list(RF_FEATURES_V2)

vc = pd.read_parquet(SC / "val_cand.parquet")
vl = np.load(SC / "val_labels.npy")
pos = vc[vl == 1].copy()        # synthetic injected trails (RF positives)
neg = vc[vl == 0].copy()        # synthetic FPs (RF negatives)
real = pd.read_csv(OUT / "probe_real_features.csv")
print(f"synth pos={len(pos)} neg={len(neg)}  real truth-cands={len(real)}")

rf = load_rf(str(REPO / "experiments/explore_rf_leakage/rf_postproc_v2_valtrain.pkl"))
imp = pd.Series(rf.feature_importances_, index=feats)

# RF score sanity: synth pos vs real
Xr = real[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
Xp = pos[feats].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
print(f"\nRF score: synth-pos med={np.median(rf.predict_proba(Xp)[:,1]):.3f}  "
      f"real med={np.median(rf.predict_proba(Xr)[:,1]):.3f}")

# For each feature: where does real sit between pos and neg medians?
# r=0 -> real looks like synth POS ; r=1 -> real looks like synth NEG (=rejected)
def med(df, f): return np.nanmedian(df[f].replace([np.inf,-np.inf], np.nan))
rowsout = []
for f in feats:
    mp, mn, mr = med(pos, f), med(neg, f), med(real, f)
    denom = (mn - mp)
    r = (mr - mp) / denom if abs(denom) > 1e-9 else np.nan
    rowsout.append(dict(feat=f, imp=imp[f], med_pos=mp, med_neg=mn, med_real=mr,
                        real_like_neg=r))
t = pd.DataFrame(rowsout)
# focus: high-importance features where real looks like NEG (r near/above 1)
t["score"] = t["imp"] * t["real_like_neg"].clip(0, 2)
print("\n=== features where REAL trail-cands look like synthetic FPs "
      "(weighted by RF importance) ===")
top = t.sort_values("score", ascending=False).head(20)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(top[["feat","imp","med_pos","med_neg","med_real","real_like_neg"]]
          .round(3).to_string(index=False))

# condition on high v7 prob (real all fire ~1): among synth cands with max_p>0.9,
# how do pos/neg/real compare on the key MF-SNR features?
print("\n=== conditioned on max_p>0.9 (matched to real's high v7 confidence) ===")
hp = vc[(vc.max_p > 0.9)]; hpl = vl[(vc.max_p > 0.9).to_numpy()]
hpos, hneg = hp[hpl == 1], hp[hpl == 0]
key = ["max_p","mean_p","top5_mean_p","or_agg_max","mf_snr","lmf_snr_30",
       "masnr_30","masnr_50","or_snr_L30","integrated_logit","area","mf_length",
       "elongation","loc_dipole","loc_max_z"]
cmp = pd.DataFrame({
    "imp": [imp[k] for k in key],
    "synthPOS_hp": [np.nanmedian(hpos[k].replace([np.inf,-np.inf],np.nan)) for k in key],
    "synthNEG_hp": [np.nanmedian(hneg[k].replace([np.inf,-np.inf],np.nan)) for k in key],
    "REAL": [np.nanmedian(real[k].replace([np.inf,-np.inf],np.nan)) for k in key],
}, index=key)
print(f"(synth high-p: pos={len(hpos)} neg={len(hneg)})")
with pd.option_context("display.width", 200):
    print(cmp.round(3).to_string())
t.to_csv(OUT / "feature_comparison.csv", index=False)
print("\nCOMPARE DONE")
