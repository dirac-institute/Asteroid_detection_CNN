"""Quantify the TRAIL-ANGLE measurement quality (the real linkage limiter) on injected truth.

The trail-state linker accepts a track only if each detection's measured trail PA tracks the object's
motion (physical_check). So linkage success depends on how accurately ADCNN measures the trail angle
(beta) -- not just on detection. Here we match ADCNN detections (test_detections.csv) to the injected
truth (DATA/test.csv, which carries the TRUE beta + trail_length + mag + SNR), and measure the
per-detection trail-angle error |beta_meas - beta_true| (mod 180) vs faintness/brightness. The fraction
with |beta_err| < pa_tol is the per-detection prob of passing the trail-PA check; a 3-epoch track needs
all three to pass, so 3-visit linkage completeness ~ recall(mag) x P(beta ok)^3.
"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
R="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
TRUTH=f"{R}/DATA/test.csv"; MEAS=f"{R}/Evaluation/catalogs_thr0/test_detections.csv"
TOLPX=20.0
truth=pd.read_csv(TRUTH); meas=pd.read_csv(MEAS)
# match each truth trail to nearest measured detection on same panel within TOLPX (centroid)
truth=truth.reset_index(drop=True); truth["mbeta"]=np.nan; truth["mscore"]=np.nan; truth["mlen"]=np.nan
mby=meas.groupby("image_id").indices
for img,t_idx in truth.groupby("image_id").indices.items():
    m=mby.get(img)
    if m is None: continue
    mr=meas.iloc[m]
    tree=cKDTree(mr[["x","y"]].values)
    tr=truth.iloc[t_idx]
    dist,j=tree.query(tr[["x","y"]].values,k=1)
    ok=dist<=TOLPX
    gi=tr.index[ok]
    truth.loc[gi,"mbeta"]=mr.beta.values[j[ok]]
    truth.loc[gi,"mscore"]=mr.score.values[j[ok]]
    truth.loc[gi,"mlen"]=mr.length.values[j[ok]]
m=truth[truth.mbeta.notna()].copy()
m["berr"]=np.abs(((m.mbeta - m.beta + 90) % 180) - 90)          # deg, [0,90]
out=[]
out.append(f"matched {len(m)}/{len(truth)} injected trails to a detection (<= {TOLPX}px)")
out.append(f"overall median |beta_err| = {m.berr.median():.1f} deg | <10deg {(m.berr<10).mean():.2f} | <20deg {(m.berr<20).mean():.2f}")
def tab(col,bins,name):
    out.append(f"\n=== trail-angle error vs {name} ===")
    out.append(f"{'bin':>14}{'n':>6}{'med|berr|':>11}{'P(<10)':>8}{'P(<20)':>8}")
    lab=pd.cut(m[col],bins)
    for b,idx in m.groupby(lab,observed=True).groups.items():
        s=m.loc[idx]
        out.append(f"{str(b):>14}{len(s):>6}{s.berr.median():>11.1f}{(s.berr<10).mean():>8.2f}{(s.berr<20).mean():>8.2f}")
tab("mag",[0,20,21,22,23,24,25,99],"magnitude (bright->faint)")
tab("SNR_estimation",[0,3,4,5,6,8,100],"SNR")
tab("trail_length",[0,10,20,30,45,60,200],"true trail length (px)")
# fold: 3-visit linkage 'pass-all-3' factor at pa_tol=10 and 20, by mag (among DETECTED)
out.append("\n=== P(all 3 trails ok)=P(<tol)^3 by mag  [the extra factor the binomial estimate omitted] ===")
lab=pd.cut(m.mag,[0,20,21,22,23,24,25,99])
for b,idx in m.groupby(lab,observed=True).groups.items():
    s=m.loc[idx]; p10=(s.berr<10).mean(); p20=(s.berr<20).mean()
    out.append(f"  mag {str(b):>12}: P(<10)^3={p10**3:.2f}  P(<20)^3={p20**3:.2f}")
open("/tmp/trail_quality_out.txt","w").write("\n".join(out)+"\n")
print("DONE")
