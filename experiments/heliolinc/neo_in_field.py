"""Did ADCNN detect the known NEOs in run_neo_field? known.csv (Rubin SSObject ephemerides:
ObjID,mjd,ra,dec) has no velocity, so derive each object's on-sky rate from consecutive ephemeris
points, flag NEO-rate (>=0.5 deg/day) objects, and crossmatch ADCNN detections against their
per-visit positions. Everything on disk -> fast."""
import pandas as pd, numpy as np
RUN = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/run_neo_field"
k = pd.read_csv(f"{RUN}/known.csv").dropna(subset=["ra","dec","mjd"]).sort_values(["ObjID","mjd"])
ad = pd.read_csv(f"{RUN}/adcnn_dets_veres.csv")

# per-object on-sky rate from consecutive same-object ephemeris points (deg/day)
def obj_rate(g):
    if len(g) < 2: return np.nan
    g = g.sort_values("mjd"); dt = g.mjd.diff();
    dra = g.ra.diff()*np.cos(np.radians(g.dec)); dd = g.dec.diff()
    sep = np.hypot(dra, dd); r = sep/dt
    r = r[(dt>0.001)&(dt<2.0)]   # use intra/adjacent-night pairs
    return r.median() if len(r) else np.nan
rate = k.groupby("ObjID").apply(obj_rate, include_groups=False)
nights = k.assign(night=k.mjd.astype(int)).groupby("ObjID").night.nunique()
summ = pd.DataFrame({"rate":rate,"nnights":nights}).dropna()
print(f"known objects in field: {len(summ)}")
for lab,m in [(">=0.5 deg/day",summ.rate>=0.5),(">=1.0",summ.rate>=1.0),(">=2.0",summ.rate>=2.0)]:
    print(f"  NEO-rate {lab}: {int(m.sum())}  (of which >=2 nights: {int((m&(summ.nnights>=2)).sum())})")

neo = set(summ[summ.rate>=0.5].index)
kn = k[k.ObjID.isin(neo)].copy(); kn["night"]=kn.mjd.astype(int)
# crossmatch ADCNN dets to NEO ephemeris points (2 arcsec, 30 min)
hit=set()
adn = ad.assign(night=ad.mjd.astype(int))
for _,row in kn.iterrows():
    c = adn[np.abs(adn.mjd-row.mjd)<0.02]
    if not len(c): continue
    if np.hypot((c.ra-row.ra)*np.cos(np.radians(row.dec)), c.dec-row.dec).min()*3600 < 2.0:
        hit.add(row.ObjID)
linkable = set(summ[(summ.rate>=0.5)&(summ.nnights>=2)].index)
print(f"\n=== ADCNN vs known NEOs in run_neo_field ===")
print(f"NEO-rate (>=0.5) known objects: {len(neo)} | >=2 nights: {len(linkable)}")
print(f"ADCNN DETECTED: {len(hit & neo)} | detected & >=2 nights (linkable): {len(hit & linkable)}")
# fastest detected ones
det = summ.loc[sorted(hit & neo)].sort_values("rate",ascending=False)
print("\nfastest ADCNN-detected known NEOs (rate deg/day, nights):")
print(det.head(15).to_string())
kn[kn.ObjID.isin(hit & linkable)][["ObjID","mjd","ra","dec"]].to_csv(f"{RUN}/neo_known_truth.csv",index=False)
print(f"\n-> neo_known_truth.csv ({len(hit & linkable)} detected+linkable known NEOs)")
