"""Scout fast movers (NEO-rate) from ss_source (ASSOCIATED detections = known objects Rubin linked).
Single dataset (~1.6M rows, one butler.get). ra/dec from the topocentric position vector (ICRS),
on-sky rate from topocentric velocity. These are KNOWN objects -> recovery/validation targets and
they show WHERE in DP2 fast movers actually appear."""
from lsst.daf.butler import Butler
import pandas as pd, numpy as np
AU = 1.495978707e8
b = Butler("dp2_prep")
COLL = "LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195/20251118T180806Z"
ref = list(b.registry.queryDatasets("ss_source", collections=COLL))[0]
print("loading ss_source...", flush=True)
s = b.get(ref).to_pandas()
print(f"ss_source rows: {len(s)} | cols: {list(s.columns)}", flush=True)

px, py, pz = s.topocentricX, s.topocentricY, s.topocentricZ
vx, vy, vz = s.topocentricVX, s.topocentricVY, s.topocentricVZ
dist = np.sqrt(px**2 + py**2 + pz**2)
vtan = np.sqrt(np.maximum(vx**2+vy**2+vz**2 - ((px*vx+py*vy+pz*vz)/dist)**2, 0))
s["rate"] = (vtan/dist)*(180/np.pi)*86400
# sky position from topocentric vector (assume equatorial/ICRS frame)
s["ra"] = (np.degrees(np.arctan2(py, px))) % 360
s["dec"] = np.degrees(np.arctan2(pz, np.sqrt(px**2+py**2)))
s["helio_AU"] = s.heliocentricDist/AU
s["topo_AU"] = dist/AU

# per-object summary
o = s.groupby("ssObjectId").agg(rate=("rate","median"), helio=("helio_AU","median"),
    topo=("topo_AU","median"), ra=("ra","median"), dec=("dec","median"), ndet=("ssObjectId","size")).reset_index()
print(f"\nassociated objects: {len(o)} | detections: {len(s)}")
for lab,m in [(">=0.5 deg/day",o.rate>=0.5),(">=1.0",o.rate>=1.0),(">=2.0",o.rate>=2.0),
              (">=5.0",o.rate>=5.0),("helio<1.3 AU",o.helio<1.3),("NEO(>=1 or helio<1.3)",(o.rate>=1.0)|(o.helio<1.3))]:
    print(f"  {lab}: {int(m.sum())}")
neo = o[(o.rate>=1.0)|(o.helio<1.3)].copy()
print(f"\n=== where the {len(neo)} fast/NEO known objects cluster (5deg boxes) ===")
neo["rabin"]=(neo.ra//5*5).astype(int); neo["decbin"]=(neo.dec//5*5).astype(int)
top = neo.groupby(["rabin","decbin"]).agg(nobj=("ssObjectId","nunique"), medrate=("rate","median")).sort_values("nobj",ascending=False)
print(top.head(12).to_string())
neo.to_csv("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_known_objects.csv", index=False)
print(f"\n-> neo_known_objects.csv ({len(neo)} fast/NEO known objects)")
