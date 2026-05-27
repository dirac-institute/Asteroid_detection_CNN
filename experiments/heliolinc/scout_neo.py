"""Scout where FAST movers (NEO-rate, >=1 deg/day) live in DP2, from the Rubin-MISSED pool
(ss_object_unassociated) -- a LOWER BOUND on NEO content, since Rubin's point-tuned detection
under-finds NEO trails. Caches the full unassociated table to parquet (the slow load), computes
on-sky rate from topocentric velocity, and reports where/when fast movers cluster + whether those
visits have ADCNN-processable (DM-53881) difference images.
"""
from lsst.daf.butler import Butler
import pandas as pd, numpy as np, os
from concurrent.futures import ThreadPoolExecutor

HL = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc"
COLL = "LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195/20251118T180806Z"
DP2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
AU = 1.495978707e8
CACHE = f"{HL}/unassoc_full.parquet"
b = Butler("dp2_prep")

if os.path.exists(CACHE):
    u = pd.read_parquet(CACHE); print(f"[cache] {len(u)} rows", flush=True)
else:
    refs = list(b.registry.queryDatasets("ss_object_unassociated", collections=COLL))
    print(f"loading {len(refs)} refs (threaded)...", flush=True)
    def g(r):
        try: return b.get(r).to_pandas()
        except Exception: return None
    with ThreadPoolExecutor(max_workers=32) as ex:
        u = pd.concat([d for d in ex.map(g, refs) if d is not None and len(d)], ignore_index=True)
    u.to_parquet(CACHE); print(f"cached {len(u)} rows -> {CACHE}", flush=True)

px, py, pz = u.topocentricX, u.topocentricY, u.topocentricZ
vx, vy, vz = u.topocentricVX, u.topocentricVY, u.topocentricVZ
dist = np.sqrt(px**2 + py**2 + pz**2)
vtan = np.sqrt(np.maximum(vx**2+vy**2+vz**2 - ((px*vx+py*vy+pz*vz)/dist)**2, 0))
u["rate"] = (vtan/dist)*(180/np.pi)*86400
u["helio_AU"] = u.heliocentricDist/AU
o = u.groupby("ObjID").agg(rate=("rate","median"), helio=("helio_AU","median"),
                           ra=("ra","median"), dec=("dec","median")).reset_index()
print(f"\nmissed objects total: {len(o)}")
for lab,m in [(">=0.5 deg/day",o.rate>=0.5),(">=1.0",o.rate>=1.0),(">=2.0",o.rate>=2.0),
              ("helio<1.3 AU",o.helio<1.3),("NEO(>=1 or helio<1.3)",(o.rate>=1.0)|(o.helio<1.3))]:
    print(f"  {lab}: {int(m.sum())}")
fast = o[(o.rate>=1.0)|(o.helio<1.3)].copy()
print(f"\n=== where the {len(fast)} fast/NEO missed objects cluster (5deg boxes) ===")
fast["rabin"]=(fast.ra//5*5).astype(int); fast["decbin"]=(fast.dec//5*5).astype(int)
print(fast.groupby(["rabin","decbin"]).size().sort_values(ascending=False).head(10).to_string())

# DP2-processable footprint: which tracts have difference_image in DM-53881
dp2_tracts = set()
try:
    for r in b.registry.queryDatasets("difference_image", collections=DP2, where="instrument='LSSTCam'"):
        dp2_tracts.add(r.dataId.get("tract"))
except Exception as e:
    print("dp2 tract query:", str(e)[:120])
print(f"\nDP2 (DM-53881) tracts with diffims: {len([t for t in dp2_tracts if t is not None])}")
fast.to_csv(f"{HL}/fast_missed_objects.csv", index=False)
print(f"-> fast_missed_objects.csv ({len(fast)} fast/NEO objects with ra/dec/rate)")
