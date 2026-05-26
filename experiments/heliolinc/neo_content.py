"""NEO content of the Rubin-MISSED pool (ss_object_unassociated) in run_neo_field, and how many
ADCNN detected. NEOs move fast / sit close; classify by on-sky rate (from topocentric velocity)
and heliocentric distance. Tells us whether a NEO-targeted recovery/discovery run is worthwhile."""
from lsst.daf.butler import Butler
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor

RUN = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/run_neo_field"
COLL = "LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195/20251118T180806Z"
AU = 1.495978707e8  # km
b = Butler("dp2_prep")

ad = pd.read_csv(f"{RUN}/adcnn_dets_veres.csv"); ad["visit"] = ad.visit.astype(int)
man = pd.read_csv(f"{RUN}/manifest.csv"); vis = set(man.visit.astype(int))
seen, refs = set(), []
for v in vis:
    for r in b.registry.queryDatasets("ss_object_unassociated", collections=COLL,
                                      where=f"instrument='LSSTCam' AND visit={v}"):
        k = (r.dataId["tract"], r.dataId["patch"])
        if k not in seen: seen.add(k); refs.append(r)
def g(r):
    try: return b.get(r).to_pandas()
    except Exception: return None
with ThreadPoolExecutor(max_workers=16) as ex:
    u = pd.concat([d for d in ex.map(g, refs) if d is not None and len(d)], ignore_index=True)
u = u[u.visit.isin(vis)].copy()

# on-sky angular rate (deg/day) from topocentric pos/vel
px, py, pz = u.topocentricX, u.topocentricY, u.topocentricZ
vx, vy, vz = u.topocentricVX, u.topocentricVY, u.topocentricVZ
dist = np.sqrt(px**2 + py**2 + pz**2)  # km
vdotr = (px*vx + py*vy + pz*vz) / dist
vtan = np.sqrt(np.maximum(vx**2+vy**2+vz**2 - vdotr**2, 0))  # km/s
u["rate_degday"] = (vtan / dist) * (180/np.pi) * 86400
u["helio_AU"] = u.heliocentricDist / AU
u["topo_AU"] = dist / AU
# one row per object (median rate)
o = u.groupby("ObjID").agg(rate=("rate_degday","median"), helio=("helio_AU","median"),
                           topo=("topo_AU","median"), ndet=("ObjID","size")).reset_index()
recs = {r.id: r.timespan.begin.mjd for r in b.registry.queryDimensionRecords("visit", instrument="LSSTCam") if r.id in vis}
u["mjd"] = u.visit.map(recs); u["night"] = u.mjd.astype(int)
pair = u.groupby(["ObjID","night"]).size().ge(2).groupby("ObjID").sum()
linkable = set(pair[pair>=2].index)

print(f"missed objects in field: {len(o)} | rate(deg/day) p50={o.rate.median():.2f} p90={o.rate.quantile(.9):.2f} max={o.rate.max():.2f}")
for lab, m in [("NEO-ish rate>=0.5 deg/day", o.rate>=0.5), ("fast rate>=1.0", o.rate>=1.0),
               ("helio<1.3 AU", o.helio<1.3), ("NEO (rate>=0.5 OR helio<1.3)", (o.rate>=0.5)|(o.helio<1.3))]:
    print(f"  {lab}: {int(m.sum())}")
# ADCNN detection of the NEO subset
neo = set(o[(o.rate>=0.5)|(o.helio<1.3)].ObjID)
un = u[u.ObjID.isin(neo)]
TOL=2.0/3600.0; hit=set()
for v,gu in un.groupby("visit"):
    ga=ad[ad.visit==v]
    if not len(ga): continue
    ar,ade=ga.ra.values,ga.dec.values
    for _,row in gu.iterrows():
        if np.hypot((ar-row.ra)*np.cos(np.radians(row.dec)),ade-row.dec).min()<TOL: hit.add(row.ObjID)
print(f"\n=== NEO missed objects in run_neo_field ===")
print(f"NEOs total: {len(neo)} | linkable: {len(neo & linkable)} | ADCNN-detected: {len(hit)} | detected+linkable: {len(hit & linkable)}")
keep = u[u.ObjID.isin(hit & linkable)][["ObjID","mjd","ra","dec"]]
keep.to_csv(f"{RUN}/missed_neo_truth.csv", index=False)
o[o.ObjID.isin(neo)].to_csv(f"{RUN}/neo_missed_objects.csv", index=False)
print(f"-> missed_neo_truth.csv ({keep.ObjID.nunique()} objs), neo_missed_objects.csv")
