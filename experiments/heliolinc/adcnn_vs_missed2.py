"""How many Rubin-MISSED objects (ss_object_unassociated) did ADCNN already detect in run_neo_field?
ss_object_unassociated is dimensioned by tract/patch -> query just this field's tract (fast)."""
from lsst.daf.butler import Butler
import pandas as pd, numpy as np, sys
from concurrent.futures import ThreadPoolExecutor

TRACT = int(sys.argv[1]) if len(sys.argv) > 1 else 6331
RUN = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/run_neo_field"
b = Butler("dp2_prep")
COLL = "LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195/20251118T180806Z"

ad = pd.read_csv(f"{RUN}/adcnn_dets_veres.csv"); ad["visit"] = ad.visit.astype(int)
man = pd.read_csv(f"{RUN}/manifest.csv"); vis_field = set(man.visit.astype(int))
print(f"run_neo_field: ADCNN dets {len(ad)} | visits {len(vis_field)} | tract {TRACT}", flush=True)

# ss_object_unassociated is tract/patch-dimensioned; query by visit (registry joins visit->patch via
# spatial overlap, ~0.2s/visit) and union the unique (tract,patch) refs across the field's visits.
seen, refs = set(), []
for v in vis_field:
    for r in b.registry.queryDatasets("ss_object_unassociated", collections=COLL,
                                      where=f"instrument='LSSTCam' AND visit={v}"):
        key = (r.dataId["tract"], r.dataId["patch"])
        if key not in seen:
            seen.add(key); refs.append(r)
print(f"unique unassoc (tract,patch) refs overlapping field visits: {len(refs)}", flush=True)
def g(r):
    try: return b.get(r).to_pandas()
    except Exception: return None
with ThreadPoolExecutor(max_workers=16) as ex:
    parts = [d for d in ex.map(g, refs) if d is not None and len(d)]
u = pd.concat(parts, ignore_index=True)
u = u[u.visit.isin(vis_field)].copy()
print(f"missed-object detections in field visits: {len(u)} | distinct missed objects: {u.ObjID.nunique()}", flush=True)
if not len(u):
    print("NONE -> tract/visit mismatch"); sys.exit()

recs = {r.id: r.timespan.begin.mjd for r in b.registry.queryDimensionRecords("visit", instrument="LSSTCam") if r.id in vis_field}
u["mjd"] = u.visit.map(recs); u["night"] = u.mjd.astype(int)
pair = u.groupby(["ObjID", "night"]).size().ge(2).groupby("ObjID").sum()
linkable = set(pair[pair >= 2].index)
print(f"linkable missed objects (>=2nt,>=2/nt): {len(linkable)}", flush=True)

TOL = 2.0 / 3600.0
hit = set()
for v, gu in u.groupby("visit"):
    ga = ad[ad.visit == v]
    if not len(ga): continue
    ar, ade = ga.ra.values, ga.dec.values
    for _, row in gu.iterrows():
        cosd = np.cos(np.radians(row.dec))
        if np.hypot((ar - row.ra) * cosd, ade - row.dec).min() < TOL:
            hit.add(row.ObjID)
print(f"\n=== ADCNN vs Rubin-MISSED in run_neo_field (tract {TRACT}) ===")
print(f"missed objects ADCNN DETECTED (>=1 det <2\"): {len(hit)} of {u.ObjID.nunique()}")
print(f"  of which linkable: {len(hit & linkable)} of {len(linkable)}")
# save the linkable+detected missed objects' truth sightings for downstream linking/crossmatch
keep = u[u.ObjID.isin(hit & linkable)][["ObjID", "mjd", "ra", "dec"]]
keep.to_csv(f"{RUN}/missed_truth.csv", index=False)
print(f"-> wrote {len(keep)} truth sightings of {keep.ObjID.nunique()} detected+linkable missed objects to missed_truth.csv")
