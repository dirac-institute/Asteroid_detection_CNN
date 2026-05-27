from lsst.daf.butler import Butler
import pandas as pd, time
b = Butler("dp2_prep")
COLL="LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195/20251118T180806Z"
dt=b.registry.getDatasetType("ss_object_unassociated")
print("dimensions:", list(dt.dimensions.names), flush=True)
man=pd.read_csv("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/run_neo_field/manifest.csv")
v=int(man.visit.iloc[0])
t=time.time()
try:
    refs=list(b.registry.queryDatasets("ss_object_unassociated",collections=COLL,where=f"visit={v} AND instrument='LSSTCam'"))
    print(f"refs for visit {v}: {len(refs)} ({time.time()-t:.1f}s)", flush=True)
except Exception as e:
    print("visit-filter ERR:", str(e)[:300], flush=True)
