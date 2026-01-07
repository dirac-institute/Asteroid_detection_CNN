import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from lsst.daf.butler import Butler

repo = "/repo/main"
collection = "LSSTComCam/runs/DRP/DP1/w_2025_17/DM-50530"
where = ("instrument='LSSTComCam' AND skymap='lsst_cells_v1' AND day_obs>=20241101 AND day_obs<=20241127 "
         "AND exposure.observation_type='science' AND band in ('u','g','r','i','z','y') AND (exposure not in (0))")

butler = Butler(repo, collections=collection)
refs = list(set(butler.registry.queryDatasets(
    "preliminary_visit_image", where=where, instrument="LSSTComCam", findFirst=True
)))

pairs = sorted({(int(r.dataId["visit"]), int(r.dataId["detector"])) for r in refs})
print("Pairs:", len(pairs), end="  ")


def check_pair(args):
    repo, collection, visit, detector = args
    try:
        b = Butler(repo, collections=collection)
        calexp = b.get(
            "preliminary_visit_image",
            dataId={"instrument": "LSSTComCam", "visit": int(visit), "detector": int(detector)},
        )
        wcs = calexp.wcs
        photocalib = calexp.getPhotoCalib()
        if (wcs is None):
            return (visit, detector, "wcs_none")
        if (photocalib is None):
            return (visit, detector, "photocalib_none")
        try:
            _ = wcs.getPixelScale()
            return None
        except Exception as e:
            return (visit, detector, f"getPixelScale_fail:{type(e).__name__}")
    except Exception as e:
        return (visit, detector, f"butler_get_fail:{type(e).__name__}")


max_workers = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
print ("Using max_workers =", max_workers)
tasks = [(repo, collection, v, d) for (v, d) in pairs]

bad = []
done = 0
total = len(tasks)

with ProcessPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(check_pair, t) for t in tasks]
    for fut in as_completed(futures):
        res = fut.result()
        done += 1
        if res is not None:
            bad.append(res)
        print(f"\rProcessed {done}/{total}  bad:{len(bad)}", end="", flush=True)

print()

bad_df = pd.DataFrame(bad, columns=["visit", "detector", "reason"])
bad_df.to_csv("bad_visits.csv", index=False)
print("Wrote bad_visits.csv with", len(bad_df), "rows")
