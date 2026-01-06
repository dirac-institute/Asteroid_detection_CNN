import numpy as np
import pandas as pd
from lsst.daf.butler import Butler

repo = "/repo/main"
collection = "LSSTComCam/runs/DRP/DP1/w_2025_17/DM-50530"
where = "instrument='LSSTComCam' AND skymap='lsst_cells_v1' AND day_obs>=20241101 AND day_obs<=20241127 AND exposure.observation_type='science' AND band in ('u','g','r','i','z','y') AND (exposure not in (0))"
butler = Butler(repo, collections=collection)
refs = list(set(butler.registry.queryDatasets("preliminary_visit_image", where=where, instrument="LSSTComCam", findFirst=True)))
refs = sorted(refs, key=lambda r: str(r.dataId["visit"]*1000+r.dataId["detector"]))

bad = []

# Only test pairs that exist in refs
pairs = sorted({(int(r.dataId["visit"]), int(r.dataId["detector"])) for r in refs})

for i, (visit, detector) in enumerate(pairs, 1):
    print(f"\rProcessing {i}/{len(pairs)}  bad:{len(bad)}", end="", flush=True)
    try:
        calexp = butler.get(
            "preliminary_visit_image",
            dataId={"instrument": "LSSTComCam", "visit": visit, "detector": detector},
        )
        wcs = calexp.wcs
        if wcs is None:
            bad.append((visit, detector, "wcs_none"))
        else:
            try:
                _ = wcs.getPixelScale()
            except Exception as e:
                bad.append((visit, detector, f"getPixelScale_fail:{type(e).__name__}"))
    except Exception as e:
        bad.append((visit, detector, f"butler_get_fail:{type(e).__name__}"))

print()  # newline

bad_df = pd.DataFrame(bad, columns=["visit", "detector", "reason"])
bad_df.to_csv("bad_pairs.csv", index=False)
print("Wrote bad_pairs.csv with", len(bad_df), "rows")
