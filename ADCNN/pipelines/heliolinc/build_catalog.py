import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from lsst.daf.butler import Butler
REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
OUT = REPO / "ADCNN/pipelines/heliolinc/run_truth"; OUT.mkdir(parents=True, exist_ok=True)
SIGHT = REPO / "experiments/explore_simreal_gap/test_real_realistic/per_sighting_forced_lsst.csv"
d = pd.read_csv(SIGHT).dropna(subset=["ra", "dec"]).reset_index(drop=True)
b = Butler("dp2_prep")
visits = sorted(int(v) for v in d.visit.unique())
print(f"[mjd] fetching {len(visits)} visits in chunks...", flush=True)
mjd = {}; t0 = time.time(); CH = 400
for i in range(0, len(visits), CH):
    chunk = visits[i:i+CH]
    for rec in b.registry.queryDimensionRecords("visit", instrument="LSSTCam",
              where="visit IN (%s)" % ",".join(map(str, chunk))):
        ts = rec.timespan; mjd[int(rec.id)] = 0.5 * (ts.begin.mjd + ts.end.mjd)
    print(f"  chunk {i//CH+1}/{(len(visits)+CH-1)//CH}: have {len(mjd)} ({time.time()-t0:.0f}s)", flush=True)
print(f"[mjd] got {len(mjd)}/{len(visits)}", flush=True)
d = d[d.visit.isin(mjd)].copy(); d["mjd"] = d.visit.map(mjd)
flux = pd.to_numeric(d.get("lsst_psf_flux", pd.Series(np.nan, index=d.index)), errors="coerce")
d["mag"] = np.where(flux > 0, 31.4 - 2.5*np.log10(flux.clip(lower=1e-3)), 21.5)
d["band1"] = d.band.astype(str).str[0]; d["obscode"] = "I11"
d = d.sort_values("mjd").reset_index(drop=True); d["detid"] = np.arange(len(d))
out = d[["detid","mjd","ra","dec","mag","band1","obscode","ObjID"]]
out.to_csv(OUT / "truth_dets.csv", index=False)
(OUT / "colformat.txt").write_text("IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
print(f"[done] {len(out)} dets, {out.ObjID.nunique()} objects, MJD {out.mjd.min():.3f}-{out.mjd.max():.3f}", flush=True)
print("CATALOG DONE", flush=True)
