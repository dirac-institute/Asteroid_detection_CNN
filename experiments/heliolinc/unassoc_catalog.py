"""Build a detection catalog from Rubin's ss_object_unassociated -- the pool of real detections that
Rubin's Solar System Processing SAW but FAILED to link into a tracked object. Each row carries the
truth ObjID, so linking these and matching member detections back to ObjID measures exactly how many
Rubin-MISSED objects our make_tracklets+heliolinc pipeline recovers (= genuine new discoveries).

Restricts to a RA/Dec box + night window, maps visit->MJD, and writes:
  std_dets.csv  [detid, mjd, ra, dec, mag, band, obscode]   (heliolinc input)
  truth.csv     [detid, ObjID]                              (per-detection truth for recovery scoring)
  std_colformat.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from lsst.daf.butler import Butler

COLL = "LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195/20251118T180806Z"
COLFORMAT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra0", type=float, required=True); ap.add_argument("--ra1", type=float, required=True)
    ap.add_argument("--dec0", type=float, required=True); ap.add_argument("--dec1", type=float, required=True)
    ap.add_argument("--night0", type=int, required=True); ap.add_argument("--nnights", type=int, default=14)
    ap.add_argument("--out", required=True, help="output run dir")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    b = Butler("dp2_prep")
    refs = list(b.registry.queryDatasets("ss_object_unassociated", collections=COLL))
    print(f"[unassoc] {len(refs)} refs; loading...", flush=True)
    u = pd.concat([b.get(r).to_pandas() for r in refs], ignore_index=True)
    u = u[(u.ra >= a.ra0) & (u.ra <= a.ra1) & (u.dec >= a.dec0) & (u.dec <= a.dec1)].copy()
    need = set(u.visit.unique())
    recs = {r.id: (r.timespan.begin.mjd, str(r.physical_filter)[:1])
            for r in b.registry.queryDimensionRecords("visit", instrument="LSSTCam") if r.id in need}
    u["mjd"] = u.visit.map(lambda v: recs.get(v, (np.nan, "r"))[0])
    u["band"] = u.visit.map(lambda v: recs.get(v, (np.nan, "r"))[1])
    u = u.dropna(subset=["mjd"])
    u["night"] = u.mjd.astype(int)
    u = u[(u.night >= a.night0) & (u.night < a.night0 + a.nnights)].copy()
    u = u.sort_values("mjd").reset_index(drop=True)
    u.insert(0, "detid", range(len(u)))

    dets = pd.DataFrame({"detid": u.detid, "mjd": u.mjd, "ra": u.ra, "dec": u.dec,
                         "mag": 21.0, "band": u.band, "obscode": "I11"})
    dets.to_csv(out / "std_dets.csv", index=False)
    u[["detid", "ObjID"]].to_csv(out / "truth.csv", index=False)
    (out / "std_colformat.txt").write_text(COLFORMAT)

    # how many objects are linkable in this slice (>=2 nights each w/ >=2 dets)?
    pair = u.groupby(["ObjID", "night"]).size().ge(2).groupby("ObjID").sum()
    print(f"[unassoc] RA[{a.ra0},{a.ra1}] Dec[{a.dec0},{a.dec1}] nights[{a.night0},{a.night0+a.nnights}) "
          f"-> {len(dets)} dets, {u.ObjID.nunique()} missed-objects, "
          f"{int((pair>=2).sum())} LINKABLE -> {out}/std_dets.csv", flush=True)


if __name__ == "__main__":
    main()
