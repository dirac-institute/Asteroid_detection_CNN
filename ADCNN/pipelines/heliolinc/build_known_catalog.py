"""Build the KNOWN-object reference for crossmatching (lsst_distrib): for every visit in the
discovery window, pull `preloaded_ss_object_visit` (Rubin's known-SSObject ephemeris service,
derived from mpcorb) and collect each known object's predicted position.

Output: known.csv [ObjID, mjd, ra, dec, mag] -- the catalogued objects expected in these visits.
A HelioLinC-linked track that matches one of these is a CONFIRMED re-discovery; a track matching
none is a NEW (uncatalogued) candidate. crossmatch.py consumes this directly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from lsst.daf.butler import Butler

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "ADCNN/pipelines/heliolinc/run_disco/manifest.csv"))
    ap.add_argument("--out", default=str(REPO / "ADCNN/pipelines/heliolinc/run_disco/known.csv"))
    a = ap.parse_args()

    b = Butler("dp2_prep")
    visits = sorted(pd.read_csv(a.manifest).visit.unique())
    frames = []
    for v in visits:
        refs = list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=STAGE4,
                                             where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
        if not refs:
            continue
        t = b.get(refs[0]).to_pandas()
        frames.append(pd.DataFrame({
            "ObjID": t["ObjID"].astype(str),
            "mjd": t["fieldMJD_TAI"].astype(float),
            "ra": t["RA_deg"].astype(float),
            "dec": t["Dec_deg"].astype(float),
            "mag": t["trailedSourceMag"].astype(float),
        }))
    known = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ObjID", "mjd", "ra", "dec", "mag"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    known.to_csv(a.out, index=False)
    print(f"[known] {len(known)} known-object sightings across {len(visits)} visits | "
          f"{known.ObjID.nunique()} distinct catalogued objects -> {a.out}")
    if len(known):
        print(f"[known] mag range {known.mag.min():.1f}..{known.mag.max():.1f} (median {known.mag.median():.1f})")


if __name__ == "__main__":
    main()
