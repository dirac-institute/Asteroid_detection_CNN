"""Build the KNOWN-object reference for crossmatching (lsst_distrib): for every visit in the
discovery window, pull `preloaded_ss_object_visit` (Rubin's known-SSObject ephemeris service,
derived from mpcorb) and collect each known object's predicted position.

Output: known.csv [ObjID, mjd, ra, dec, mag] -- the catalogued objects expected in these visits.
A HelioLinC-linked track that matches one of these is a CONFIRMED re-discovery; a track matching
none is a NEW (uncatalogued) candidate. crossmatch.py consumes this directly.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd
from lsst.daf.butler import Butler

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")  # all runtime OUTPUT goes here
STAGE4 = os.environ.get("BUTLER_COLLECTION", "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4")
BUTLER_REPO = os.environ.get("BUTLER_REPO", "main")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(OUTPUTS / "runs/run_disco/manifest.csv"))
    ap.add_argument("--out", default=str(OUTPUTS / "runs/run_disco/known.csv"))
    ap.add_argument("--butler-repo", default=BUTLER_REPO, help="Butler repo (default $BUTLER_REPO or main)")
    ap.add_argument("--collection", default=STAGE4, help="SSObject collection (default $BUTLER_COLLECTION)")
    a = ap.parse_args()

    b = Butler(a.butler_repo)
    visits = sorted(pd.read_csv(a.manifest).visit.unique())
    frames = []; n_missing = 0
    for v in visits:
        refs = list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=a.collection,
                                             where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
        if not refs:
            n_missing += 1                      # visit has no preloaded ephemerides -> can't confirm recoveries here
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
    if n_missing:
        frac = n_missing / max(len(visits), 1)
        msg = f"[known] WARNING: {n_missing}/{len(visits)} visits ({frac:.0%}) had NO preloaded ephemerides"
        msg += " -- CONFIRMED/NEW labels for tracks in those visits are unreliable" if frac > 0.05 else ""
        print(msg)
    if len(known):
        print(f"[known] mag range {known.mag.min():.1f}..{known.mag.max():.1f} (median {known.mag.median():.1f})")


if __name__ == "__main__":
    main()
