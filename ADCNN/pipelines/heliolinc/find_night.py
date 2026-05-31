"""Find the best SINGLE NIGHT for same-night NEO detection in the DP2 diffim field.

Registry-only (no pixel reads). For each day_obs in the window + tract, count visits/detectors,
then pull preloaded_ss_object_visit (known SSObject ephemerides) and rank nights by the number of
FAST movers (high sky-motion, NEO-like) that have >= 2 sightings THAT NIGHT — the objects a
same-night linker could actually catch. Prints a ranked table; writes the per-night summary.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from lsst.daf.butler import Butler

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tract", type=int, default=8489)
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--day-start", type=int, default=20250701)
    ap.add_argument("--day-end", type=int, default=20250801)
    ap.add_argument("--fast-degday", type=float, default=0.5, help="NEO-like sky-motion threshold (deg/day)")
    ap.add_argument("--out", default=str(REPO / "ADCNN/pipelines/heliolinc/run_night/nights.csv"))
    a = ap.parse_args()

    b = Butler("dp2_prep")
    # 1) Enumerate difference_image visits/detectors per night in the field.
    refs = list(b.registry.queryDatasets(
        "difference_image", collections=STAGE4, findFirst=True,
        where=(f"instrument='LSSTCam' AND skymap='{a.skymap}' AND tract={a.tract} "
               f"AND visit.day_obs>={a.day_start} AND visit.day_obs<{a.day_end}")))
    rows = []
    for r in refs:
        did = r.dataId
        rows.append((int(did["visit"]), int(did["detector"]), did.get("band", "")))
    diffim = pd.DataFrame(rows, columns=["visit", "detector", "band"]).drop_duplicates()
    # visit -> day_obs (first 8 digits of the visit id)
    diffim["day_obs"] = diffim.visit.map(lambda v: int(str(v)[:8]))
    print(f"[night] {len(refs)} diffim refs | {diffim.visit.nunique()} visits | "
          f"{diffim.day_obs.nunique()} nights in tract {a.tract}")

    per_night_diffim = diffim.groupby("day_obs").agg(
        n_visit=("visit", "nunique"), n_panel=("visit", "size"),
        n_det=("detector", "nunique")).reset_index()

    # 2) For nights with >=2 visits, pull known SSObjects and find fast movers w/ >=2 same-night sightings.
    summary = []
    for _, nr in per_night_diffim.iterrows():
        day = int(nr.day_obs)
        vis = sorted(diffim[diffim.day_obs == day].visit.unique())
        frames = []
        for v in vis:
            sref = list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=STAGE4,
                                                 where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
            if not sref:
                continue
            t = b.get(sref[0]).to_pandas()
            if not len(t):
                continue
            frames.append(pd.DataFrame({"ObjID": t["ObjID"].astype(str), "visit": v,
                                        "mjd": t["fieldMJD_TAI"].astype(float),
                                        "ra": t["RA_deg"].astype(float), "dec": t["Dec_deg"].astype(float),
                                        "mag": t["trailedSourceMag"].astype(float)}))
        if not frames:
            summary.append(dict(day_obs=day, n_visit=int(nr.n_visit), n_panel=int(nr.n_panel),
                                n_known=0, n_known_2x=0, n_fast=0, n_fast_2x=0))
            continue
        kn = pd.concat(frames, ignore_index=True)
        # per-object same-night multiplicity + sky motion (deg/day)
        fast_2x = known_2x = n_fast = 0
        for oid, g in kn.groupby("ObjID"):
            g = g.sort_values("mjd")
            n = len(g)
            if n >= 2:
                known_2x += 1
                dt = g.mjd.max() - g.mjd.min()
                cosd = np.cos(np.radians(g.dec.mean()))
                dra = (g.ra.max() - g.ra.min()) * cosd; ddec = g.dec.max() - g.dec.min()
                degday = np.hypot(dra, ddec) / dt if dt > 0 else 0.0
                if degday >= a.fast_degday:
                    n_fast += 1; fast_2x += 1
        summary.append(dict(day_obs=day, n_visit=int(nr.n_visit), n_panel=int(nr.n_panel),
                            n_known=kn.ObjID.nunique(), n_known_2x=known_2x,
                            n_fast=n_fast, n_fast_2x=fast_2x))

    s = pd.DataFrame(summary).sort_values(["n_fast_2x", "n_known_2x", "n_visit"], ascending=False).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out, index=False)
    pd.set_option("display.width", 160)
    print(f"\n[night] ranked nights (fast = >= {a.fast_degday} deg/day, 2x = >= 2 same-night sightings):")
    print(s.to_string(index=False))
    print(f"\n[night] -> {a.out}")
    if len(s) and s.iloc[0].n_fast_2x > 0:
        best = s.iloc[0]
        print(f"\n[night] BEST: day_obs={int(best.day_obs)} | {int(best.n_visit)} visits, {int(best.n_panel)} panels | "
              f"{int(best.n_fast_2x)} fast movers with >=2 same-night sightings")


if __name__ == "__main__":
    main()
