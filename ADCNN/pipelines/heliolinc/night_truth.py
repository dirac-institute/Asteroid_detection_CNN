"""Characterize the known movers on ONE night: per-object same-night sightings, sky-motion
(deg/day), mag — so we know what the same-night pipeline should recover. Also builds the
night's diffim manifest (visit,detector,band,fits_path) and known.csv (ObjID,mjd,ra,dec,mag).
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
    ap.add_argument("--day-obs", type=int, default=20250706)
    ap.add_argument("--tract", type=int, default=8489)
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--exclude", default=str(REPO / "ADCNN/pipelines/heliolinc/train_visit_detector.csv"))
    ap.add_argument("--out-dir", default=str(REPO / "ADCNN/pipelines/heliolinc/run_night"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    b = Butler("dp2_prep")
    refs = list(b.registry.queryDatasets(
        "difference_image", collections=STAGE4, findFirst=True,
        where=(f"instrument='LSSTCam' AND skymap='{a.skymap}' AND tract={a.tract} "
               f"AND visit.day_obs={a.day_obs}")))
    exclude = set()
    if Path(a.exclude).exists():
        ex = pd.read_csv(a.exclude); exclude = set(zip(ex.visit.astype(int), ex.detector.astype(int)))
    rows, n_excl = [], 0
    for r in refs:
        v, d = int(r.dataId["visit"]), int(r.dataId["detector"])
        if (v, d) in exclude:
            n_excl += 1; continue
        rows.append((v, d, r.dataId.get("band", ""), b.getURI(r).ospath))
    man = pd.DataFrame(rows, columns=["visit", "detector", "band", "fits_path"]).drop_duplicates("fits_path")
    man = man.sort_values(["visit", "detector"]).reset_index(drop=True)
    man.insert(0, "image_id", range(len(man)))
    man.to_csv(out / "manifest.csv", index=False)
    print(f"[truth] night {a.day_obs}: {len(man)} panels | {man.visit.nunique()} visits | "
          f"{man.detector.nunique()} detectors (excluded {n_excl} train) -> {out/'manifest.csv'}")

    # Known SSObjects this night
    vis = sorted(man.visit.unique())
    frames = []
    for v in vis:
        sref = list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=STAGE4,
                                             where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
        if not sref:
            continue
        t = b.get(sref[0]).to_pandas()
        if len(t):
            frames.append(pd.DataFrame({"ObjID": t["ObjID"].astype(str), "visit": v,
                                        "mjd": t["fieldMJD_TAI"].astype(float),
                                        "ra": t["RA_deg"].astype(float), "dec": t["Dec_deg"].astype(float),
                                        "mag": t["trailedSourceMag"].astype(float)}))
    kn = pd.concat(frames, ignore_index=True)
    kn.to_csv(out / "known.csv", index=False)

    recs = []
    for oid, g in kn.groupby("ObjID"):
        g = g.sort_values("mjd"); n = len(g)
        dt = g.mjd.max() - g.mjd.min()
        cosd = np.cos(np.radians(g.dec.mean()))
        degday = np.hypot((g.ra.max() - g.ra.min()) * cosd, g.dec.max() - g.dec.min()) / dt if dt > 0 else 0.0
        recs.append(dict(ObjID=oid, n_sight=n, deg_day=degday, mag=g.mag.median(),
                         ra=g.ra.mean(), dec=g.dec.mean(), arc_hr=dt * 24))
    tr = pd.DataFrame(recs)
    tr.to_csv(out / "night_objects.csv", index=False)
    fast2 = tr[(tr.n_sight >= 2) & (tr.deg_day >= 0.5)].sort_values("deg_day", ascending=False)
    print(f"[truth] {kn.ObjID.nunique()} known objects, {len(tr[tr.n_sight>=2])} with >=2 sightings")
    print(f"[truth] FAST movers (>=0.5 deg/day, >=2 sightings) = {len(fast2)}:")
    pd.set_option("display.width", 160)
    print(fast2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    neo = fast2[fast2.deg_day >= 1.0]
    print(f"\n[truth] of those, {len(neo)} move >= 1.0 deg/day (NEO regime)")


if __name__ == "__main__":
    main()
