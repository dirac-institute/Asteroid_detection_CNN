"""Stage B (lsst_distrib / Butler): turn the ADCNN detection catalog (pixel x,y per panel,
produced by Stage A = ``python -m ADCNN.inference.catalog``) into a HelioLinC detection
catalog in sky coordinates.

For each (visit, detector) it fetches the WCS + MJD once from the Butler, converts every
candidate ``(x, y)`` -> ``(RA, Dec)`` (adding the panel's ``xy0`` origin), and writes
``detid, mjd, ra, dec, mag, band, obscode`` (+ bookkeeping cols) plus the ``colformat.txt``
HelioLinC needs. ``--validate`` matches detections to the known-object truth sightings on the
same panel and reports the WCS separation + how many truths were recovered.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lsst.geom as geom
from lsst.daf.butler import Butler

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
TRUTH = REPO / "experiments/explore_simreal_gap/test_real_realistic/per_sighting_forced_lsst.csv"
COLFORMAT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"


def _load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in ("visit", "detector", "x", "y") if c not in df.columns]
    if missing:
        raise ValueError(f"catalog {path} missing {missing} (run Stage A with --panels to attach routing keys)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", default=str(REPO / "experiments/heliolinc/run_adcnn/catalog.csv"),
                    help="Stage-A catalog (csv/parquet) with visit,detector,x,y[,band,score_rf]")
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/run_adcnn/adcnn_dets.csv"))
    ap.add_argument("--validate", action="store_true", help="report WCS sep + truth recovery")
    a = ap.parse_args()

    cat = _load_catalog(a.cands)
    butler = Butler("dp2_prep", collections=[STAGE3, STAGE2])
    truth = pd.read_csv(TRUTH).dropna(subset=["ra", "dec"]) if a.validate else None

    rows = []
    for (visit, det), grp in cat.groupby(["visit", "detector"]):
        try:
            pvi = butler.get("preliminary_visit_image",
                             dataId={"instrument": "LSSTCam", "visit": int(visit), "detector": int(det)})
            wcs, xy0 = pvi.getWcs(), pvi.getBBox().getBegin()
            mjd = pvi.getInfo().getVisitInfo().getDate().get()
        except Exception as e:
            print(f"  WCS fail v={visit} d={det}: {e}", flush=True)
            continue
        for _, r in grp.iterrows():
            sp = wcs.pixelToSky(geom.Point2D(r.x + xy0.getX(), r.y + xy0.getY()))
            rows.append(dict(detid=len(rows), mjd=mjd, ra=sp.getRa().asDegrees(), dec=sp.getDec().asDegrees(),
                             mag=float(r.get("mag", 21.5)), band=str(r.get("band", "r"))[0], obscode="I11",
                             visit=int(visit), detector=int(det), x=r.x, y=r.y,
                             score_rf=r.get("score_rf", float("nan"))))

    out = pd.DataFrame(rows)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    (out_path.parent / "colformat.txt").write_text(COLFORMAT)
    print(f"[stageB] wrote {len(out)} detections (RA/Dec) over {out[['visit','detector']].drop_duplicates().shape[0]} "
          f"visit-detectors -> {out_path}", flush=True)

    if a.validate and truth is not None and len(out):
        seps = []
        for (v, d), g in out.groupby(["visit", "detector"]):
            for _, t in truth[(truth.visit == v) & (truth.detector == d)].iterrows():
                sep = np.hypot((g.ra - t.ra) * np.cos(np.radians(t.dec)), g.dec - t.dec).min() * 3600
                if sep < 5:
                    seps.append(sep)
        print(f"[validate] {len(seps)} truth sightings matched a detection within 5\"; "
              f"median WCS sep {np.median(seps):.2f}\"" if seps else "[validate] no matches", flush=True)
    print("STAGEB DONE", flush=True)


if __name__ == "__main__":
    main()
