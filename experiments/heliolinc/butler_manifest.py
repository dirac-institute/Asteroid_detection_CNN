"""Step 1 (lsst_distrib, registry only): build a tiny MANIFEST of difference_image FITS paths
for one skymap tract + day_obs window, EXCLUDING any (visit, detector) the model trained on.

No pixel data is read or copied -- we only resolve each dataset's datastore URI. The streaming
consumer (discover_stream.py, asteroid_cnn env) then reads those FITS directly with astropy, so
nothing is duplicated to disk. Output: manifest.csv [image_id, visit, detector, band, fits_path].
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
    ap.add_argument("--tract", type=int, default=8489)
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--day-start", type=int, default=20250709, help="day_obs >= (inclusive)")
    ap.add_argument("--day-end", type=int, default=20250723, help="day_obs <  (exclusive)")
    ap.add_argument("--exclude", default=str(REPO / "experiments/heliolinc/train_visit_detector.csv"),
                    help="CSV of train (visit,detector) to exclude (leakage guard)")
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/run_disco/manifest.csv"))
    a = ap.parse_args()

    b = Butler("dp2_prep")
    refs = list(b.registry.queryDatasets(
        "difference_image", collections=STAGE4, findFirst=True,
        where=(f"instrument='LSSTCam' AND skymap='{a.skymap}' AND tract={a.tract} "
               f"AND visit.day_obs>={a.day_start} AND visit.day_obs<{a.day_end}")))
    print(f"[manifest] {len(refs)} difference_image refs in tract {a.tract}, {a.day_start}-{a.day_end}")

    exclude = set()
    if a.exclude and Path(a.exclude).exists():
        ex = pd.read_csv(a.exclude)
        exclude = set(zip(ex.visit.astype(int), ex.detector.astype(int)))

    rows, n_excl = [], 0
    for r in refs:
        v, d = int(r.dataId["visit"]), int(r.dataId["detector"])
        if (v, d) in exclude:
            n_excl += 1
            continue
        rows.append((v, d, r.dataId.get("band", ""), b.getURI(r).ospath))
    df = pd.DataFrame(rows, columns=["visit", "detector", "band", "fits_path"]).drop_duplicates("fits_path")
    df = df.sort_values(["visit", "detector"]).reset_index(drop=True)
    df.insert(0, "image_id", range(len(df)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    days = sorted({int(str(v)[:8]) for v in df.visit})
    print(f"[manifest] {len(df)} panels | {df.visit.nunique()} visits | {df.detector.nunique()} detectors | "
          f"{len(days)} nights (excluded {n_excl} train panels) -> {a.out}")


if __name__ == "__main__":
    main()
