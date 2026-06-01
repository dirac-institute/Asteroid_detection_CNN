"""Build a diffim manifest (visit, detector, band, fits_path) for a set of skymap tracts + a day_obs
window, excluding any train (visit,detector) for leakage safety. Step 1 of the canonical same-night
NEO pipeline (sn_run.slurm). Registry-only; no pixel data read.

Examples:
  # one tract:
  python -m ADCNN.pipelines.heliolinc.build_manifest --tracts 8731 --day-start 20250706 --day-end 20250707 --out run_x/manifest.csv
  # an ecliptic NEO band over the July fortnight (the canonical discovery field):
  python -m ADCNN.pipelines.heliolinc.build_manifest --tracts 8487-8493,8729-8735 \
      --day-start 20250709 --day-end 20250723 --out run_band/manifest.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from lsst.daf.butler import Butler

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"


def parse_tracts(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracts", required=True, help="comma list / ranges, e.g. 8731 or 8487-8493,8729-8735")
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--day-start", type=int, required=True, help="day_obs >= (inclusive)")
    ap.add_argument("--day-end", type=int, required=True, help="day_obs <  (exclusive)")
    ap.add_argument("--exclude", default=str(REPO / "ADCNN/pipelines/heliolinc/train_visit_detector.csv"),
                    help="CSV of train (visit,detector) to exclude (leakage guard)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tracts = parse_tracts(a.tracts); tl = ",".join(map(str, tracts))
    b = Butler("dp2_prep")
    refs = list(b.registry.queryDatasets(
        "difference_image", collections=STAGE4, findFirst=True,
        where=(f"instrument='LSSTCam' AND skymap='{a.skymap}' AND tract IN ({tl}) "
               f"AND visit.day_obs>={a.day_start} AND visit.day_obs<{a.day_end}")))
    exclude = set()
    if Path(a.exclude).exists():
        ex = pd.read_csv(a.exclude); exclude = set(zip(ex.visit.astype(int), ex.detector.astype(int)))
    rows, nex = [], 0
    for r in refs:
        v, d = int(r.dataId["visit"]), int(r.dataId["detector"])
        if (v, d) in exclude:
            nex += 1; continue
        rows.append((v, d, r.dataId.get("band", ""), b.getURI(r).ospath))
    df = pd.DataFrame(rows, columns=["visit", "detector", "band", "fits_path"]).drop_duplicates("fits_path")
    df = df.sort_values(["visit", "detector"]).reset_index(drop=True)
    df.insert(0, "image_id", range(len(df)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    nights = len({int(str(v)[:8]) for v in df.visit})
    print(f"[manifest] {len(df)} panels | {df.visit.nunique()} visits | {nights} nights | "
          f"{len(tracts)} tracts (excluded {nex} train) -> {a.out}")


if __name__ == "__main__":
    main()
