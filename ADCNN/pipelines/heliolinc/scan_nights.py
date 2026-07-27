#!/usr/bin/env python3
"""Survey a date range: which nights have difference images, and how much of each is PAIRABLE.

Run this before committing GPU time to a campaign. Two things it answers that the raw dataset
count does not:

  * which nights actually have prompt-processing diffims at all (a night can be in the collection
    list and carry none);
  * how many of a night's visits can contribute to the 2-visit product AT ALL. A visit whose
    pointing is never revisited within the linker window cannot form a pair, so detecting its
    panels produces detections that no 2-visit alert can ever use. On a survey night that repeats
    only part of its footprint this is most of the night.

"Pairable" = the visit shares a boresight with another visit of the same night (within --sep-deg)
separated in time by [--gap-min, --gap-max] minutes -- the same window the linker's
auto_2v_window_min derives, so the count matches what linking will actually see.

Usage:
  python -m ADCNN.pipelines.heliolinc.scan_nights --first 20260629 --last 20260726
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

_REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np


def scan(first, last, repo="embargo", sep_deg=1.0, gap_min=5.0, gap_max=75.0, out_csv=None):
    from lsst.daf.butler import Butler
    import datetime as dt
    b = Butler(repo)
    cols = sorted(b.registry.queryCollections("LSSTCam/runs/prompt/*"))
    d0 = dt.datetime.strptime(str(first), "%Y%m%d").date()
    d1 = dt.datetime.strptime(str(last), "%Y%m%d").date()
    days = [(d0 + dt.timedelta(n)).strftime("%Y%m%d") for n in range((d1 - d0).days + 1)]

    rows = []
    print(f"{'night':<10}{'visits':>7}{'panels':>9}{'pairable_v':>12}{'pair_panels':>13}  collection")
    for day in days:
        cs = [c for c in cols if f"/prompt/{day}/ApPipe" in c]
        if not cs:
            print(f"{day:<10}{'-':>7}{'-':>9}{'-':>12}{'-':>13}  (no ApPipe collection)")
            rows.append((day, 0, 0, 0, 0, ""))
            continue
        col = sorted(cs)[-1]
        per = {}
        for r in b.registry.queryDatasets("difference_image", collections=col, findFirst=True):
            v = r.dataId["visit"]
            per[v] = per.get(v, 0) + 1
        if not per:
            print(f"{day:<10}{0:>7}{0:>9}{0:>12}{0:>13}  (collection exists, no diffims)")
            rows.append((day, 0, 0, 0, 0, col))
            continue
        recs = {}
        for r in b.registry.queryDimensionRecords("visit", instrument="LSSTCam",
                                                  where=f"visit.day_obs = {day}"):
            if r.id not in per:
                continue
            ra = getattr(r, "ra", None); dec = getattr(r, "dec", None)
            if ra is None or dec is None:
                # this Butler's visit record carries a region, not ra/dec columns
                reg = getattr(r, "region", None)
                if reg is None:
                    continue
                c = reg.getBoundingCircle().getCenter()
                from lsst.sphgeom import LonLat
                ll = LonLat(c)
                ra, dec = ll.getLon().asDegrees(), ll.getLat().asDegrees()
            recs[r.id] = (float(ra), float(dec), r.timespan.begin.mjd)
        vs = [(v, ) + recs[v] for v in per if v in recs]
        link = set()
        for i, (v1, r1, d1_, t1) in enumerate(vs):
            for (v2, r2, d2_, t2) in vs[i + 1:]:
                sep = np.hypot((r1 - r2) * np.cos(np.radians(d1_)), d1_ - d2_)
                gap = abs(t1 - t2) * 1440.0
                if sep < sep_deg and gap_min <= gap <= gap_max:
                    link.add(v1); link.add(v2)
        lp = sum(per[v] for v in link)
        rows.append((day, len(per), sum(per.values()), len(link), lp, col))
        print(f"{day:<10}{len(per):>7}{sum(per.values()):>9}{len(link):>12}{lp:>13}  {col.split('/')[-1][:28]}")

    tv = sum(r[2] for r in rows); tp = sum(r[4] for r in rows)
    nn = sum(1 for r in rows if r[2] > 0)
    print(f"\n{nn}/{len(rows)} nights with diffims | panels {tv:,} total, {tp:,} pairable "
          f"({100 * tp / max(tv, 1):.0f}%)")
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["night", "visits", "panels", "pairable_visits", "pairable_panels", "collection"])
            w.writerows(rows)
        print(f"-> {out_csv}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True, help="first day_obs, e.g. 20260629")
    ap.add_argument("--last", required=True)
    # NOT $BUTLER_REPO: pipeline_config.sh sets that to dp2_prep, which holds no prompt-processing
    # collections at all -- sourcing the config then scanning silently reported "no ApPipe
    # collection" for every night of a range that in fact has data.
    ap.add_argument("--repo", default="embargo",
                    help="Butler repo holding the prompt collections (default embargo)")
    ap.add_argument("--sep-deg", type=float, default=1.0, help="boresight match radius")
    ap.add_argument("--gap-min", type=float, default=5.0, help="min visit gap to be pairable (min)")
    ap.add_argument("--gap-max", type=float, default=75.0, help="max visit gap (the 2v window)")
    ap.add_argument("--out", default=None, help="write the table to CSV")
    a = ap.parse_args(argv)
    scan(a.first, a.last, repo=a.repo, sep_deg=a.sep_deg, gap_min=a.gap_min, gap_max=a.gap_max,
         out_csv=a.out)


if __name__ == "__main__":
    sys.exit(main())
