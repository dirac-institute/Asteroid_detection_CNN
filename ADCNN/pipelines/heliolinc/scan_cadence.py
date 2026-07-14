"""Find dense-cadence tract-nights that ARE processable (have difference images in stage4): rank
(tract, night) by the number of distinct same-night visits with diffim panels. >=3 needed for
3-visit linking; >=4-5 ideal (faint completeness rises with revisits). Registry-only.

queryDataIds with the tract dimension does the spatial join, so each diffim dataId carries its tract
(unlike queryDatasets, whose dataId is visit+detector only). Output cadence.csv [tract, night,
n_visits, n_panels] sorted desc. Build a manifest with build_manifest --tracts <tract> --day <night>.
"""
from __future__ import annotations
import argparse
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
from lsst.daf.butler import Butler

STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skymap", default="lsst_cells_v1")
    ap.add_argument("--min-visits", type=int, default=4)
    ap.add_argument("--butler-repo", default=os.environ.get("BUTLER_REPO", "main"))
    ap.add_argument("--out", default=str(OUTPUTS / "query_snapshots/cadence.csv"))
    a = ap.parse_args()

    b = Butler(a.butler_repo)
    visit_day = {int(r.id): int(r.day_obs) for r in b.registry.queryDimensionRecords("visit", where="instrument='LSSTCam'")}
    tn_visits = defaultdict(set); tn_panels = defaultdict(int); n = 0
    for did in b.registry.queryDataIds(["visit", "detector", "tract"], datasets="difference_image",
                                       collections=STAGE4, where=f"instrument='LSSTCam' AND skymap='{a.skymap}'"):
        t = int(did["tract"]); v = int(did["visit"]); night = visit_day.get(v)
        if night is None:
            continue
        tn_visits[(t, night)].add(v); tn_panels[(t, night)] += 1
        n += 1
        if n % 300000 == 0:
            print(f"[cadence] {n} diffim panels scanned...", flush=True)
    rows = [dict(tract=t, night=night, n_visits=len(vs), n_panels=tn_panels[(t, night)])
            for (t, night), vs in tn_visits.items()]
    df = pd.DataFrame(rows).sort_values(["n_visits", "n_panels"], ascending=False).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    dense = df[df.n_visits >= a.min_visits]
    print(f"[cadence] {n} diffim panels | {len(df)} tract-nights | {len(dense)} with >= {a.min_visits} visits -> {a.out}", flush=True)
    print(dense.head(40).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
