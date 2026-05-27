"""Score how many Rubin-MISSED objects (ss_object_unassociated) our HelioLinC tracks recovered.

Uses the per-detection truth (truth.csv: detid->ObjID) instead of a spatial crossmatch: for each
refined track, look up the ObjID of its member detections; if a single ObjID accounts for >= min_frac
of them, that missed object is RECOVERED (a genuine discovery Rubin's pipeline failed to make).
Tracks whose members are a blend of ObjIDs are spurious (MIXED).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-frac", type=float, default=0.7)
    a = ap.parse_args()
    run = Path(a.run)

    truth = pd.read_csv(run / "truth.csv").set_index("detid").ObjID
    lr = pd.read_csv(run / "lr.csv")
    lr.columns = [c.lstrip("#").strip() for c in lr.columns]
    rms = pd.read_csv(run / "lr_rms.csv"); rms.columns = [c.lstrip("#").strip() for c in rms.columns]
    # lr maps each linked detection to a clusternum; the detid column varies by build
    idcol = next((c for c in ["detid", "i1", "detindex", "index"] if c in lr.columns), lr.columns[1])
    clcol = next((c for c in ["clusternum", "clusternum", "cluster"] if c in lr.columns), lr.columns[0])

    rows = []
    for cl, g in lr.groupby(clcol):
        objs = truth.reindex(g[idcol].values).dropna()
        if not len(objs):
            continue
        top = objs.value_counts()
        obj, n = top.index[0], int(top.iloc[0])
        frac = n / len(objs)
        prm = rms[rms[clcol] == cl]
        rows.append(dict(cluster=int(cl), ndet=len(g), nobj_distinct=objs.nunique(),
                         top_obj=obj, frac=round(frac, 2),
                         posRMS=float(prm.posRMS.iloc[0]) if len(prm) else np.nan,
                         obsnights=int(prm.obsnights.iloc[0]) if len(prm) and "obsnights" in prm else -1,
                         recovered=frac >= a.min_frac))
    res = pd.DataFrame(rows)
    if not len(res):
        print("no tracks"); return
    rec = res[res.recovered]
    print(f"refined tracks      : {len(res)}")
    print(f"RECOVERED (>= {a.min_frac:.0%} one ObjID) : {len(rec)} tracks -> "
          f"{rec.top_obj.nunique()} distinct MISSED objects rediscovered")
    print(f"spurious/MIXED      : {len(res)-len(rec)} tracks")
    res.sort_values(["recovered", "posRMS"], ascending=[False, True]).to_csv(run / "recovery.csv", index=False)
    if len(rec):
        print("\n-- sample recovered missed-objects --")
        print(rec.sort_values("posRMS")[["cluster", "ndet", "obsnights", "top_obj", "frac", "posRMS"]]
              .head(15).to_string(index=False))
    print(f"\n-> {run}/recovery.csv")


if __name__ == "__main__":
    main()
