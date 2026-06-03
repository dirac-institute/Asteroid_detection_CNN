"""For the top dense-cadence fields (from cadence.csv), count recoverable known FAST movers: known
SSObjects with >=3 distinct same-night epochs and on-sky speed >= a threshold (deg/day). These are
the positive controls a 3-visit discovery run should recover. Ranks fields so we pick the best first.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from lsst.daf.butler import Butler

STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
SOLARDAY = 86400.0


def field_known(b, visits):
    frames = []
    for v in visits:
        refs = list(b.registry.queryDatasets("preloaded_ss_object_visit", collections=STAGE4,
                                             where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
        if not refs:
            continue
        t = b.get(refs[0]).to_pandas()
        frames.append(pd.DataFrame(dict(ObjID=t["ObjID"].astype(str), mjd=t["fieldMJD_TAI"].astype(float),
                                        ra=t["RA_deg"].astype(float), dec=t["Dec_deg"].astype(float),
                                        mag=t["trailedSourceMag"].astype(float))))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ObjID","mjd","ra","dec","mag"])


def obj_stats(g):
    g = g.sort_values("mjd"); t = g.mjd.to_numpy()
    ep = 1 + int(np.sum(np.diff(t) * SOLARDAY > 120)) if len(t) > 1 else 1
    cd = np.cos(np.radians(g.dec.mean()))
    if len(t) > 1:
        dt = np.diff(t); spd = np.hypot(np.diff(g.ra.to_numpy()) * cd, np.diff(g.dec.to_numpy())) / np.where(dt > 0, dt, 1e9)
        speed = float(np.nanmax(spd))
    else:
        speed = 0.0
    return ep, speed, float(g.mag.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadence", default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/cadence.csv")
    ap.add_argument("--min-visits", type=int, default=6)
    ap.add_argument("--max-visits", type=int, default=40, help="skip huge DDF fields for the first pass")
    ap.add_argument("--top", type=int, default=15, help="how many candidate fields to probe")
    ap.add_argument("--cap-visits", type=int, default=30, help="cap visits queried per field (cost)")
    ap.add_argument("--fast-degday", type=float, default=1.0)
    ap.add_argument("--out", default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/field_fastmovers.csv")
    a = ap.parse_args()

    cad = pd.read_csv(a.cadence)
    cand = cad[(cad.n_visits >= a.min_visits) & (cad.n_visits <= a.max_visits)].head(a.top)
    print(f"[ff] probing {len(cand)} fields (n_visits {a.min_visits}-{a.max_visits})", flush=True)
    b = Butler("dp2_prep")
    rows = []
    for _, f in cand.iterrows():
        visits = [int(x) for x in str(f.visits).split()][:a.cap_visits]
        k = field_known(b, visits)
        nfast = nfast3 = nfaint3 = 0; examples = []
        if len(k):
            for o, g in k.groupby("ObjID"):
                ep, speed, mag = obj_stats(g)
                if speed >= a.fast_degday:
                    nfast += 1
                    if ep >= 3:
                        nfast3 += 1
                        faint = 21.5 <= mag <= 24.5            # the ideal faint-fast control (clean long trail)
                        if faint:
                            nfaint3 += 1
                            examples.append(f"{o}({speed:.1f}d/d,{ep}ep,m{mag:.1f})")
        rows.append(dict(night=int(f.night), ra=round(f.ra,3), dec=round(f.dec,3), n_visits=int(f.n_visits),
                         n_fast=nfast, n_fast_3ep=nfast3, n_faintfast_3ep=nfaint3, examples="; ".join(examples[:6])))
        print(f"  night {int(f.night)} ra{f.ra:.1f} dec{f.dec:.1f} nv={int(f.n_visits)}: "
              f"fast>=3ep={nfast3} FAINT-fast>=3ep={nfaint3}", flush=True)
    out = pd.DataFrame(rows).sort_values(["n_faintfast_3ep","n_fast_3ep"], ascending=False).reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print(f"\n[ff] ranked -> {a.out}", flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
