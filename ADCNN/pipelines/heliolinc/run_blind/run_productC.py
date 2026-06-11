#!/usr/bin/env python3
"""Blind-test product C: same-night >=3-detection confirmation tier at the FROZEN op_3v_confirm.json.

Per field k: (a) known_k.csv = injected truth sightings at RETIMED mjd (ObjID,ra,dec,mjd) for the
linker's crossmatch; (b) dets3v_k.csv = adcnn_dets_masked_k.csv with mjd re-stamped from the retime
map (the linker sees the same synthetic same-night cadence the injection used); (c) trail_state_link
--op-point op_3v_confirm.json; (d) reduce tracks: a track whose crossmatch ObjID is an injected object
= recovery, unmatched = FP candidate. NO TUNING: op-point JSON is read-only.

Run (repo root on PYTHONPATH): python run_blind/run_productC.py [--fields 0 1 ...]
"""
import argparse, json, os, subprocess, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HELIO = os.path.dirname(HERE)
PY = sys.executable
ALL_KS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29]


def prep_field(k):
    rt = pd.read_csv(f"{HERE}/retime_{k}.csv").set_index("visit").mjd_retimed
    inj = pd.read_csv(f"{HERE}/inject_{k}.csv")
    known = pd.DataFrame(dict(ObjID=inj.objID, ra=inj.ra, dec=inj.dec,
                              mjd=inj.visit.map(rt)))
    assert known.mjd.notna().all(), f"field {k}: inject visit missing from retime map"
    known.to_csv(f"{HERE}/known_{k}.csv", index=False)
    d = pd.read_csv(f"{HERE}/adcnn_dets_masked_{k}.csv")
    mjd_new = d.visit.map(rt)
    assert mjd_new.notna().all(), f"field {k}: dets visit missing from retime map"
    d["mjd"] = mjd_new
    d.to_csv(f"{HERE}/dets3v_{k}.csv", index=False)
    return len(known), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, nargs="*", default=ALL_KS)
    a = ap.parse_args()
    summary = []
    for k in a.fields:
        if not os.path.exists(f"{HERE}/adcnn_dets_masked_{k}.csv"):
            print(f"[C] field {k}: no masked dets -- skip", flush=True)
            continue
        out = f"{HERE}/tracks3v_{k}.csv"
        if not os.path.exists(out):
            nk, nd = prep_field(k)
            r = subprocess.run([PY, f"{HELIO}/trail_state_link.py",
                                "--dets", f"{HERE}/dets3v_{k}.csv",
                                "--known", f"{HERE}/known_{k}.csv",
                                "--op-point", f"{HELIO}/op_3v_confirm.json",
                                "--out", out],
                               capture_output=True, text=True)
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
            print(f"[C] field {k}: dets={nd} known={nk} rc={r.returncode} | " + " | ".join(tail), flush=True)
            if r.returncode != 0 or not os.path.exists(out):
                summary.append(dict(field=k, error=True))
                continue
        t = pd.read_csv(out) if os.path.getsize(out) > 0 else pd.DataFrame()
        if not len(t):
            summary.append(dict(field=k, n_tracks=0))
            continue
        is3 = t.n_epochs >= 3
        m = t.status == "CONFIRMED"   # crossmatched to an injected ObjID (known_k = injected truth)
        summary.append(dict(field=k, n_tracks=int(len(t)), n_3v=int(is3.sum()),
                            n_3v_tp=int((is3 & m).sum()), n_3v_fp=int((is3 & ~m).sum()),
                            n_2v=int((~is3).sum()), n_2v_tp=int((~is3 & m).sum()),
                            rec_obj=sorted(set(t.loc[is3 & m, "match_obj"].astype(str)))))
    json.dump(summary, open(f"{HERE}/productC_summary.json", "w"), indent=1)
    n3 = sum(s.get("n_3v", 0) for s in summary); ntp = sum(s.get("n_3v_tp", 0) for s in summary)
    nfp = sum(s.get("n_3v_fp", 0) for s in summary)
    print(f"PRODUCTC_DONE fields={len(summary)} 3v_tracks={n3} tp={ntp} fp={nfp}", flush=True)


if __name__ == "__main__":
    main()
