#!/usr/bin/env python3
"""Multi-night linking campaign over the nine delivered ~1k products (2026-08-16, user request).

Feeds every epoch of every DELIVERED alert (the per-night vetted product, NOT the raw streams)
through the validated heliolinx chain -- make_tracklets -> heliolinc (NEO + main-belt hypothesis
grids) -> link_purify -- and keeps ONLY linkages spanning >= 2 distinct nights. Same-night
re-linkages are machinery, not the product: the user asked for a folder that contains nothing
but cross-night candidates.

Layout follows the night-dir convention (top = product, work/ = machinery):
    outputs/runs/multinight/
      candidates.csv     one row per >=2-night linkage (orbit, nights, members, quality)
      candidates/        per-candidate member table + the member alerts' pair images
      README.md          what was run, with the exact invocations
      work/              adapter CSV, tracklets, raw heliolinc/link_purify outputs

Parameters descend from the validated runs (heliolinc-linking-fix: full grid + <=2-week window +
mjd at midpoint; multinight-real-data-threshold: the real-data MB recovery). Deltas for THIS
population (delivered alerts are 1-8 deg/day close-approachers): maxvel 10 deg/day, NEO grid
alongside MB, minobsnights 2 (the user's 2+ nights), mingeodist lowered to 0.01 AU.

    python -m ADCNN.analysis.multinight.run_multinight
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HLX = REPO / "external/heliolinx/bin"
AUX = REPO / "external/heliolinx-aux/tests"
NIGHTS = ["20260629", "20260630", "20260705", "20260706", "20260708",
          "20260710", "20260711", "20260712", "20260713"]
RUNS = REPO / "outputs/runs/10k_cadence"
OUT = REPO / "outputs/runs/multinight"
W = OUT / "work"

GRIDS = {  # two passes; link_purify dedups across both
    "neo": AUX / "hypotheses/NEO/hihyp00ab_neo.txt",     # r 1.1-5.9 AU, mid resolution
    "mb":  AUX / "hypotheses/main_belt/hihyp02a_mb.txt", # r 1.5-9.5 AU
}


def sh(cmd, log):
    print(f"    $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    log.write_text(log.read_text() + f"\n$ {' '.join(str(c) for c in cmd)}\n" + r.stdout + r.stderr
                   if log.exists() else f"$ {' '.join(str(c) for c in cmd)}\n" + r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"FAILED rc={r.returncode}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def build_dets():
    """Delivered alerts -> heliolinx detection CSV; ID carries night+alertId for direct traceback."""
    rows, seen = [], set()
    for n in NIGHTS:
        for al in map(json.loads, open(RUNS / f"run_night_{n}" / "alerts.jsonl")):
            for k, e in enumerate(al["epochs"]):
                key = (e["visit"], e["detector"], round(e["ra"], 6), round(e["dec"], 6))
                if key in seen:      # 3+visit promotions can share epochs with their 2v parents
                    continue
                seen.add(key)
                mag = e.get("mag")
                rows.append((f"{n}:{al['alertId']}:e{k}", f"{e['mjd']:.7f}",
                             f"{e['ra']:.7f}", f"{e['dec']:.7f}",
                             f"{(mag if mag is not None else 22.5):.2f}", "r", "X05"))
    W.mkdir(parents=True, exist_ok=True)
    with open(W / "dets.csv", "w") as f:
        f.write("#ID,MJD,RA,Dec,mag,band,obscode\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    (W / "colformat.txt").write_text(
        "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
    mjds = sorted(float(r[1]) for r in rows)
    print(f"  adapter: {len(rows)} unique detections from the nine delivered products, "
          f"MJD {mjds[0]:.2f}-{mjds[-1]:.2f} (span {mjds[-1]-mjds[0]:.1f} d)")
    return (mjds[0] + mjds[-1]) / 2


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists -- move it aside first (this campaign never overwrites).")
    mjd_mid = build_dets()
    log = W / "chain.log"

    sh([HLX / "make_tracklets", "-dets", W / "dets.csv", "-colformat", W / "colformat.txt",
        "-earth", AUX / "Earth1day2020s_02a.csv", "-obscode", AUX / "ObsCodesNew.txt",
        "-pairdets", W / "pairdets.csv", "-tracklets", W / "tracklets.csv",
        "-trk2det", W / "trk2det.csv", "-outimgs", W / "imgs.csv",
        "-maxtime", "2.5", "-mintime", "0.01", "-maxGCR", "2.0", "-mintrkpts", "2",
        "-minvel", "0.3", "-maxvel", "10.0"], log)

    # ---- FAITHFUL tracklets: one tracklet per DELIVERED alert, no re-pairing --------------------
    # make_tracklets, fed loose epochs at maxvel 10, forms ~23k tracklets of which most are chance
    # cross-alert pairings within a night. MEASURED consequence (this campaign's first pass): both
    # "candidates" it produced were one orphan epoch + a synthetic pair of epochs from two
    # DIFFERENT alerts, with the 4th point rejected by the fit -- the npt=2 chance floor, not a
    # linkage of our product. The question the user asked is "do any two DELIVERED alerts link
    # across nights?", so the primary arm restricts tracklets to exactly the delivered alerts
    # (reusing make_tracklets' pairdets/imgs for formats and indices). The loose arm's outputs
    # stay in work/ as the documented chance-floor control.
    import csv as _csv
    pd_rows = list(_csv.reader(open(W / "pairdets.csv")))
    hdr = [h.lstrip("#") for h in pd_rows[0]]
    i_img, i_orig = hdr.index("image"), hdr.index("origindex")
    dets = [l.split(",") for l in open(W / "dets.csv") if not l.startswith("#")]
    byalert = {}
    for detnum, r in enumerate(pd_rows[1:]):
        night_alert = ":".join(dets[int(r[i_orig])][0].split(":")[:2])
        byalert.setdefault(night_alert, []).append((float(r[0]), detnum, r))
    with open(W / "tracklets_f.csv", "w") as ft, open(W / "trk2det_f.csv", "w") as f2:
        ft.write("#Image1,RA1,Dec1,Image2,RA2,Dec2,npts,trk_ID\n")
        f2.write("#trk_ID,detnum\n")
        for tid, (_, mem) in enumerate(sorted(byalert.items())):
            mem = sorted(mem)
            a, b = mem[0][2], mem[-1][2]
            ft.write(f"{a[i_img]},{float(a[1]):.7f},{float(a[2]):.7f},"
                     f"{b[i_img]},{float(b[1]):.7f},{float(b[2]):.7f},{len(mem)},{tid}\n")
            for _, dn, _r in mem:
                f2.write(f"{tid},{dn}\n")
    print(f"  faithful tracklets: {len(byalert)} (one per delivered alert)")

    lflist = []
    for tag, grid in GRIDS.items():
        args = [HLX / "heliolinc", "-imgs", W / "imgs.csv", "-pairdets", W / "pairdets.csv",
                "-tracklets", W / "tracklets_f.csv", "-trk2det", W / "trk2det_f.csv",
                "-mjd", f"{mjd_mid:.2f}", "-obspos", AUX / "Earth1day2020s_02a.csv",
                # npt MUST equal the night floor: a 2-night object contributes exactly one
                # tracklet per night, so npt 3 + minobsnights 2 is self-contradictory and
                # produced 0 clusters on the first pass.
                "-heliodist", grid, "-npt", "2", "-minobsnights", "2", "-mintimespan", "0.8",
                "-outsum", W / f"hl_{tag}_sum.csv", "-clust2det", W / f"hl_{tag}_c2d.csv"]
        if tag == "neo":
            args += ["-mingeodist", "0.01", "-maxgeodist", "3.0"]
        else:
            args += ["-maxgeodist", "4.0"]
        sh(args, log)
        lflist.append(f"{W}/hl_{tag}_sum.csv {W}/hl_{tag}_c2d.csv")
    (W / "lflist.txt").write_text("\n".join(lflist) + "\n")

    # -maxrms is the STATE-VECTOR rms in KM (not arcsec): 1000 km silently rejected every cluster
    # at "initial screening" (real clusters sit at 2e4-4e4 km; the validated run used 1e5).
    # -minpointnum is EXCLUSIVE (link_purify rejects ptnum<=minpointnum), so the 4-point
    # two-tracklet minimum needs 3, not 4 -- verified with -verbose 2 ("Cluster is too small").
    # -rejnum 0: with only two tracklets, letting the fit REJECT a point reduces the arc to three
    # near-free points and any chance pair "fits" -- the first pass demonstrated exactly that.
    # A faithful 2-night candidate must fit with EVERY delivered point kept.
    sh([HLX / "link_purify", "-imgs", W / "imgs.csv", "-pairdet", W / "pairdets.csv",
        "-lflist", W / "lflist.txt", "-minobsnights", "2", "-minpointnum", "3",
        "-maxrms", "200000", "-rejfrac", "0", "-rejnum", "0",
        "-outsum", W / "LPLsum.csv", "-clust2det", W / "LPLc2d.csv"], log)
    print(f"  chain done -> {W}/LPLsum.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
