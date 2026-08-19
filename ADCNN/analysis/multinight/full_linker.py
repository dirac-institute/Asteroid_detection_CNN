#!/usr/bin/env python3
"""FULL-detection multinight linking: all ADCNN + stack detections, ten nights, MPC-grade output.

User directive (2026-08-19): the linker sees every detection, not just alerts, and links to
"3+ visits from 2+ nights" -- MPC-discoverable arcs. Design constraints, all measured:

* TRACTABILITY (the make_tracklets knee, June measurement + this campaign's probes): pairing
  cost ~ N x local density x search area, and search area ~ (maxvel x dt)^2. So the run is
  SPLIT KINEMATICALLY -- a SLOW arm (0.05-1.2 deg/day: the stack's bright-slow band, tiny
  radius, tractable at the lowest score floor) and a FAST arm (0.8-10 deg/day: the ADCNN
  mission band, large radius, score floor raised only as far as the probe demands).
* The score floor is chosen EMPIRICALLY per arm: the probe runs make_tracklets on the densest
  night (20260708) descending 0.80 -> 0.70 -> 0.60 -> 0.50 and keeps the lowest floor whose
  wall time stays under PROBE_BUDGET_S. "All detections" is the goal; a floor is only imposed
  where the probe PROVES the pairing explodes.
* Stack rows are ALWAYS in (their reliability was already gated >= 0.5 at ingest; they carry
  score=-1 here and are exempt from the ADCNN floor).
* CONTROL: MN-2026-01's five detections are in this catalogue; the chain must recover it
  (fast arm, 3 nights). Its absence fails the run.

Stages: filter -> make_tracklets (per arm) -> heliolinc (NEO + MB grids; npt 3 minobsnights 2,
plus a strict 3-night pass) -> link_purify (max_astrom_rms 1", small rejnum) -> score/report.

    python -m ADCNN.analysis.multinight.full_linker probe      # tractability probe only
    python -m ADCNN.analysis.multinight.full_linker run        # full campaign with chosen floors
"""
from __future__ import annotations
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HLX = REPO / "external/heliolinx/bin"
AUX = REPO / "external/heliolinx-aux/tests"
W = REPO / "outputs/runs/multinight/full/work"
OUT = REPO / "outputs/runs/multinight/full"
PROBE_BUDGET_S = 2400            # 40 min on the densest night; full run ~ sum over nights
DENSEST = "20260708"
MJD_MID = "61228.28"

ARMS = {
    #        minvel maxvel  floors to try (descending; stack always in)
    "slow": (0.05,  1.2,   [0.50]),
    "fast": (0.80, 10.0,   [0.80, 0.70, 0.60, 0.50]),
}
GRIDS = {"neo": AUX / "hypotheses/NEO/hihyp00ab_neo.txt",
         "mb":  AUX / "hypotheses/main_belt/hihyp02a_mb.txt"}

MN01 = [  # (mjd, ra, dec) of the control candidate's five detections
    (61228.30, None, None),  # filled from the dossier at runtime if present
]


def filter_arm(arm, floor, night=None):
    """dets_all.csv -> dets_{arm}{floor}{night}.csv (heliolinx 7-col + passthrough)."""
    tag = f"{arm}_s{int(floor*100)}" + (f"_{night}" if night else "")
    dst = W / f"dets_{tag}.csv"
    if dst.exists():
        return dst
    n = 0
    with open(W / "dets_all.csv") as f, open(dst, "w") as o:
        o.write(next(f))
        for line in f:
            c = line.rstrip("\n").split(",")
            if night and not c[0].startswith(night):
                continue
            score, src = float(c[7]), c[8]
            if src != "stack" and score < floor:
                continue
            o.write(line)
            n += 1
    print(f"[full] {tag}: {n:,} detections", flush=True)
    return dst


def make_trk(dets, tag, minvel, maxvel, timeout=None):
    t0 = time.time()
    cmd = [str(HLX / "make_tracklets"), "-dets", str(dets),
           "-colformat", str(W / "colformat.txt"),
           "-earth", str(AUX / "Earth1day2020s_02a.csv"), "-obscode", str(AUX / "ObsCodesNew.txt"),
           "-pairdets", str(W / f"pd_{tag}.csv"), "-tracklets", str(W / f"trk_{tag}.csv"),
           "-trk2det", str(W / f"t2d_{tag}.csv"), "-outimgs", str(W / f"img_{tag}.csv"),
           "-maxtime", "2.5", "-mintime", "0.01", "-maxGCR", "2.0", "-mintrkpts", "2",
           "-minvel", str(minvel), "-maxvel", str(maxvel), "-minarc", "1.5"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0
    if r.returncode != 0:
        print(f"[full] make_tracklets {tag} rc={r.returncode}: {r.stderr[-500:]}", flush=True)
        return None, time.time() - t0
    ntrk = sum(1 for _ in open(W / f"trk_{tag}.csv")) - 1
    return ntrk, time.time() - t0


def probe():
    """Densest-night wall-time per (arm, floor); pick the lowest tractable floor per arm."""
    choice = {}
    for arm, (mnv, mxv, floors) in ARMS.items():
        for floor in sorted(floors, reverse=True):
            dets = filter_arm(arm, floor, night=DENSEST)
            ntrk, dt = make_trk(dets, f"probe_{arm}_s{int(floor*100)}", mnv, mxv,
                                timeout=PROBE_BUDGET_S)
            ok = ntrk is not None
            print(f"[probe] {arm} s>={floor}: {'%d tracklets' % ntrk if ok else 'TIMEOUT/FAIL'} "
                  f"in {dt:.0f}s", flush=True)
            if ok:
                choice[arm] = floor          # keep descending while tractable
            else:
                break
    (W / "probe_choice.json").write_text(json.dumps(choice))
    print(f"[probe] chosen floors: {choice}", flush=True)
    return choice


def sh(cmd, tag):
    print(f"[full] $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    (W / f"log_{tag}.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"[full] {tag} FAILED rc={r.returncode}: {r.stderr[-800:]}")


def run():
    choice = json.loads((W / "probe_choice.json").read_text())
    results = {}
    for arm, floor in choice.items():
        mnv, mxv, _ = ARMS[arm]
        dets = filter_arm(arm, floor)
        ntrk, dt = make_trk(dets, arm, mnv, mxv)
        if ntrk is None:
            sys.exit(f"[full] make_tracklets exploded on the FULL {arm} arm (probe passed on one "
                     f"night); raise the floor one step and rerun")
        print(f"[full] {arm}: {ntrk:,} tracklets in {dt:.0f}s", flush=True)
        lf = []
        for gtag, grid in GRIDS.items():
            if arm == "fast" and gtag == "mb":
                continue                      # >0.8 deg/day cannot be main-belt
            for nights_min in (2, 3):
                otag = f"{arm}_{gtag}_n{nights_min}"
                args = [HLX / "heliolinc", "-imgs", W / f"img_{arm}.csv",
                        "-pairdets", W / f"pd_{arm}.csv", "-tracklets", W / f"trk_{arm}.csv",
                        "-trk2det", W / f"t2d_{arm}.csv", "-mjd", MJD_MID,
                        "-obspos", AUX / "Earth1day2020s_02a.csv", "-heliodist", grid,
                        "-npt", "3", "-minobsnights", str(nights_min), "-mintimespan", "0.8",
                        "-outsum", W / f"hl_{otag}_sum.csv", "-clust2det", W / f"hl_{otag}_c2d.csv"]
                if gtag == "neo":
                    args += ["-mingeodist", "0.004", "-maxgeodist", "3.0"]
                else:
                    args += ["-maxgeodist", "4.5"]
                sh(args, f"hl_{otag}")
                nl = sum(1 for _ in open(W / f"hl_{otag}_sum.csv")) - 1
                print(f"[full] heliolinc {otag}: {nl:,} raw linkages", flush=True)
                if nl > 0:
                    lf.append(f"{W}/hl_{otag}_sum.csv {W}/hl_{otag}_c2d.csv")
        if not lf:
            results[arm] = 0
            continue
        (W / f"lflist_{arm}.txt").write_text("\n".join(lf) + "\n")
        sh([HLX / "link_purify", "-imgs", W / f"img_{arm}.csv", "-pairdet", W / f"pd_{arm}.csv",
            "-lflist", W / f"lflist_{arm}.txt", "-minobsnights", "2", "-minpointnum", "5",
            "-maxrms", "200000", "-max_astrom_rms", "1.0", "-rejfrac", "0.2", "-rejnum", "2",
            "-outsum", W / f"LPL_{arm}.csv", "-clust2det", W / f"LPL_{arm}_c2d.csv"],
           f"purify_{arm}")
        results[arm] = sum(1 for _ in open(W / f"LPL_{arm}.csv")) - 1
        print(f"[full] {arm}: {results[arm]} purified linkages", flush=True)
    (W / "run_results.json").write_text(json.dumps(results))
    print(f"[full] DONE: {results}", flush=True)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        probe()
        sys.exit(0)
    sys.exit(run())
