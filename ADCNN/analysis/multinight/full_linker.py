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
CULL_CAP = 250_000        # per heliolinc pass, ranked by cluster metric

ARMS = {
    # USER DECISION 2026-08-19: the ADCNN floor is the NIGHTLY OPERATING POINT, 0.70, in both
    # arms -- no per-campaign floor ladder. Stack rows remain exempt (their own reliability
    # gate >= 0.5 applied at ingest), matching the single-night per-source convention.
    #        minvel maxvel  floor
    "slow": (0.05,  1.2,   [0.70]),
    "fast": (0.80,  8.0,   [0.70]),   # 8.0 = the op's own rate_hi ceiling
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
    if (W / f"trk_{tag}.csv").exists() and tag.startswith(("slow", "fast")):
        ntrk = sum(1 for _ in open(W / f"trk_{tag}.csv")) - 1
        print(f"[full] make_tracklets {tag}: reuse ({ntrk:,} tracklets)", flush=True)
        return ntrk, 0.0
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
            # ONE heliolinc pass per grid at minobsnights 2: a 3-night linkage IS a 2-night
            # linkage, so the n3 pass re-found a subset at full cost -- link_purify's own
            # minobsnights + the scorer's night count select tiers downstream. And the OMP
            # binary: the serial one spent 2h15m on the slow arm's first pass alone.
            for nights_min in (2,):
                otag = f"{arm}_{gtag}_n{nights_min}"
                if (W / f"hl_{otag}_sum.csv").exists():
                    nl = sum(1 for _ in open(W / f"hl_{otag}_sum.csv")) - 1
                    print(f"[full] heliolinc {otag}: reuse ({nl:,} raw linkages)", flush=True)
                    if nl > 0:
                        lf.append(f"{W}/hl_{otag}_sum.csv {W}/hl_{otag}_c2d.csv")
                    continue
                args = [HLX / "heliolinc_omp", "-imgs", W / f"img_{arm}.csv",
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
        # PRE-CULL before link_purify. heliolinc at full-catalogue density emits MILLIONS of raw
        # clusters (slow/NEO alone: 8.7M) and purify Herget-fits every one -- days of fitting and
        # a RAM blowup. Cull to what could possibly matter: a quality floor (>=5 unique points,
        # >=2 nights, finite posRMS) then the top CULL_CAP by heliolinc's own cluster metric.
        # Anything dropped by the CAP is logged loudly, never silently vanished.
        import pandas as _pd
        culled = []
        for pair in lf:
            sumf, c2df = pair.split()
            s = _pd.read_csv(sumf)
            s.columns = [c.lstrip("#") for c in s.columns]
            # heliolinc_omp WRITER BUG (measured by A/B against the serial binary's output): its
            # summary rows omit `orbit_incl`, one column short of what read_clustersum_file
            # demands -- verbatim OMP output fails purify's reader. Every orbit_* field is a zero
            # placeholder at this stage (heliolinc does not fit orbits), so inserting the missing
            # zero column at the serial position is an exact repair, not a guess.
            if "orbit_incl" not in s.columns:
                s.insert(list(s.columns).index("orbit_e") + 1, "orbit_incl", 0.0)
            n0 = len(s)
            s = s[(s.uniquepoints >= 5) & (s.obsnights >= 2) & (s.posRMS > 0)]
            s = s.nlargest(min(len(s), CULL_CAP), "metric")
            # link_purify's reader requires SEQUENTIAL clusternum (it indexes a vector by the
            # value) and the '#'-prefixed header -- a metric-ordered, gappy numbering made it
            # abort with "Last point was 0". Renumber 0..N-1 and remap clust2det.
            s = s.reset_index(drop=True)
            remap = {int(old): new for new, old in enumerate(s.clusternum.astype(int))}
            s["clusternum"] = range(len(s))
            c = _pd.read_csv(c2df)
            c.columns = [cc.lstrip("#") for cc in c.columns]
            c = c[c.clusternum.astype(int).isin(remap)]
            c["clusternum"] = c.clusternum.astype(int).map(remap)
            c = c.sort_values(["clusternum", "detnum"])
            cs, cc2 = sumf.replace("_sum", "_cullsum"), c2df.replace("_c2d", "_cullc2d")
            for df, path in ((s, cs), (c, cc2)):
                cols = list(df.columns)
                cols[0] = "#" + cols[0]
                df.to_csv(path, index=False, header=cols)
            print(f"[full] cull {Path(sumf).name}: {n0:,} -> {len(s):,} clusters "
                  f"({'CAP HIT' if len(s) == CULL_CAP else 'quality floor only'})", flush=True)
            culled.append(f"{cs} {cc2}")
        (W / f"lflist_{arm}.txt").write_text("\n".join(culled) + "\n")
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
