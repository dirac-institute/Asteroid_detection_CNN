"""Derive HelioLinC's FP budget empirically. Fix the true-positive detections of the catalogued
objects in a real DP2 field (run_neo_wide, 719 visits / 16 nights / 29 d); sweep the density of
REAL ADCNN false positives added back (FP per visit); at each level run the full linker
(make_tracklets -> heliolinc -> link_refine -> crossmatch) and record:
  * completeness  -- distinct known asteroids recovered / linkable ceiling
  * false_links   -- NEW (unmatched) tracks = spurious links from FP (purity = confirmed/total)
  * n_tracklets   -- make_tracklets pairs (combinatorial load)
  * runtime       -- wallclock of make_tracklets + heliolinc + link_refine (explosion indicator)
The FP budget = the FP/visit where completeness is still ~maxed but false_links/runtime have not
yet exploded. Output: fp_budget_results.csv. Usage: python fp_budget_sweep.py --fpv 0 2 5 10 20 40 80 200
"""
from __future__ import annotations
import argparse, subprocess, time, shutil, re
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trail_tracklets import build_tracklet_files

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
BIN = HL / "heliolinc2/src"
SRC = HL / "run_neo_wide"               # source field: labeled dets + aux files
AUX = ["colformat.txt", "Earth1day2020s_02a.txt", "ObsCodes.txt", "heliohypo_all.txt"]
LINKABLE_NIGHTS = 3                      # heliolinc -minobsnights 3

ap = argparse.ArgumentParser()
ap.add_argument("--fpv", type=int, nargs="+", default=[0, 2, 5, 10, 20, 40, 80, 200],
                help="target FP-per-visit densities to sweep")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--mode", choices=["pair", "trail"], default="pair",
                help="pair = make_tracklets (>=2 dets/night, N_fp^2->links N_fp^6); "
                     "trail = trail_tracklets (one trail=one tracklet, linear->links N_fp^3) [discovery mode]")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
OUT = Path(a.outdir) if a.outdir else (HL / f"fp_budget_{a.mode}")
OUT.mkdir(exist_ok=True)

det = pd.read_csv(SRC / "adcnn_dets_labeled.csv")
det["night"] = np.floor(det.mjd - 0.5).astype(int)
tp = det[det.objid.notna()].copy()
fp = det[det.objid.isna()].copy()
nvis = det.visit.nunique()
# linkable ceiling: known objects with TP on >= LINKABLE_NIGHTS nights
ceil_objs = set(tp.groupby("objid").night.nunique().pipe(lambda s: s[s >= LINKABLE_NIGHTS]).index)
print(f"field: {nvis} visits, {det.night.nunique()} nights | TP {len(tp)} ({tp.objid.nunique()} objs) | "
      f"FP {len(fp)} | linkable ceiling (>= {LINKABLE_NIGHTS} nt): {len(ceil_objs)} objs", flush=True)

MJDREF = round(float(det.mjd.median()), 3)
rng = np.random.default_rng(a.seed)

def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

rows = []
for fpv in a.fpv:
    rd = OUT / f"fpv_{fpv:04d}"; rd.mkdir(exist_ok=True)
    for f in AUX:
        d = rd / f
        if not d.exists(): d.symlink_to(SRC / f)
    # build det subset: ALL TP + up to fpv FP per visit (controlled density)
    keep_fp = [] if fpv == 0 else [
        g.sample(min(fpv, len(g)), random_state=int(rng.integers(1 << 30))) for _, g in fp.groupby("visit")]
    sub = pd.concat([tp] + keep_fp, ignore_index=True).sort_values("mjd")
    sub.to_csv(rd / "adcnn_dets.csv", index=False)
    real_fpv = (len(sub) - len(tp)) / nvis

    t0 = time.time()
    if a.mode == "trail":
        # one trail = one tracklet (endpoints -> state); the actual fast-mover discovery mode
        build_tracklet_files(sub.reset_index(drop=True), str(rd / "Earth1day2020s_02a.txt"), rd)
        ntrk = sum(1 for ln in open(rd / "pairs.txt") if ln.startswith("T "))
    else:
        run([str(BIN / "make_tracklets"), "-dets", "adcnn_dets.csv", "-earth", "Earth1day2020s_02a.txt",
             "-obscode", "ObsCodes.txt", "-colformat", "colformat.txt", "-pairdets", "pairdets.csv",
             "-pairs", "pairs.txt", "-outimgs", "imgs.txt", "-maxtime", "3.0", "-mintime", "0.0",
             "-maxGCR", "2.0", "-mintrkpts", "2", "-maxvel", "2.0", "-minvel", "0.0"], rd)
        ntrk = (sum(1 for _ in open(rd / "pairs.txt")) if (rd / "pairs.txt").exists() else 0)
    r2 = run([str(BIN / "heliolinc"), "-dets", "pairdets.csv", "-pairs", "pairs.txt", "-mjd", str(MJDREF),
              "-obspos", "Earth1day2020s_02a.txt", "-heliodist", "heliohypo_all.txt", "-npt", "3",
              "-minobsnights", "3", "-mintimespan", "0.5", "-out", "hl_clusters.csv", "-outsum", "hl_summary.csv"], rd)
    (rd / "lflist.txt").write_text("hl_clusters.csv hl_summary.csv\n")
    r3 = run([str(BIN / "link_refine"), "-pairdet", "pairdets.csv", "-lflist", "lflist.txt",
              "-maxrms", "100000", "-outfile", "lr.csv", "-outrms", "lr_rms.csv"], rd)
    dt = time.time() - t0

    ntracks = nconf = nobj = nnew = 0
    if (rd / "lr.csv").exists() and (rd / "lr_rms.csv").exists() and (rd / "lr.csv").stat().st_size > 0:
        cm = run(["python", str(HL / "crossmatch.py"), "--run", str(rd),
                  "--known", str(SRC / "known.csv"), "--tol-arcsec", "3.0", "--tol-day", "0.02"], rd)
        out = cm.stdout
        m = re.search(r"(\d+) linked tracks", out);     ntracks = int(m.group(1)) if m else 0
        m = re.search(r"-> (\d+) distinct known", out);  nobj = int(m.group(1)) if m else 0
        m = re.search(r"CONFIRMED \(known\) : (\d+)", out); nconf = int(m.group(1)) if m else 0
        m = re.search(r"NEW candidates    : (\d+)", out);   nnew = int(m.group(1)) if m else 0
    rec = dict(mode=a.mode, fpv_target=fpv, fpv_real=round(real_fpv, 1), n_dets=len(sub), n_tracklets=ntrk,
               n_tracks=ntracks, completeness=round(nobj / max(len(ceil_objs), 1), 3),
               n_known_recovered=nobj, false_links=nnew,
               purity=round(nconf / max(ntracks, 1), 3), runtime_s=round(dt, 1))
    rows.append(rec)
    print(f"  fpv~{real_fpv:5.1f} | dets {len(sub):6d} | tracklets {ntrk:6d} | tracks {ntracks:4d} | "
          f"recovered {nobj:3d}/{len(ceil_objs)} | NEW {nnew:4d} | purity {rec['purity']:.2f} | {dt:5.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "fp_budget_results.csv", index=False)

print("\n=== FP BUDGET SWEEP DONE ===")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n-> {OUT}/fp_budget_results.csv")
