"""Trash-only Monte Carlo for the HelioLinC FP budget -- ONE density per invocation (for SLURM-array
parallelism). Generates synthetic trash at K FP PER PANEL (panel = one detector diffim, the ADCNN
unit): for each real (visit,detector) panel, K detections uniform in that panel's sky footprint, at
the panel's MJD, with trail velocity (rate+direction) drawn from the REAL ADCNN-FP velocity
distribution. So the spatial density and per-trail kinematics are controlled and scalable to any K
(the real pool caps at ~88/panel; synthetic lets us reach 1000). Builds trail-tracklets (one >=6px
streak = one tracklet), runs the REAL linker (heliolinc -> link_refine); every surviving track is a
FALSE link (pure trash). Counts tracks passing the operational gate posRMS<2000 km & obsnights>=3.

Reports FP/PANEL and total_FP (NOT per-visit; this is a targeted field, ~189 CCDs would be a real
visit). Writes one row to fp_budget_mc/result_<K>.csv.  Usage: python fp_budget_mc.py --fpp 32
"""
from __future__ import annotations
import argparse, subprocess, time, sys, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trail_tracklets import build_tracklet_files

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
BIN = HL / "heliolinc2/src"
SRC = HL / "run_neo_wide"
AUX = ["Earth1day2020s_02a.txt", "ObsCodes.txt", "heliohypo_all.txt"]
POSRMS_GATE, NIGHT_GATE = 2000.0, 3
EXPT = 30.0 / 86400.0                       # exposure (day)

ap = argparse.ArgumentParser()
ap.add_argument("--fpp", type=int, required=True, help="FP PER PANEL (detector diffim) to generate")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--nshard", type=int, default=96, help="grid shards (parallel heliolinc, like hunt_parallel.sh)")
ap.add_argument("--outdir", default=str(HL / "fp_budget_mc"))
a = ap.parse_args()
OUT = Path(a.outdir); OUT.mkdir(exist_ok=True)
rd = OUT / f"run_{a.fpp:05d}"; rd.mkdir(exist_ok=True)
for f in AUX:
    if not (rd / f).exists(): (rd / f).symlink_to(SRC / f)

det = pd.read_csv(SRC / "adcnn_dets_labeled.csv")
fp = det[det.objid.isna()].copy()
NVIS = det.visit.nunique()
# per-panel sky footprint (bbox of that detector's real dets) + epoch
fp["panel"] = fp.visit.astype(str) + "_" + fp.detector.astype(str)
pan = fp.groupby("panel").agg(ra_lo=("ra", "min"), ra_hi=("ra", "max"), dec_lo=("dec", "min"),
                              dec_hi=("dec", "max"), mjd=("mjd", "first"))
NPAN = len(pan)
# global real-FP trail velocity distribution (rate deg/day, direction deg)
cd = np.cos(np.radians(fp.dec.values))
v_rate = np.hypot((fp.ra1 - fp.ra0).values * cd, (fp.dec1 - fp.dec0).values) / EXPT
v_dir = np.degrees(np.arctan2((fp.dec1 - fp.dec0).values, (fp.ra1 - fp.ra0).values * cd))
rng = np.random.default_rng(a.seed + a.fpp)

rows = []
for p, r in pan.iterrows():
    n = a.fpp
    ra = rng.uniform(r.ra_lo, r.ra_hi, n) if r.ra_hi > r.ra_lo else np.full(n, r.ra_lo)
    dec = rng.uniform(r.dec_lo, r.dec_hi, n) if r.dec_hi > r.dec_lo else np.full(n, r.dec_lo)
    j = rng.integers(0, len(v_rate), n)
    om, ph = v_rate[j], np.radians(v_dir[j])
    half = 0.5 * om * EXPT                  # half trail length (deg)
    cdp = np.cos(np.radians(dec))
    dra = half * np.cos(ph) / np.clip(cdp, 1e-6, None); ddec = half * np.sin(ph)
    for k in range(n):
        rows.append((ra[k], dec[k], r.mjd, ra[k] - dra[k], dec[k] - ddec[k], ra[k] + dra[k], dec[k] + ddec[k]))
trash = pd.DataFrame(rows, columns=["ra", "dec", "mjd", "ra0", "dec0", "ra1", "dec1"])
trash["mag"] = 22.0; trash["band"] = "r"
MJDREF = round(float(det.mjd.median()), 3)
print(f"fpp={a.fpp}: {len(trash)} synthetic trash over {NPAN} panels (mjdref={MJDREF})", flush=True)

t0 = time.time()
build_tracklet_files(trash, str(rd / "Earth1day2020s_02a.txt"), rd)
ntrk = sum(1 for ln in open(rd / "pairs.txt") if ln.startswith("T "))
# GRID-SHARDED heliolinc (each hypothesis is independent -> split grid NSHARD ways, run in parallel,
# then link_refine merges all shards) -- the production hunt_parallel.sh approach. ~NSHARDx faster.
grid = (rd / "heliohypo_all.txt").read_text().splitlines()
hdr, body = grid[0], grid[1:]
chunks = [body[i::a.nshard] for i in range(a.nshard)]          # round-robin split of the grid
for f in rd.glob("hl_clusters_*.csv"): f.unlink()
for f in rd.glob("hl_summary_*.csv"): f.unlink()
procs = []
for s, ch in enumerate(chunks):
    if not ch: continue
    gp = rd / f"grid_{s:03d}.txt"; gp.write_text(hdr + "\n" + "\n".join(ch) + "\n")
    p = subprocess.Popen([str(BIN / "heliolinc"), "-dets", "pairdets.csv", "-pairs", "pairs.txt",
        "-mjd", str(MJDREF), "-obspos", "Earth1day2020s_02a.txt", "-heliodist", gp.name,
        "-clustrad", "100000", "-npt", "3", "-minobsnights", "3", "-mintimespan", "0.5",
        "-out", f"hl_clusters_{s:03d}.csv", "-outsum", f"hl_summary_{s:03d}.csv"],
        cwd=rd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
nfail = sum(p.wait() != 0 for p in procs)
lflist = [f"hl_clusters_{s:03d}.csv hl_summary_{s:03d}.csv" for s in range(a.nshard)
          if (rd / f"hl_clusters_{s:03d}.csv").exists() and (rd / f"hl_clusters_{s:03d}.csv").stat().st_size > 0]
(rd / "lflist.txt").write_text("\n".join(lflist) + "\n")
subprocess.run([str(BIN / "link_refine"), "-pairdet", "pairdets.csv", "-lflist", "lflist.txt",
                "-maxrms", "100000", "-outfile", "lr.csv", "-outrms", "lr_rms.csv"], cwd=rd, capture_output=True, text=True)
dt = time.time() - t0
print(f"fpp={a.fpp}: {len(procs)} grid shards ({nfail} failed), {dt:.0f}s", flush=True)
n_raw = n_gated = 0
if (rd / "lr_rms.csv").exists() and (rd / "lr_rms.csv").stat().st_size > 0:
    rms = pd.read_csv(rd / "lr_rms.csv"); rms.columns = [c.lstrip("#") for c in rms.columns]
    n_raw = len(rms); n_gated = int(((rms.posRMS < POSRMS_GATE) & (rms.obsnights >= NIGHT_GATE)).sum())
rec = dict(fp_per_panel=a.fpp, total_fp=len(trash), n_tracklets=ntrk, n_clusters_raw=n_raw,
           n_false_gated=n_gated, runtime_s=round(dt, 1))
pd.DataFrame([rec]).to_csv(OUT / f"result_{a.fpp:05d}.csv", index=False)
print(f"fpp={a.fpp} | total {len(trash)} | tracklets {ntrk} | raw clusters {n_raw} | "
      f"FALSE links (posRMS<2000,>=3nt) {n_gated} | {dt:.0f}s -> result_{a.fpp:05d}.csv", flush=True)
