"""Completeness (and purity) of fast-NEO linking vs FP-per-panel. Inject synthetic >=1 deg/day NEOs
(real 2-body orbits, synth_neo) on an LSST cadence (4 nights, 2 visits, 15-day window), add a real-FP
background at K detections PER PANEL locally around each NEO detection (the FP that can actually
contaminate a NEO's cluster; bulk FP elsewhere give ~0 false links per the trash-MC), run the full
trail-tracklet -> sharded heliolinc -> link_refine -> gate(posRMS<2000,>=3nt) pipeline, and crossmatch
the gated tracks to the injected NEO truth.
  completeness = distinct injected NEOs recovered / N_injected
  false links  = gated tracks matching NO injected NEO
  purity       = recovered / (recovered + false)
One K per invocation (SLURM-array). Usage: python fp_budget_completeness.py --fpp 32 --nshard 96
"""
from __future__ import annotations
import argparse, subprocess, time, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trail_tracklets import build_tracklet_files
import synth_neo

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
BIN = HL / "heliolinc2/src"; SRC = HL / "run_neo_wide"
AUX = ["Earth1day2020s_02a.txt", "ObsCodes.txt", "heliohypo_all.txt"]
POSRMS_GATE, NIGHT_GATE = 1.0e9, 3   # production NEO gate: loose (link_refine maxrms=1e5)+crossmatch, NOT slow-object posRMS<2000
EXPT = 30.0/86400.0; CCD = 0.22                 # panel size (deg)

ap = argparse.ArgumentParser()
ap.add_argument("--fpp", type=int, required=True); ap.add_argument("--nshard", type=int, default=96)
ap.add_argument("--nneo", type=int, default=200); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--outdir", default=str(HL/"fp_budget_compl")); a = ap.parse_args()
OUT = Path(a.outdir); OUT.mkdir(exist_ok=True); rd = OUT/f"run_{a.fpp:05d}"; rd.mkdir(exist_ok=True)
for f in AUX:
    if not (rd/f).exists(): (rd/f).symlink_to(SRC/f)
EARTH = str(rd/"Earth1day2020s_02a.txt")

# LSST cadence epochs: 4 nights / 15-day window / 2 visits per night (pairs ~30 min)
base = 60858.0; nights = [base, base+3.5, base+8.0, base+14.0]
epochs = np.array([n+dv for n in nights for dv in (0.0, 0.02)])

# --- signal: injected NEOs (fixed seed -> SAME NEOs across all K); trail dets w/ true instantaneous motion ---
NEO = synth_neo.generate(epochs, n_target=a.nneo, seed=a.seed, earth_file=EARTH)
N_INJ = NEO.ObjID.nunique()

# --- background: K real-FP-like trash per panel, locally around each NEO detection ---
fp_pool = pd.read_csv(SRC/"adcnn_dets_labeled.csv"); fp_pool = fp_pool[fp_pool.objid.isna()]
cdp = np.cos(np.radians(fp_pool.dec.values))
v_rate = np.hypot((fp_pool.ra1-fp_pool.ra0).values*cdp, (fp_pool.dec1-fp_pool.dec0).values)/EXPT
v_dir = np.degrees(np.arctan2((fp_pool.dec1-fp_pool.dec0).values, (fp_pool.ra1-fp_pool.ra0).values*cdp))
rng = np.random.default_rng(a.seed + a.fpp)
bg = []
if a.fpp > 0:
    for _, r in NEO.iterrows():
        n = a.fpp
        ra = r.ra + rng.uniform(-CCD/2, CCD/2, n)/np.cos(np.radians(r.dec)); dec = r.dec + rng.uniform(-CCD/2, CCD/2, n)
        j = rng.integers(0, len(v_rate), n); om, ph = v_rate[j], np.radians(v_dir[j])
        half = 0.5*om*EXPT; cc = np.cos(np.radians(dec)); dra = half*np.cos(ph)/np.clip(cc,1e-6,None); dd = half*np.sin(ph)
        for k in range(n):
            bg.append((ra[k], dec[k], r.mjd, ra[k]-dra[k], dec[k]-dd[k], ra[k]+dra[k], dec[k]+dd[k]))
BG = pd.DataFrame(bg, columns=["ra","dec","mjd","ra0","dec0","ra1","dec1"]); BG["mag"]=22.0; BG["band"]="r"; BG["ObjID"]="FP"
dets = pd.concat([NEO[["ObjID","ra","dec","mjd","ra0","dec0","ra1","dec1","mag","band"]], BG], ignore_index=True)
MJDREF = round(float(dets.mjd.median()), 3)
print(f"fpp={a.fpp}: {N_INJ} NEOs ({len(NEO)} dets) + {len(BG)} FP = {len(dets)} dets; mjdref={MJDREF}", flush=True)

t0 = time.time()
build_tracklet_files(dets.reset_index(drop=True), EARTH, rd)
ntrk = sum(1 for ln in open(rd/"pairs.txt") if ln.startswith("T "))
grid = (rd/"heliohypo_all.txt").read_text().splitlines(); hdr, body = grid[0], grid[1:]
for f in rd.glob("hl_clusters_*.csv"): f.unlink()
procs = []
for s in range(a.nshard):
    ch = body[s::a.nshard]
    if not ch: continue
    (rd/f"grid_{s:03d}.txt").write_text(hdr+"\n"+"\n".join(ch)+"\n")
    procs.append(subprocess.Popen([str(BIN/"heliolinc"),"-dets","pairdets.csv","-pairs","pairs.txt","-mjd",str(MJDREF),
        "-obspos","Earth1day2020s_02a.txt","-heliodist",f"grid_{s:03d}.txt","-clustrad","100000","-npt","3",
        "-minobsnights","3","-mintimespan","0.5","-out",f"hl_clusters_{s:03d}.csv","-outsum",f"hl_summary_{s:03d}.csv"],
        cwd=rd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
for p in procs: p.wait()
lf = [f"hl_clusters_{s:03d}.csv hl_summary_{s:03d}.csv" for s in range(a.nshard)
      if (rd/f"hl_clusters_{s:03d}.csv").exists() and (rd/f"hl_clusters_{s:03d}.csv").stat().st_size>0]
(rd/"lflist.txt").write_text("\n".join(lf)+"\n")
subprocess.run([str(BIN/"link_refine"),"-pairdet","pairdets.csv","-lflist","lflist.txt","-maxrms","100000",
                "-outfile","lr.csv","-outrms","lr_rms.csv"], cwd=rd, capture_output=True, text=True)
dt = time.time()-t0

# crossmatch gated tracks to injected NEO truth (position+time)
def sep(r1,d1,r2,d2): return np.hypot((r1-r2)*np.cos(np.radians(d2)),d1-d2)*3600
n_rec=n_false=n_gated=0
if (rd/"lr.csv").exists() and (rd/"lr_rms.csv").stat().st_size>0:
    lr=pd.read_csv(rd/"lr.csv"); lr.columns=[c.lstrip("#") for c in lr.columns]
    rms=pd.read_csv(rd/"lr_rms.csv"); rms.columns=[c.lstrip("#") for c in rms.columns]
    good=set(rms[(rms.posRMS<POSRMS_GATE)&(rms.obsnights>=NIGHT_GATE)].clusternum)
    rec=set()
    for cl,trk in lr.groupby("clusternum"):
        if cl not in good: continue
        n_gated+=1
        hits=[]
        for _,dd in trk.iterrows():
            c=NEO[np.abs(NEO.mjd-dd.MJD)<0.02]
            if len(c):
                s=sep(dd.RA,dd.Dec,c.ra.values,c.dec.values); j=s.argmin()
                if s[j]<3.0: hits.append(c.ObjID.values[j])
        if hits and len(hits)>=0.5*len(trk):
            from collections import Counter; rec.add(Counter(hits).most_common(1)[0][0])
        else: n_false+=1
    n_rec=len(rec)
res=dict(fp_per_panel=a.fpp, n_injected=N_INJ, n_dets=len(dets), n_tracklets=ntrk, n_gated_tracks=n_gated,
         n_recovered=n_rec, completeness=round(n_rec/max(N_INJ,1),3), n_false=n_false,
         purity=round(n_rec/max(n_rec+n_false,1),3), runtime_s=round(dt,1))
pd.DataFrame([res]).to_csv(OUT/f"result_{a.fpp:05d}.csv", index=False)
print(f"fpp={a.fpp} | inj {N_INJ} | gated {n_gated} | recovered {n_rec} (compl {res['completeness']}) | "
      f"false {n_false} (purity {res['purity']}) | {dt:.0f}s", flush=True)
