"""Trash-only Monte Carlo for the HelioLinC FP budget -- ONE density per invocation. Generates
synthetic trash at K FP PER PANEL (panel = one detector diffim, the ADCNN unit): for each real
(visit,detector) panel, K detections uniform in that panel's sky footprint, at the panel's MJD,
with trail velocity (rate+direction) drawn from the REAL ADCNN-FP velocity distribution. Builds
trail-tracklets (one >=6px streak = one tracklet), runs the REAL linker (heliolinc -> link_refine);
every surviving track is a FALSE link (pure trash).

GATE (corrected): the operational fast-NEO gate is LOOSE -- link_refine maxrms=1e5 + >=3 nights
(real fast NEOs have posRMS ~9k-54k km, so the old posRMS<2000 cut rejected them too; see the
fp-budget-derivation memo "KEY GATE CORRECTION"). The reported false-link count `n_false` is now
link_refine survivors with >=3 nights; `n_false_strict2000` keeps the old over-tight cut for
reference. Same definition as the completeness family -> apples-to-apples.

Splits across NNODE nodes (see fp_budget_mn): --stage prep (build tracklets once), --stage shard
--node-idx K --nnode N (each node links its grid slice with streaming progress), --stage finalize
(link_refine + count). Default --stage all = original single-node run. Usage:
    python fp_budget_mc.py --fpp 2800 --stage prep
    python fp_budget_mc.py --fpp 2800 --stage shard --node-idx 0 --nnode 3 --ncores 120
    python fp_budget_mc.py --fpp 2800 --stage finalize
"""
from __future__ import annotations
import argparse, os, time, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trail_tracklets import build_tracklet_files
import fp_budget_mn as mn

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
BIN = HL / "heliolinc2/src"
SRC = HL / "run_neo_wide"
AUX = ["Earth1day2020s_02a.txt", "ObsCodes.txt", "heliohypo_all.txt"]
NIGHT_GATE = 3
MAXRMS = 100000.0                           # link_refine loose gate (production fast-NEO value)
EXPT = 30.0 / 86400.0                        # exposure (day)


def _ncores_default(n):
    if n:
        return n
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8


def stage_prep(a, rd):
    det = pd.read_csv(SRC / "adcnn_dets_labeled.csv")
    fp = det[det.objid.isna()].copy()
    fp["panel"] = fp.visit.astype(str) + "_" + fp.detector.astype(str)
    pan = fp.groupby("panel").agg(ra_lo=("ra", "min"), ra_hi=("ra", "max"), dec_lo=("dec", "min"),
                                  dec_hi=("dec", "max"), mjd=("mjd", "first"))
    NPAN = len(pan)
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
        half = 0.5 * om * EXPT
        cdp = np.cos(np.radians(dec))
        dra = half * np.cos(ph) / np.clip(cdp, 1e-6, None); ddec = half * np.sin(ph)
        for k in range(n):
            rows.append((ra[k], dec[k], r.mjd, ra[k] - dra[k], dec[k] - ddec[k], ra[k] + dra[k], dec[k] + ddec[k]))
    trash = pd.DataFrame(rows, columns=["ra", "dec", "mjd", "ra0", "dec0", "ra1", "dec1"])
    trash["mag"] = 22.0; trash["band"] = "r"
    mjdref = round(float(det.mjd.median()), 3)
    print(f"fpp={a.fpp}: {len(trash)} synthetic trash over {NPAN} panels (mjdref={mjdref})", flush=True)
    t0 = time.time()
    build_tracklet_files(trash, str(rd / "Earth1day2020s_02a.txt"), rd)
    ntrk = sum(1 for ln in open(rd / "pairs.txt") if ln.startswith("T "))
    shutil.rmtree(rd / "clusters_mn", ignore_errors=True)   # fresh shard dir for this prep
    mn.write_meta(rd, mjdref=mjdref, total_fp=int(len(trash)), n_tracklets=int(ntrk),
                  fpp=a.fpp, prep_s=round(time.time() - t0, 1))
    print(f"PREP DONE fpp={a.fpp}: {ntrk} tracklets, {time.time()-t0:.0f}s -> {rd}", flush=True)


def stage_finalize(a, rd, outdir):
    meta = mn.read_meta(rd)
    t0 = time.time()
    mn.finalize_link_refine(rd, BIN, maxrms=MAXRMS)
    n_raw = n_loose = n_strict = 0
    lrr = rd / "lr_rms.csv"
    if lrr.exists() and lrr.stat().st_size > 0:
        rms = pd.read_csv(lrr); rms.columns = [c.lstrip("#") for c in rms.columns]
        n_raw = len(rms)
        n_loose = int((rms.obsnights >= NIGHT_GATE).sum())                            # CORRECTED gate
        n_strict = int(((rms.posRMS < 2000.0) & (rms.obsnights >= NIGHT_GATE)).sum())  # old over-tight
    rec = dict(fp_per_panel=meta["fpp"], total_fp=meta["total_fp"], n_tracklets=meta["n_tracklets"],
               n_clusters_raw=n_raw, n_false=n_loose, n_false_strict2000=n_strict,
               runtime_s=round(meta.get("prep_s", 0) + time.time() - t0, 1))
    pd.DataFrame([rec]).to_csv(outdir / f"result_{meta['fpp']:05d}.csv", index=False)
    print(f"fpp={meta['fpp']} | total {meta['total_fp']} | tracklets {meta['n_tracklets']} | "
          f"raw {n_raw} | FALSE links (loose maxrms1e5,>=3nt) {n_loose} | "
          f"[old posRMS<2000: {n_strict}] -> result_{meta['fpp']:05d}.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpp", type=int, required=True, help="FP PER PANEL (detector diffim) to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nshard", type=int, default=96, help="grid shards (single-node 'all' mode)")
    ap.add_argument("--ncores", type=int, default=0, help="shard-stage local cores (0=auto)")
    ap.add_argument("--stage", choices=["all", "prep", "shard", "finalize"], default="all")
    ap.add_argument("--node-idx", type=int, default=0)
    ap.add_argument("--nnode", type=int, default=1)
    ap.add_argument("--outdir", default=str(HL / "fp_budget_mc"))
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(exist_ok=True)
    rd = outdir / f"run_{a.fpp:05d}"; rd.mkdir(exist_ok=True)
    for f in AUX:
        if not (rd / f).exists(): (rd / f).symlink_to(SRC / f)

    if a.stage in ("all", "prep"):
        stage_prep(a, rd)
    if a.stage in ("all", "shard"):
        meta = mn.read_meta(rd)
        ncores = _ncores_default(a.ncores if a.stage == "shard" else a.nshard)
        mn.run_grid_shards(rd, rd / "heliohypo_all.txt", meta["mjdref"], node_idx=a.node_idx,
                           nnode=a.nnode, ncores=ncores, bin_dir=BIN,
                           clustrad=100000.0, npt=3, minnights=NIGHT_GATE, mintimespan=0.5,
                           tag=f"mc{a.fpp}")
    if a.stage in ("all", "finalize"):
        stage_finalize(a, rd, outdir)


if __name__ == "__main__":
    main()
