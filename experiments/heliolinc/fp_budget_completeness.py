"""Completeness (and purity) of fast-NEO linking vs FP-per-panel. Inject synthetic >=1 deg/day NEOs
(real 2-body orbits, synth_neo) on an LSST cadence (4 nights, 2 visits, 15-day window), add a real-FP
background at K detections PER PANEL locally around each NEO detection (the FP that can actually
contaminate a NEO's cluster; bulk FP elsewhere give ~0 false links per the trash-MC), run the full
trail-tracklet -> sharded heliolinc -> link_refine -> gate(loose maxrms=1e5,>=3nt) pipeline, and
crossmatch the gated tracks to the injected NEO truth.
  completeness = distinct injected NEOs recovered / N_injected
  false links  = gated tracks matching NO injected NEO
  purity       = recovered / (recovered + false)

Splits across NNODE nodes (see fp_budget_mn): --stage prep (inject + build tracklets once + save NEO
truth), --stage shard --node-idx K --nnode N (each node links its grid slice w/ streaming progress),
--stage finalize (link_refine + crossmatch). Default --stage all = original single-node run. Usage:
    python fp_budget_completeness.py --fpp 2800 --nneo 50 --stage prep
    python fp_budget_completeness.py --fpp 2800 --stage shard --node-idx 0 --nnode 3 --ncores 120
    python fp_budget_completeness.py --fpp 2800 --stage finalize
"""
from __future__ import annotations
import argparse, os, time, sys, shutil
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trail_tracklets import build_tracklet_files
import fp_budget_mn as mn
import synth_neo

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
BIN = HL / "heliolinc2/src"; SRC = HL / "NEO_large"
AUX = ["Earth1day2020s_02a.txt", "ObsCodes.txt", "heliohypo_all.txt"]
POSRMS_GATE, NIGHT_GATE = 1.0e9, 3   # production NEO gate: loose (link_refine maxrms=1e5)+crossmatch
MAXRMS = 100000.0
EXPT = 30.0 / 86400.0; CCD = 0.22                 # panel size (deg)


def _ncores_default(n):
    if n:
        return n
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8


def stage_prep(a, rd):
    earth = str(rd / "Earth1day2020s_02a.txt")
    base = 60858.0; nights = [base, base + 3.5, base + 8.0, base + 14.0]
    epochs = np.array([n + dv for n in nights for dv in (0.0, 0.02)])
    NEO = synth_neo.generate(epochs, n_target=a.nneo, seed=a.seed, earth_file=earth)
    N_INJ = NEO.ObjID.nunique()

    fp_pool = pd.read_csv(SRC / "adcnn_dets_labeled.csv"); fp_pool = fp_pool[fp_pool.objid.isna()]
    cdp = np.cos(np.radians(fp_pool.dec.values))
    v_rate = np.hypot((fp_pool.ra1 - fp_pool.ra0).values * cdp, (fp_pool.dec1 - fp_pool.dec0).values) / EXPT
    v_dir = np.degrees(np.arctan2((fp_pool.dec1 - fp_pool.dec0).values, (fp_pool.ra1 - fp_pool.ra0).values * cdp))
    rng = np.random.default_rng(a.seed + a.fpp)
    bg = []
    if a.fpp > 0:
        for _, r in NEO.iterrows():
            n = a.fpp
            ra = r.ra + rng.uniform(-CCD / 2, CCD / 2, n) / np.cos(np.radians(r.dec)); dec = r.dec + rng.uniform(-CCD / 2, CCD / 2, n)
            j = rng.integers(0, len(v_rate), n); om, ph = v_rate[j], np.radians(v_dir[j])
            half = 0.5 * om * EXPT; cc = np.cos(np.radians(dec)); dra = half * np.cos(ph) / np.clip(cc, 1e-6, None); dd = half * np.sin(ph)
            for k in range(n):
                bg.append((ra[k], dec[k], r.mjd, ra[k] - dra[k], dec[k] - dd[k], ra[k] + dra[k], dec[k] + dd[k]))
    BG = pd.DataFrame(bg, columns=["ra", "dec", "mjd", "ra0", "dec0", "ra1", "dec1"])
    BG["mag"] = 22.0; BG["band"] = "r"; BG["ObjID"] = "FP"
    dets = pd.concat([NEO[["ObjID", "ra", "dec", "mjd", "ra0", "dec0", "ra1", "dec1", "mag", "band"]], BG], ignore_index=True)
    mjdref = round(float(dets.mjd.median()), 3)
    print(f"fpp={a.fpp}: {N_INJ} NEOs ({len(NEO)} dets) + {len(BG)} FP = {len(dets)} dets; mjdref={mjdref}", flush=True)
    t0 = time.time()
    build_tracklet_files(dets.reset_index(drop=True), earth, rd)
    ntrk = sum(1 for ln in open(rd / "pairs.txt") if ln.startswith("T "))
    NEO[["ObjID", "ra", "dec", "mjd"]].to_csv(rd / "neo_truth.csv", index=False)   # for finalize crossmatch
    shutil.rmtree(rd / "clusters_mn", ignore_errors=True)
    mn.write_meta(rd, mjdref=mjdref, n_injected=int(N_INJ), n_dets=int(len(dets)),
                  n_tracklets=int(ntrk), fpp=a.fpp, prep_s=round(time.time() - t0, 1))
    print(f"PREP DONE fpp={a.fpp}: {ntrk} tracklets, {time.time()-t0:.0f}s -> {rd}", flush=True)


def _sep(r1, d1, r2, d2):
    return np.hypot((r1 - r2) * np.cos(np.radians(d2)), d1 - d2) * 3600


def stage_finalize(a, rd, outdir):
    meta = mn.read_meta(rd)
    NEO = pd.read_csv(rd / "neo_truth.csv")
    t0 = time.time()
    mn.finalize_link_refine(rd, BIN, maxrms=MAXRMS)
    n_rec = n_false = n_gated = 0
    if (rd / "lr.csv").exists() and (rd / "lr_rms.csv").stat().st_size > 0:
        lr = pd.read_csv(rd / "lr.csv"); lr.columns = [c.lstrip("#") for c in lr.columns]
        rms = pd.read_csv(rd / "lr_rms.csv"); rms.columns = [c.lstrip("#") for c in rms.columns]
        good = set(rms[(rms.posRMS < POSRMS_GATE) & (rms.obsnights >= NIGHT_GATE)].clusternum)
        rec = set()
        for cl, trk in lr.groupby("clusternum"):
            if cl not in good:
                continue
            n_gated += 1
            hits = []
            for _, dd in trk.iterrows():
                c = NEO[np.abs(NEO.mjd - dd.MJD) < 0.02]
                if len(c):
                    s = _sep(dd.RA, dd.Dec, c.ra.values, c.dec.values); j = s.argmin()
                    if s[j] < 3.0: hits.append(c.ObjID.values[j])
            if hits and len(hits) >= 0.5 * len(trk):
                rec.add(Counter(hits).most_common(1)[0][0])
            else:
                n_false += 1
        n_rec = len(rec)
    N_INJ = meta["n_injected"]
    res = dict(fp_per_panel=meta["fpp"], n_injected=N_INJ, n_dets=meta["n_dets"],
               n_tracklets=meta["n_tracklets"], n_gated_tracks=n_gated, n_recovered=n_rec,
               completeness=round(n_rec / max(N_INJ, 1), 3), n_false=n_false,
               purity=round(n_rec / max(n_rec + n_false, 1), 3),
               runtime_s=round(meta.get("prep_s", 0) + time.time() - t0, 1))
    pd.DataFrame([res]).to_csv(outdir / f"result_{meta['fpp']:05d}.csv", index=False)
    print(f"fpp={meta['fpp']} | inj {N_INJ} | gated {n_gated} | recovered {n_rec} (compl {res['completeness']}) | "
          f"false {n_false} (purity {res['purity']}) -> result_{meta['fpp']:05d}.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpp", type=int, required=True)
    ap.add_argument("--nneo", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nshard", type=int, default=96, help="grid shards (single-node 'all' mode)")
    ap.add_argument("--ncores", type=int, default=0, help="shard-stage local cores (0=auto)")
    ap.add_argument("--stage", choices=["all", "prep", "shard", "finalize"], default="all")
    ap.add_argument("--node-idx", type=int, default=0)
    ap.add_argument("--nnode", type=int, default=1)
    ap.add_argument("--outdir", default=str(HL / "fp_budget_compl"))
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
                           tag=f"compl{a.fpp}")
    if a.stage in ("all", "finalize"):
        stage_finalize(a, rd, outdir)


if __name__ == "__main__":
    main()
