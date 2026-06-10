#!/usr/bin/env python3
"""EXACT per-pair table at LOW score floors (0.60+) -- no FP subsampling.

The pure-python physical_check loop cannot enumerate the 40-144M chord-seed pairs/field at floor 0.60.
This module reproduces the measure_nomfsnr evidence chain EXACTLY but vectorizes the cheap geometric
pre-gates (the same dpa_tm/dpa_tt/dspeed cuts pair_chi2 applies before its orbit solve) with numpy, so the
expensive python pair_chi2 (astropy/Lambert orbit fit) runs only on the ~1e-3 fraction of survivors.

BIT-CONSISTENCY GATE: run with --validate against a field's uncapped v2 cache (floor 0.80) -- the surviving
pair set and FP counts must MATCH the python chain before any low-floor number is trusted.

Pipeline per field (mirrors measure_nomfsnr.eval_field semantics exactly):
  _load_field (len/art/recur filters + truth labels)  ->  score floor smin
  visit pairs: all same-night pairs with 0 < dt <= 40 min, capped to the 200 nearest-in-time
  KD annulus per visit pair: chord radius <= rate_max*dt (=10 deg/day), exact sep >= rate_min*dt (=0.3)
  VECTORIZED pre-gates (identical formulas to pair_chi2):
      dpa_tm = max member |trail PA - motion PA| (mod 180)   > 20  -> reject
      dpa_tt = |trail PA A - trail PA B| (mod 180)           > 15  -> reject
      dspeed = max member |trail speed - chord speed|/max(chord,0.3) > 0.6 -> reject
  survivors -> python physical_check (full: epoch/arc/chi2<=5 incl. bound-orbit fit)
  rows: (min_score, min_mfsnr, rate, label, n_fp, obj, max_score, min_len, chi2, dpa_tm, dspeed, perp)
Cache: {dir}/_nomfsnr_cache/{k}_smin{smin}_v3exact.json (same row schema as v2 -> same reducers work).
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd

import ADCNN.pipelines.heliolinc.sweep_S as sw
from ADCNN.pipelines.heliolinc.trail_state_link import (radec_to_unit, _chord_radius, physical_check,
                                                        pair_chi2)

PCHECK = dict(pa_tol_deg=20.0, lin_rms_arcsec=1.0, min_epochs=2, pa_tol_2v_deg=10.0, orbit_check_2v=True,
              orbit_rate_tol=0.5, max_arc_2v_min=40.0, chi2_2v_max=5.0)
SOLARDAY = 86400.0


def _cache_path(d_dir, k, smin):
    return f"{d_dir}/_nomfsnr_cache/{k}_smin{smin}_v3exact.json"


def field_pairs_exact(ds, exptime_s=30.0):
    """Vectorized seed enumeration + pre-gates; returns candidate (i,j) surviving the cheap gates."""
    from scipy.spatial import cKDTree
    mjd = ds.mjd.to_numpy(); ra = ds.ra.to_numpy(); dec = ds.dec.to_numpy(); vis = ds.visit.to_numpy()
    # member trail velocities (deg/day on-sky), identical to pair_chi2's tv()
    dt_exp = exptime_s / SOLARDAY
    cosd = np.cos(np.radians(dec))
    tvx = (ds.ra1.to_numpy() - ds.ra0.to_numpy()) * cosd / dt_exp
    tvy = (ds.dec1.to_numpy() - ds.dec0.to_numpy()) / dt_exp
    tpa = np.degrees(np.arctan2(tvy, tvx)) % 180.0
    tsp = np.hypot(tvx, tvy)
    uv = sorted(set(vis.tolist()))
    vmjd = {v: float(np.median(mjd[vis == v])) for v in uv}
    idx_by = {v: np.where(vis == v)[0] for v in uv}
    vpairs = [(vmjd[b] - vmjd[a_], a_, b) for ii, a_ in enumerate(uv) for b in uv[ii + 1:]
              if 0 < vmjd[b] - vmjd[a_] <= PCHECK["max_arc_2v_min"] / 1440.0]
    if len(vpairs) > 200:
        vpairs.sort(key=lambda t: t[0]); vpairs = vpairs[:200]
    out_i = []; out_j = []
    for dtv, a_, b in vpairs:
        ia, ib = idx_by[a_], idx_by[b]
        if not len(ia) or not len(ib):
            continue
        tree = cKDTree(radec_to_unit(ra[ib], dec[ib]))
        qmax = _chord_radius(10.0 * dtv)                       # rate_max=10 deg/day annulus
        nb = tree.query_ball_point(radec_to_unit(ra[ia], dec[ia]), qmax)
        cnt = np.fromiter((len(x) for x in nb), int, len(nb))
        if cnt.sum() == 0:
            continue
        I = np.repeat(ia, cnt)
        J = ib[np.concatenate([np.asarray(x, int) for x in nb if len(x)])]
        # exact angular sep + rate annulus floor (rate_min=0.3), identical to chord_seed_pairs
        cd = np.cos(np.radians(dec[I]))
        dra = (ra[J] - ra[I] + 180.0) % 360.0 - 180.0
        sep = np.hypot(dra * cd, dec[J] - dec[I])
        keep = sep >= 0.3 * dtv
        I, J, sep, dra, cd = I[keep], J[keep], sep[keep], dra[keep], cd[keep]
        if not len(I):
            continue
        # chord motion (deg/day) -- pair_chi2 computes mdt from member mjds (== dtv to visit median; use
        # member mjds exactly)
        mdt = mjd[J] - mjd[I]
        mx_ = dra * cd / mdt
        my_ = (dec[J] - dec[I]) / mdt
        mpa = np.degrees(np.arctan2(my_, mx_)) % 180.0
        msp = np.hypot(mx_, my_)
        d1 = np.abs(((tpa[I] - mpa + 90) % 180) - 90)
        d2 = np.abs(((tpa[J] - mpa + 90) % 180) - 90)
        dpa_tm = np.maximum(d1, d2)
        dpa_tt = np.abs(((tpa[I] - tpa[J] + 90) % 180) - 90)
        dspeed = np.maximum(np.abs(tsp[I] - msp), np.abs(tsp[J] - msp)) / np.maximum(msp, 0.3)
        g = (dpa_tm <= 20.0) & (dpa_tt <= 15.0) & (dspeed <= 0.6)
        out_i.append(I[g]); out_j.append(J[g])
    if not out_i:
        return np.empty(0, int), np.empty(0, int), 0
    n_seed = int(sum(len(x) for x in out_i))   # survivors only; raw seed count not retained (vectorized)
    return np.concatenate(out_i), np.concatenate(out_j), n_seed


def eval_field_exact(d_dir, k, smin):
    _, d, recoverable = sw._load_field(d_dir, k, 6.0, 0.3, 2, 1e9)
    ds = d[d.score >= smin].reset_index(drop=True)
    if not len(ds):
        return [], recoverable, 0
    I, J, n_surv = field_pairs_exact(ds)
    sc = ds.score.to_numpy(); oid = ds.objID.to_numpy()
    mfs = ds.mf_snr.to_numpy() if "mf_snr" in ds else np.full(len(ds), np.nan)
    lens = ds.len_db.to_numpy() if "len_db" in ds else np.full(len(ds), np.nan)
    ra = ds.ra.to_numpy(); dec = ds.dec.to_numpy(); mjd = ds.mjd.to_numpy()
    rows = []
    for i, j in zip(I, J):
        ok, _info, nep = physical_check(ds, [int(i), int(j)], **PCHECK)
        if not (ok and nep == 2):
            continue
        cd = np.cos(np.radians(dec[i])); dt = abs(mjd[j] - mjd[i])
        rate = np.hypot((ra[j] - ra[i]) * cd, dec[j] - dec[i]) / dt if dt > 0 else 0.0
        same = pd.notna(oid[i]) and oid[i] == oid[j]
        n_fp = int(pd.isna(oid[i])) + int(pd.isna(oid[j]))
        c2, ci = pair_chi2(ds.iloc[[int(i), int(j)]], 30.0)
        rows.append((float(min(sc[i], sc[j])), float(min(mfs[i], mfs[j])), float(rate),
                     "tp" if same else "fp", n_fp, (oid[i] if same else ""),
                     float(max(sc[i], sc[j])), float(min(lens[i], lens[j])), float(c2),
                     float(ci.get("dpa_tm", np.nan)), float(ci.get("dspeed", np.nan)),
                     float(ci.get("perp", np.nan))))
    return rows, recoverable, n_surv


def _worker(args):
    d_dir, k, smin = args
    cp = _cache_path(d_dir, k, smin)
    if os.path.exists(cp):
        return k, "cached"
    try:
        rows, recoverable, n_surv = eval_field_exact(d_dir, k, smin)
        json.dump({"rows": rows, "rec": recoverable, "n_seed": n_surv, "n_capped": 0, "fp_f": 1.0},
                  open(cp + ".tmp", "w"))
        os.replace(cp + ".tmp", cp)
        return k, f"done surv={n_surv} pass={len(rows)}"
    except Exception as e:
        return k, f"ERR {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="run_lambda")
    ap.add_argument("--smin", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--validate", action="store_true",
                    help="run at smin=0.8 and diff FP/TP counts per field against the v2 cache")
    a = ap.parse_args()
    ks = sorted({os.path.basename(f).split("adcnn_dets_masked_")[1].rsplit(".csv", 1)[0]
                 for f in glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv")})
    if a.validate:
        a.smin = 0.8
        for k in ks[:4]:
            rows, _rec, _ = eval_field_exact(a.dir, k, 0.8)
            tp = sum(1 for r in rows if r[3] == "tp"); fp = sum(1 for r in rows if r[3] == "fp")
            v2 = json.load(open(f"{a.dir}/_nomfsnr_cache/{k}_smin0.8_v2.json"))
            tp2 = sum(1 for r in v2["rows"] if r[3] == "tp"); fp2 = sum(1 for r in v2["rows"] if r[3] == "fp")
            tag = "OK" if (tp == tp2 and fp == fp2) else "MISMATCH"
            print(f"[validate] field {k}: vec tp={tp} fp={fp} | python tp={tp2} fp={fp2} -> {tag}", flush=True)
        return
    os.makedirs(f"{a.dir}/_nomfsnr_cache", exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(_worker, (a.dir, k, a.smin)) for k in ks]):
            k, msg = fut.result()
            print(f"[field {k}] {msg}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
