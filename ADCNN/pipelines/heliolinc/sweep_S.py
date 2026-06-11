"""Sweep the ADCNN score floor S over the injected off-ecliptic master catalog to find the OPERATING
THRESHOLD S* where the same-night 2-visit false-link rate lambda(S) crosses the 3-sigma budget
1.35e-3 / pair -- MEASURED (with ~10 false links near the crossing), not bounded.

Per field we have ADCNN detections (score S, re-timed mjd, trail geometry) over the clean-FP off-ecliptic
substrate WITH inline-injected NEO-like movers (inject.csv: objID truth + target SNR). Off-ecliptic =>
zero real asteroids => any surviving 2-track that does NOT match an injected objID is a genuine FALSE link;
one that matches a single injected objID is a RECOVERY. So one pass gives both purity (lambda) and
completeness, the latter binned by injected SNR (incl. the faint SNR 2-5 movers ADCNN exists for).

For each S: chord-seed adjacent same-night pairs, apply the shipped physical_check (chord + chi2<=3.0),
classify survivors recovered/false, and report lambda, its 95% Poisson upper limit, the implied one-sided
sigma, and completeness(S). Then interpolate S* where lambda(S) = 1.35e-3.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm, chi2 as chi2dist

from ADCNN.pipelines.heliolinc.trail_state_link import chord_seed_pairs, physical_check
from ADCNN.pipelines.heliolinc.recurrence import add_recurrence
from ADCNN.pipelines.heliolinc.retime_cadence import apply_retime

BUDGET_3SIG = 1.35e-3
PC = dict(pa_tol_deg=20.0, lin_rms_arcsec=1.0, min_epochs=2, pa_tol_2v_deg=10.0, orbit_check_2v=True,
          orbit_rate_tol=0.25, max_arc_2v_min=40.0, perp_collinear_2v_arcsec=0.30,
          chi2_2v_max=3.0, chi2_sig=None)


def label_injected(d, inj, tol_px=10.0):
    """Tag each detection with the injected objID it matches (per visit,detector, nearest within tol_px)."""
    d = d.copy(); d["objID"] = None
    if inj is None or not len(inj):
        return d
    for (v, det), g in inj.groupby(["visit", "detector"]):
        sel = d[(d.visit == v) & (d.detector == det)]
        if not len(sel):
            continue
        tree = cKDTree(g[["x", "y"]].values)
        dist, idx = tree.query(sel[["x", "y"]].values, distance_upper_bound=tol_px)
        for di, dd, ii in zip(sel.index, dist, idx):
            if np.isfinite(dd):
                d.at[di, "objID"] = g.iloc[ii].objID
    return d


def field_eval(d, scores):
    """Evaluate one field across ALL score thresholds in a SINGLE pass.
    physical_check is score-INDEPENDENT geometry, so we seed pairs once at the lowest score, run the
    check once per pair, and threshold afterward by each pair's MIN member score. ~5x faster than
    re-seeding/re-checking per score. (Valid because every visit keeps dets at any score in this FP-dense
    regime, so visit-adjacency -- and thus the pair set as a function of min-score -- is monotone.)
    A pair is a RECOVERY only if BOTH members are the SAME injected objID; obj+FP or FP+FP => false link.
    Returns {S: (n_pairs_trials, n_false, recovered_objID_set)}."""
    smin = min(scores)
    ds = d[d.score >= smin].reset_index(drop=True)
    sc = ds.score.to_numpy(); oid = ds.objID.to_numpy()
    # LINKING-stage purity cuts (non-ML): the recovered movers are bright in BOTH visits while surviving FP
    # are faint marginal detections -> require the fainter member's mf_snr >= mfsnr_min_2v; + a NEO rate band.
    mfsnr_cut = PC.get("mfsnr_min_2v"); rlo = PC.get("rate_lo_2v"); rhi = PC.get("rate_hi_2v")
    mfs = ds.mf_snr.to_numpy() if "mf_snr" in ds else None
    ra = ds.ra.to_numpy(); dec = ds.dec.to_numpy(); mjd = ds.mjd.to_numpy()
    pcheck = {k: v for k, v in PC.items() if k not in ("mfsnr_min_2v", "rate_lo_2v", "rate_hi_2v")}
    checked = []   # (pair_min_score, recovered_objID or None)
    for m in chord_seed_pairs(ds, max_arc_min=PC["max_arc_2v_min"]):
        ok, _info, nep = physical_check(ds, m, **pcheck)
        if not (ok and nep == 2):
            continue
        i, j = m
        if mfsnr_cut is not None and mfs is not None and min(mfs[i], mfs[j]) < mfsnr_cut:
            continue
        if rlo is not None:
            dt = abs(mjd[j] - mjd[i]); cd = np.cos(np.radians(dec[i]))
            rate = np.hypot((ra[j] - ra[i]) * cd, dec[j] - dec[i]) / dt if dt > 0 else 0.0
            if rate < rlo or rate > rhi:
                continue
        o = oid[i] if (pd.notna(oid[i]) and oid[i] == oid[j]) else None   # same injected obj => recovery
        checked.append((float(min(sc[i], sc[j])), o))
    out = {}
    for S in scores:
        dss = ds[ds.score >= S]
        vis = sorted(dss.visit.unique())
        mj = {v: dss[dss.visit == v].mjd.median() for v in vis}
        npair = sum(1 for i in range(len(vis) - 1) if (mj[vis[i + 1]] - mj[vis[i]]) * 1440 <= PC["max_arc_2v_min"])
        nf = 0; rec = set()
        for ps, o in checked:
            if ps < S:
                continue
            if o is not None:
                rec.add(o)
            else:
                nf += 1
        out[S] = (npair, nf, rec)
    return out


def _load_field(d_dir, k, len_db_min, art_frac_max, recur_max, len_db_max=1e9):
    """Worker: load+filter+retime+recur+label one field. Returns (k, labelled_df, recoverable{objID:snr})."""
    import pandas as pd
    from pathlib import Path
    f = f"{d_dir}/adcnn_dets_masked_{k}.csv"
    d = pd.read_csv(f)
    d = d[(d.len_db >= len_db_min) & (d.len_db <= len_db_max) &
          (d.get("art_frac", 0) < art_frac_max)].reset_index(drop=True)
    rmf = f"{d_dir}/retime_{k}.csv"
    if Path(rmf).exists():
        d = apply_retime(d, pd.read_csv(rmf))
    if recur_max is not None:
        d = add_recurrence(d); d = d[d.recur < recur_max].reset_index(drop=True)
    injf = f"{d_dir}/inject_{k}.csv"
    inj = pd.read_csv(injf) if Path(injf).exists() else None
    d = label_injected(d, inj)
    # sim_orbits REUSES objID names (SNEO00000..) in every field -> make them FIELD-UNIQUE so recovered/
    # recoverable count PHYSICAL objects, not names (else completeness inflates with field count).
    m = d.objID.notna()
    d.loc[m, "objID"] = f"{k}_" + d.loc[m, "objID"].astype(str)
    recoverable = {}
    if inj is not None and len(inj):
        cnt = inj.groupby("objID").size(); snr = inj.groupby("objID").snr_target.first()
        recoverable = {f"{k}_{o}": float(snr[o]) for o in cnt[cnt >= 2].index}
    return k, d, recoverable


def _cache_path(d_dir, k, params):
    import hashlib
    tag = hashlib.md5(repr(params).encode()).hexdigest()[:10]   # keyed by scores + ALL filter params
    return f"{d_dir}/_sweepcache/{k}_{tag}.json"


def _read_cache(d_dir, k, params):
    import json, os
    cp = _cache_path(d_dir, k, params)
    if not os.path.exists(cp):
        return None
    with open(cp) as f:
        c = json.load(f)
    res = {float(s): (v[0], v[1], set(v[2])) for s, v in c["res"].items()}
    return res, {o: float(sn) for o, sn in c["rec"].items()}


def _eval_field_worker(args):
    """Per-field eval with on-disk caching (keyed by scores+filters) -> resumable; a giant slow field
    can't block aggregation of the rest (use --aggregate-only)."""
    import json, os
    d_dir, k, scores, len_db_min, len_db_max, art_frac_max, recur_max = args
    params = (tuple(scores), len_db_min, len_db_max, art_frac_max, recur_max,
              PC.get("mfsnr_min_2v"), PC.get("rate_lo_2v"), PC.get("rate_hi_2v"),
              PC.get("max_arc_2v_min"), PC.get("chi2_2v_max"))
    cached = _read_cache(d_dir, k, params)
    if cached is not None:
        return (k, *cached)
    _, d, recoverable = _load_field(d_dir, k, len_db_min, art_frac_max, recur_max, len_db_max)
    res = field_eval(d, scores)
    os.makedirs(f"{d_dir}/_sweepcache", exist_ok=True)
    cp = _cache_path(d_dir, k, params)
    with open(cp + ".tmp", "w") as f:
        json.dump({"res": {f"{s}": [v[0], v[1], sorted(v[2])] for s, v in res.items()}, "rec": recoverable}, f)
    os.replace(cp + ".tmp", cp)   # atomic
    return k, res, recoverable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run dir with per-field adcnn_dets_masked_*.csv + inject_*.csv + retime_*.csv")
    ap.add_argument("--scores", nargs="+", type=float,
                    default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--len-db-max", type=float, default=1e9, help="upper len_db band (50 = the tuned-params band; default off = pin-equivalent)")
    ap.add_argument("--art-frac-max", type=float, default=0.3)
    ap.add_argument("--recur-max", type=int, default=2)
    ap.add_argument("--max-arc", type=float, default=None, help="override Δt link window (min); must exceed the pair gap")
    ap.add_argument("--chi2-max", type=float, default=3.0, help="orbit-fit chi2 gate; loosen (with mfsnr carrying purity) to recover noisy true movers")
    ap.add_argument("--mfsnr-min", type=float, default=None, help="LINKING purity cut: require fainter member mf_snr >= this")
    ap.add_argument("--rate-lo", type=float, default=None, help="NEO rate band low (deg/day); pair with --rate-hi")
    ap.add_argument("--rate-hi", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=32, help="parallel fields")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip compute; build the curve from existing _sweepcache (partial results anytime)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out or a.dir)
    scores = sorted(a.scores)
    if a.max_arc is not None:
        PC["max_arc_2v_min"] = a.max_arc   # forked workers inherit this module global
    PC["mfsnr_min_2v"] = a.mfsnr_min
    PC["rate_lo_2v"] = a.rate_lo; PC["rate_hi_2v"] = a.rate_hi
    PC["chi2_2v_max"] = a.chi2_max

    ks = [f.split("adcnn_dets_masked_")[1].split(".csv")[0]
          for f in sorted(glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv"))]
    print(f"[sweep] {len(ks)} fields, {a.workers} workers, scores {scores}", flush=True)

    per_field = {}            # k -> {S: (NP, nf, recset)}
    all_recoverable = {}
    params = (tuple(scores), a.len_db_min, a.len_db_max, a.art_frac_max, a.recur_max,
              a.mfsnr_min, a.rate_lo, a.rate_hi, PC["max_arc_2v_min"], PC["chi2_2v_max"])
    if a.aggregate_only:
        for k in ks:
            c = _read_cache(a.dir, k, params)
            if c is not None:
                per_field[k] = c[0]; all_recoverable.update(c[1])
        ks = [k for k in ks if k in per_field]
        print(f"[sweep] AGGREGATE-ONLY: {len(ks)} cached fields", flush=True)
    else:
        tasks = [(a.dir, k, scores, a.len_db_min, a.len_db_max, a.art_frac_max, a.recur_max) for k in ks]
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(_eval_field_worker, t) for t in tasks]):
                k, res_k, recoverable = fut.result()
                per_field[k] = res_k; all_recoverable.update(recoverable)
                print(f"[sweep] field {k} done ({len(recoverable)} recoverable injected)", flush=True)

    snr_bins = [(2, 5), (5, 10), (10, 1e9)]
    rows = []
    for S in scores:
        NP = nf = 0; rec = set()
        for k in ks:
            p, x, r = per_field[k][S]
            NP += p; nf += x; rec |= r
        lam = nf / max(NP, 1)
        ul = 0.5 * chi2dist.ppf(0.95, 2 * (nf + 1)) / max(NP, 1)      # exact Poisson 95% upper limit
        sig = float(norm.isf(ul)) if ul > 0 else np.inf
        comp = len(rec) / max(len(all_recoverable), 1)
        row = dict(score=S, pairs=NP, false=nf, lambda_pair=lam, lambda_ul95=ul, sigma=sig,
                   recovered=len(rec), recoverable=len(all_recoverable), completeness=comp)
        for lo, hi in snr_bins:
            tot = sum(1 for s in all_recoverable.values() if lo <= s < hi)
            got = sum(1 for o in rec if lo <= all_recoverable.get(o, -1) < hi)
            row[f"comp_snr{int(lo)}_{int(hi) if hi < 1e8 else 'inf'}"] = got / tot if tot else np.nan
        rows.append(row)
        print(f"[sweep] S={S:.2f}: {NP} pairs, {nf} false, lam={lam:.2e}, UL={ul:.2e} ({sig:.2f}sig), "
              f"comp={comp:.2f} ({len(rec)}/{len(all_recoverable)})", flush=True)

    res = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "lambda_vs_S.csv", index=False)

    # interpolate S* where lambda(S) crosses the 3-sigma budget (lambda decreasing in S)
    s_star = np.nan
    g = res.sort_values("score")
    lo = g[g.lambda_pair > BUDGET_3SIG]; hi = g[g.lambda_pair <= BUDGET_3SIG]
    if len(lo) and len(hi):
        s0, l0 = lo.iloc[-1][["score", "lambda_pair"]]
        s1, l1 = hi.iloc[0][["score", "lambda_pair"]]
        if l0 != l1:
            s_star = float(s0 + (s1 - s0) * (np.log(BUDGET_3SIG) - np.log(l0)) / (np.log(l1) - np.log(l0)))
    summary = dict(s_star=s_star, budget=BUDGET_3SIG, total_pairs=int(res.pairs.max()),
                   total_recoverable=len(all_recoverable))
    pd.DataFrame([summary]).to_csv(outdir / "s_star.csv", index=False)
    print(f"\n[sweep] === S* (lambda=1.35e-3) = {s_star:.3f} ===  total recoverable injected={len(all_recoverable)}")
    print(res.to_string(index=False))
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
