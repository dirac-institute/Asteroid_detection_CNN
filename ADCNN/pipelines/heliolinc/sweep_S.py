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
    checked = []   # (pair_min_score, recovered_objID or None)
    for m in chord_seed_pairs(ds, max_arc_min=PC["max_arc_2v_min"]):
        ok, _info, nep = physical_check(ds, m, **PC)
        if not (ok and nep == 2):
            continue
        i, j = m
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


def _load_field(d_dir, k, len_db_min, art_frac_max, recur_max):
    """Worker: load+filter+retime+recur+label one field. Returns (k, labelled_df, recoverable{objID:snr})."""
    import pandas as pd
    from pathlib import Path
    f = f"{d_dir}/adcnn_dets_masked_{k}.csv"
    d = pd.read_csv(f)
    d = d[(d.len_db >= len_db_min) & (d.get("art_frac", 0) < art_frac_max)].reset_index(drop=True)
    rmf = f"{d_dir}/retime_{k}.csv"
    if Path(rmf).exists():
        d = apply_retime(d, pd.read_csv(rmf))
    if recur_max is not None:
        d = add_recurrence(d); d = d[d.recur < recur_max].reset_index(drop=True)
    injf = f"{d_dir}/inject_{k}.csv"
    inj = pd.read_csv(injf) if Path(injf).exists() else None
    d = label_injected(d, inj)
    recoverable = {}
    if inj is not None and len(inj):
        cnt = inj.groupby("objID").size(); snr = inj.groupby("objID").snr_target.first()
        recoverable = {o: float(snr[o]) for o in cnt[cnt >= 2].index}
    return k, d, recoverable


def _eval_field_worker(args):
    d_dir, k, scores, len_db_min, art_frac_max, recur_max = args
    _, d, recoverable = _load_field(d_dir, k, len_db_min, art_frac_max, recur_max)
    return k, field_eval(d, scores), recoverable


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run dir with per-field adcnn_dets_masked_*.csv + inject_*.csv + retime_*.csv")
    ap.add_argument("--scores", nargs="+", type=float,
                    default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--art-frac-max", type=float, default=0.3)
    ap.add_argument("--recur-max", type=int, default=2)
    ap.add_argument("--workers", type=int, default=32, help="parallel fields")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out or a.dir)
    scores = sorted(a.scores)

    ks = [f.split("adcnn_dets_masked_")[1].split(".csv")[0]
          for f in sorted(glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv"))]
    print(f"[sweep] {len(ks)} fields, {a.workers} workers, scores {scores}", flush=True)
    tasks = [(a.dir, k, scores, a.len_db_min, a.art_frac_max, a.recur_max) for k in ks]

    # per-field eval in parallel; aggregate per score
    per_field = {}            # k -> {S: (NP, nf, recset)}
    all_recoverable = {}
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
