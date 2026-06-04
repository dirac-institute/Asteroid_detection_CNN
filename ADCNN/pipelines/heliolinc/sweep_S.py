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


def field_pairs(d, S):
    """Return (n_pairs_trials, n_false, recovered_objIDs) at score floor S for one field's labelled dets."""
    ds = d[d.score >= S].reset_index(drop=True)
    vis = sorted(ds.visit.unique())
    mj = {v: ds[ds.visit == v].mjd.median() for v in vis}
    npair = sum(1 for i in range(len(vis) - 1) if (mj[vis[i + 1]] - mj[vis[i]]) * 1440 <= PC["max_arc_2v_min"])
    n_false = 0; rec = set()
    for m in chord_seed_pairs(ds, max_arc_min=PC["max_arc_2v_min"]):
        ok, _info, nep = physical_check(ds, m, **PC)
        if not (ok and nep == 2):
            continue
        oids = set(ds.loc[list(m), "objID"].dropna().unique())
        if len(oids) == 1:                 # both endpoints = same injected object -> recovery
            rec.add(next(iter(oids)))
        else:                              # unmatched / cross-object -> false link
            n_false += 1
    return npair, n_false, rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run dir with per-field adcnn_dets_masked_*.csv + inject_*.csv + retime_*.csv")
    ap.add_argument("--scores", nargs="+", type=float,
                    default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--art-frac-max", type=float, default=0.3)
    ap.add_argument("--recur-max", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out or a.dir)

    fields = []
    for f in sorted(glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv")):
        k = f.split("adcnn_dets_masked_")[1].split(".csv")[0]
        d = pd.read_csv(f)
        d = d[(d.len_db >= a.len_db_min) & (d.get("art_frac", 0) < a.art_frac_max)].reset_index(drop=True)
        rmf = f"{a.dir}/retime_{k}.csv"
        if Path(rmf).exists():
            d = apply_retime(d, pd.read_csv(rmf))
        if a.recur_max is not None:
            d = add_recurrence(d); d = d[d.recur < a.recur_max].reset_index(drop=True)
        injf = f"{a.dir}/inject_{k}.csv"
        inj = pd.read_csv(injf) if Path(injf).exists() else None
        d = label_injected(d, inj)
        # recoverable denominator: injected objects with >=2 sightings landing on panels (cadence+footprint)
        recoverable = {}
        if inj is not None and len(inj):
            cnt = inj.groupby("objID").size()
            snr = inj.groupby("objID").snr_target.first()
            for oid in cnt[cnt >= 2].index:
                recoverable[oid] = float(snr[oid])
        fields.append(dict(k=k, d=d, recoverable=recoverable))
        print(f"[sweep] field {k}: {len(d)} dets, {len(recoverable)} recoverable injected (>=2 sightings)", flush=True)

    all_recoverable = {oid: s for fl in fields for oid, s in fl["recoverable"].items()}
    snr_bins = [(2, 5), (5, 10), (10, 1e9)]
    rows = []
    for S in a.scores:
        NP = nf = 0; rec = set()
        for fl in fields:
            p, x, r = field_pairs(fl["d"], S)
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
