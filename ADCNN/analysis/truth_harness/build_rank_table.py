#!/usr/bin/env python3
"""Build the ranker's feature/label table from an injection run, with LABEL and FEATURE hygiene.

LABEL HYGIENE -- an alert is True only if BOTH of its epochs match the SAME injected object.
A chance link that pairs one injected detection with one real FP detection is a WRONG link that
happens to contain a real detection; labelling it True would teach the ranker to promote wrong links.

FEATURE HYGIENE -- every feature must be a property of the CANDIDATE, never of the FIELD.
`arcMin`, `fpp.dtMin`, `fpp.n1`, `fpp.n2` are visit-pair properties, and the CV groups ARE visit
pairs, so including them lets the model identify the held-out fold instead of ranking within it.
`lambdaPair` is kept: it is the chance-link expectation, which is per-candidate content.

Every retained feature has a physically-correct monotone direction (SIGN below), so the fit can be
constrained to mover physics rather than to this night's FP taxonomy.

Usage:  python build_rank_table.py <alerts.jsonl> <truth.csv> <out.csv>
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# feature -> required sign of its logistic coefficient (+1 raises P(real), -1 lowers it)
SIGN = {
    "log_chi2":   -1,   # linker's physical (orbit + trail) consistency: lower is better
    "smin":       +1,   # weakest member's CNN score
    "smax":       +1,   # strongest member's CNN score
    "log_mfsnr":  +1,   # matched-filter SNR: brighter is more real
    "log_tlen":   +1,   # trail length: the signature of a fast mover
    "log_lam":    -1,   # expected chance-link count for this pair: lower is better
    "d_tlen":     -1,   # |tA-tB|/(tA+tB): one object has ONE trail length
    "d_speed":    -1,   # |log(rate implied by trail / rate implied by motion)|: they must agree
}
PIX, EXPTIME, SOLARDAY = 0.2, 30.0, 86400.0


def radec_to_unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)


def build(alerts_path, truth_path, out_path, tol_arcsec=3.0):
    A = [json.loads(l) for l in open(alerts_path)]
    T = pd.read_csv(truth_path)
    tol = 2 * np.sin(np.radians(tol_arcsec / 3600.0) / 2)

    # one KD-tree per (visit, epoch-side) over the TRUE injected positions, carrying oid
    trees = {}
    for v, col in [("visitA", "A"), ("visitB", "B")]:
        for vis, g in T.groupby(v):
            key = (int(vis), col)
            xyz = radec_to_unit(g[f"ra{col}"].to_numpy(), g[f"dec{col}"].to_numpy())
            trees[key] = (cKDTree(xyz), g["oid"].to_numpy(), g)

    rows = []
    for a in A:
        eps = a["epochs"]
        if len(eps) < 2:
            continue
        # --- label: BOTH epochs must hit the SAME oid ---
        oids, meta = [], None
        for e in eps:
            hit = None
            for side in "AB":
                tr = trees.get((int(e["visit"]), side))
                if tr is None:
                    continue
                d, i = tr[0].query(radec_to_unit([e["ra"]], [e["dec"]]), k=1)
                if d[0] < tol:
                    hit = (int(tr[1][i[0]]), tr[2].iloc[i[0]])
                    break
            oids.append(hit[0] if hit else -1)
            if hit and meta is None:
                meta = hit[1]
        y = len(set(oids)) == 1 and oids[0] >= 0

        def num(d, k):
            x = d.get(k)
            return np.nan if x is None else float(x)

        v = a.get("vetting") or {}
        o = a.get("orbit") or {}
        f = a.get("fpp") or {}
        m = a.get("motion") or {}
        tl = v.get("trail_len_px") or [np.nan, np.nan]
        tA = np.nan if tl[0] is None else float(tl[0])
        tB = np.nan if tl[1] is None else float(tl[1])
        tmean = 0.5 * (tA + tB)
        rate = num(m, "rate_degday")
        rate_from_trail = tmean * PIX / 3600.0 * (SOLARDAY / EXPTIME)
        rows.append(dict(
            log_chi2=np.log10(max(num(o, "chi2"), 1e-3)),
            smin=num(v, "score_min"),
            smax=num(v, "score_max"),
            log_mfsnr=np.log10(max(num(v, "mfsnr_min"), 0.1)),
            log_tlen=np.log10(max(tmean, 0.5)),
            log_lam=np.log10(max(num(f, "lambdaPair"), 1e-6)),
            d_tlen=abs(tA - tB) / max(tA + tB, 1e-6),
            d_speed=abs(np.log10(max(rate_from_trail, 1e-6) / max(rate, 1e-6))),
            # --- NOT features: bookkeeping, stratification, grouping ---
            pscore=num(a, "priorityScore"),
            rate=rate,
            group=f"{min(int(e['visit']) for e in eps)}_{max(int(e['visit']) for e in eps)}",
            alertId=a.get("alertId"),
            y=bool(y),
            snr_t=float(meta["snr_t"]) if (y and meta is not None) else np.nan,
            L_px=float(meta["L_px"]) if (y and meta is not None) else np.nan,
        ))
    D = pd.DataFrame(rows)
    D = D[np.isfinite(D[list(SIGN)].to_numpy()).all(1)].reset_index(drop=True)
    D.to_csv(out_path, index=False)
    n_part = 0
    print(f"[rank-table] {len(D):,} alerts | {int(D.y.sum()):,} TRUE (both epochs, same oid) | "
          f"{D.group.nunique()} visit-pair groups -> {out_path}")
    print(f"[rank-table] true-alert SNR: " +
          " ".join(f"{lo}-{hi}:{int(((D.y) & (D.snr_t >= lo) & (D.snr_t < hi)).sum())}"
                   for lo, hi in [(2, 4), (4, 6), (6, 8), (8, 10)]))
    return D


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3])
