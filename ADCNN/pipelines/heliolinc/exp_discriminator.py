"""EXPERIMENT: find the best 2-visit purity discriminator that is NOT biased against fast long-trail movers.
The shipped mf_snr>=10 cut rejects fast faint movers (mf_snr ~ snr_target*sqrt(PSF/trail_area)). Hypothesis:
for FAST movers the GEOMETRIC features (trail-PA vs chord-PA, trail-length-vs-rate, collinearity, orbit chi2)
are well-measured and separate true/false without the brightness bias. Enumerate ALL same-night visit pairs
(fixing the adjacency-only seeding bug), label true (both members -> same injected objID) / false, dump
features, and report which discriminator gives the best completeness@fixed-purity.
"""
from __future__ import annotations
import argparse, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
from ADCNN.pipelines.heliolinc.trail_state_link import pair_chi2

PIXSCALE = 0.2


def tag(dets, inj, tol_arcsec=3.0):
    out = np.array([None] * len(dets), dtype=object)
    tol = tol_arcsec / 3600.0
    for v, g in inj.groupby("visit"):
        m = (dets.visit == v).to_numpy(); dd = dets[m]
        if not len(dd):
            continue
        cd = np.cos(np.radians(g.dec.mean())); tr = cKDTree(np.c_[g.ra * cd, g.dec])
        dist, j = tr.query(np.c_[dd.ra * cd, dd.dec], k=1); idx = np.where(m)[0]
        for kk, jj, h in zip(idx, j, dist < tol):
            if h:
                out[kk] = g.objID.values[jj]
    return out


def all_pair_seeds(d, max_arc_min=40.0, rate_min=0.3, rate_max=10.0):
    """ALL same-night visit pairs within max_arc (fixes adjacency-only bug)."""
    mjd = d.mjd.to_numpy(); ra = d.ra.to_numpy(); dec = d.dec.to_numpy(); vis = d.visit.to_numpy()
    uv = sorted(set(vis.tolist()))
    vmjd = {v: float(np.median(mjd[vis == v])) for v in uv}
    idx_by = {v: np.where(vis == v)[0] for v in uv}
    pairs = []
    for ai in range(len(uv)):
        for bi in range(ai + 1, len(uv)):
            a_, b_ = uv[ai], uv[bi]; dt = vmjd[b_] - vmjd[a_]
            if dt <= 0 or dt * 1440.0 > max_arc_min:
                continue
            ia, ib = idx_by[a_], idx_by[b_]
            if not len(ia) or not len(ib):
                continue
            cd = float(np.cos(np.radians(dec[ib].mean())))
            tree = cKDTree(np.c_[ra[ib] * cd, dec[ib]]); dmin, dmax = rate_min * dt, rate_max * dt
            for i in ia:
                for jp in tree.query_ball_point([ra[i] * cd, dec[i]], dmax):
                    j = int(ib[jp])
                    if np.hypot((ra[j] - ra[i]) * cd, dec[j] - dec[i]) >= dmin:
                        pairs.append((int(i), j))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True); ap.add_argument("--inject", required=True)
    ap.add_argument("--truth", required=True); ap.add_argument("--score-min", type=float, default=0.80)
    ap.add_argument("--max-arc-min", type=float, default=40.0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = pd.read_csv(a.dets); inj = pd.read_csv(a.inject); truth = pd.read_csv(a.truth)
    d = d[(d.score >= a.score_min) & (d.get("art_frac", 0) < 0.3)].reset_index(drop=True)
    d["obj"] = tag(d, inj)
    print(f"[exp] {len(d)} dets >=score, {d.obj.notna().sum()} matched to injected", flush=True)
    pairs = all_pair_seeds(d, max_arc_min=a.max_arc_min)
    print(f"[exp] {len(pairs)} candidate pairs (all-visit-pairs seeding)", flush=True)
    rows = []
    for i, j in pairs:
        g = d.iloc[[i, j]]
        oi, oj = d.obj.iloc[i], d.obj.iloc[j]
        is_true = (oi is not None) and (oi == oj)
        chi2, info = pair_chi2(g)
        mf = float(g.mf_snr.min());
        li, lj = float(g.len_db.iloc[0]), float(g.len_db.iloc[1])
        len_ratio = max(li, lj) / max(min(li, lj), 1.0)
        # length-normalized SNR: mf_snr per unit sqrt(trail length) ~ surface-brightness significance
        mf_surf = mf * np.sqrt(max(min(li, lj), 1.0) / 4.0)   # rescale to ~PSF length 4px
        rows.append(dict(is_true=is_true, obj=(oi if is_true else None), mf=mf, mf_surf=mf_surf,
                         chi2=chi2, perp=info.get("perp", np.nan), resid=info.get("resid", np.nan),
                         dpa_tm=info.get("dpa_tm", np.nan), dspeed=info.get("dspeed", np.nan),
                         len_ratio=len_ratio, score_min=float(g.score.min()),
                         bound=info.get("bound", False)))
    df = pd.DataFrame(rows)
    nt = int(df.is_true.sum()); nf = len(df) - nt
    n_recoverable = int((truth.n_sightings >= 2).sum())
    print(f"[exp] pairs: TRUE {nt} (distinct objs {df[df.is_true].obj.nunique()}) / FALSE {nf} | recoverable objs {n_recoverable}", flush=True)
    if a.out:
        df.to_csv(a.out, index=False)
    # report discriminator separation + completeness@purity for candidate cuts
    def comp_pur(mask):
        sub = df[mask]; t = int(sub.is_true.sum()); f = len(sub) - t
        objs = sub[sub.is_true].obj.nunique()
        pur = t / max(len(sub), 1)
        return objs / max(n_recoverable, 1), pur, objs, f
    print("\n[exp] discriminator scan (completeness of recoverable objs, purity, n_obj, n_false):")
    print(f"  {'cut':32} {'compl':>7} {'purity':>7} {'nobj':>5} {'nfalse':>6}")
    cuts = {
        "shipped mf>=10 + chi2<=10": (df.mf >= 10) & (df.chi2 <= 10),
        "no-mf, chi2<=10": (df.chi2 <= 10),
        "no-mf, chi2<=5": (df.chi2 <= 5),
        "no-mf, chi2<=3": (df.chi2 <= 3),
        "no-mf, chi2<=10 + bound": (df.chi2 <= 10) & df.bound,
        "mf>=5 + chi2<=10": (df.mf >= 5) & (df.chi2 <= 10),
        "mf>=5 + chi2<=5": (df.mf >= 5) & (df.chi2 <= 5),
        "mf_surf>=8 + chi2<=10": (df.mf_surf >= 8) & (df.chi2 <= 10),
        "chi2<=5 + dpa_tm<8 + dspeed<.4": (df.chi2 <= 5) & (df.dpa_tm < 8) & (df.dspeed < 0.4),
        "chi2<=3 + len_ratio<1.5": (df.chi2 <= 3) & (df.len_ratio < 1.5),
    }
    for name, m in cuts.items():
        c, p, no, nfa = comp_pur(m)
        print(f"  {name:32} {c:7.3f} {p:7.3f} {no:5d} {nfa:6d}")
    print("\n[exp] feature medians (TRUE vs FALSE):")
    for col in ["mf", "mf_surf", "chi2", "perp", "resid", "dpa_tm", "dspeed", "len_ratio"]:
        t = df[df.is_true][col].replace([np.inf, -np.inf], np.nan).median()
        f = df[~df.is_true][col].replace([np.inf, -np.inf], np.nan).median()
        print(f"  {col:12} true {t:8.2f}  false {f:8.2f}")


if __name__ == "__main__":
    main()
