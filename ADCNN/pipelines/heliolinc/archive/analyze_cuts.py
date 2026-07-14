"""Investigate NON-ML purity levers on the dumped true/false pair features. Goal: cut the surviving FALSE
2-visit links ~100x (0.143 -> 1.35e-3, i.e. 3sigma) while KEEPING >=75% completeness, so we can lower the
ADCNN score floor and recover more real movers at the 3sigma price.

For each lever (orbit physicality a/e/q, the 5 chi2 components, chord rate, trail-length consistency) it
reports the true-vs-false separation and the false-retention at a completeness-preserving cut, then greedily
combines levers. Completeness is EXACT (distinct injected objID recovered / recoverable denominator).
"""
from __future__ import annotations
import argparse, glob
import numpy as np, pandas as pd


def comp(df_true_kept, n_recoverable):
    # PHYSICAL objects: (field, objID) -- objID names repeat across fields
    return df_true_kept[["field", "obj"]].drop_duplicates().shape[0] / max(n_recoverable, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dir", default="ADCNN/pipelines/heliolinc/run_lambda", help="for inject_*.csv recoverable denom")
    ap.add_argument("--chi2-base", type=float, default=3.0)
    ap.add_argument("--retain", type=float, default=0.90, help="keep >= this FRACTION of baseline-recovered objects")
    a = ap.parse_args()
    df = pd.read_parquet(a.features)
    # field-unique recoverable denominator from the inject files
    recoverable = set()
    for k in df.field.unique():
        try:
            inj = pd.read_csv(f"{a.dir}/inject_{k}.csv"); cnt = inj.groupby("objID").size()
            for o in cnt[cnt >= 2].index: recoverable.add((str(k), str(o)))
        except Exception: pass
    nrec = len(recoverable)
    base = df[df.chi2 <= a.chi2_base].copy()
    base["obj"] = base["obj"].astype(str); base["field"] = base["field"].astype(str)
    T = base[base.is_true]; F = base[~base.is_true]
    comp0 = comp(T, nrec)
    a.min_comp = a.retain * comp0   # retain >=90% of the (already few) real recoveries
    print(f"=== baseline chi2<={a.chi2_base}: {len(T)} true pairs ({T[['field','obj']].drop_duplicates().shape[0]} "
          f"PHYSICAL objs), {len(F)} false pairs | completeness {comp0:.3f} (denom {nrec}) ===", flush=True)
    print(f"target: keep completeness >= {a.min_comp:.3f} ({a.retain:.0%} of baseline) while cutting false "
          f"-> need false-retention <= {1.35e-3/0.143:.4f} (~100x) for 3sigma at S=0.80\n", flush=True)

    # per-feature distributions (true vs false) -- where do they separate?
    feats = ["a", "e", "q", "perp", "resid", "dsnr", "dpa_tm", "dspeed", "rate", "len_ratio", "len_min",
             "chi2", "mfsnr_min", "nnp_min", "dens_max", "dens_min", "art_sum_max", "art_any_max", "m_detneg_max"]
    feats = [f for f in feats if f in base.columns]
    print(f"{'feat':>9} | {'true p50':>9} {'true p90':>9} | {'false p50':>9} {'false p10':>9} {'false p90':>9}")
    for f in feats:
        if f not in base: continue
        t = T[f].replace([np.inf, -np.inf], np.nan).dropna(); ff = F[f].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(t) or not len(ff): continue
        print(f"{f:>9} | {t.median():9.3f} {t.quantile(.9):9.3f} | {ff.median():9.3f} {ff.quantile(.1):9.3f} {ff.quantile(.9):9.3f}")

    # single-lever: tightest cut on each feature that keeps completeness >= min_comp; report false-retention
    print(f"\n=== single-lever cuts (keep completeness >= {a.min_comp}) ===")
    print(f"{'lever':>22} {'cut':>22} {'comp':>6} {'false_ret':>10}")
    levers = []
    def upper(f):  # keep <= thresh
        for thr in np.quantile(T[f].dropna(), np.linspace(0.99, 0.5, 60)):
            kept_t = T[T[f] <= thr]
            if comp(kept_t, nrec) >= a.min_comp:
                fr = (F[f] <= thr).mean()
                return thr, comp(kept_t, nrec), fr
        return np.nan, np.nan, np.nan
    def band(f, lo_q=0.0):  # keep within [lo, hi]
        lo = T[f].quantile(0.02); hi = T[f].quantile(0.98)
        kept_t = T[(T[f] >= lo) & (T[f] <= hi)]
        fr = ((F[f] >= lo) & (F[f] <= hi)).mean()
        return (lo, hi), comp(kept_t, nrec), fr
    for f in ["dpa_tm", "dspeed", "resid", "perp", "len_ratio", "chi2", "e",
              "dens_max", "dens_min", "art_sum_max", "art_any_max", "m_detneg_max"]:
        if f in base:
            thr, c, fr = upper(f)
            print(f"{f+' <=':>22} {thr:22.3f} {c:6.3f} {fr:10.4f}"); levers.append((f, "u", thr, fr))
    for f in ["a", "q", "rate", "mfsnr_min", "len_min", "nnp_min"]:
        if f in base:
            (lo, hi), c, fr = band(f)
            print(f"{f+' band':>22} {f'[{lo:.2f},{hi:.2f}]':>22} {c:6.3f} {fr:10.4f}"); levers.append((f, "b", (lo, hi), fr))

    # greedy AND-combination: add the lever that most reduces surviving false, while completeness stays >=min
    print(f"\n=== greedy AND-combination (completeness >= {a.min_comp}) ===")
    keepT = pd.Series(True, index=T.index); keepF = pd.Series(True, index=F.index)
    used = []
    for _ in range(len(levers)):
        best = None
        for (f, kind, cut, _) in levers:
            if f in used: continue
            if kind == "u":
                mt = T[f] <= cut; mf = F[f] <= cut
            else:
                lo, hi = cut; mt = (T[f] >= lo) & (T[f] <= hi); mf = (F[f] >= lo) & (F[f] <= hi)
            c = comp(T[keepT & mt], nrec); fr = (keepF & mf).sum()
            if c >= a.min_comp and (best is None or fr < best[3]):
                best = (f, kind, cut, fr, mt, mf, c)
        if best is None: break
        f, kind, cut, fr, mt, mf, c = best
        keepT &= mt; keepF &= mf; used.append(f)
        false_ret = keepF.sum() / max(len(F), 1)
        print(f"+ {f:>10} ({cut if kind=='u' else f'[{cut[0]:.2f},{cut[1]:.2f}]'}): comp {c:.3f}, "
              f"false survivors {int(keepF.sum())}/{len(F)} = {false_ret:.4f}", flush=True)
        if false_ret <= 1.35e-3/0.143:
            print(f"  >>> reached ~100x false cut at completeness {c:.3f} <<<")
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
