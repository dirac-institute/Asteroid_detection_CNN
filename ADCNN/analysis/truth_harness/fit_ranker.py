#!/usr/bin/env python3
"""Fit and STRESS-TEST the alert ranker on injections. Nothing here uses known-object recovery:
known objects are bright and slow, so recovering them cannot validate a faint-FAST ranker.

Three guards, because the positives are synthetic and the negatives are a real night's FPs, so a
free-form fit will happily learn "synthetic-ness" or "this night's FP taxonomy" instead of mover-ness:

  1. SIGN CONSTRAINTS. Every coefficient is forced to the physically-correct direction (see
     build_rank_table.SIGN). The unconstrained fit put NEGATIVE weight on mfsnr and trail length --
     brighter and longer => less real -- which is not mover physics, it is 0706's FP taxonomy
     (bright-star residuals are bright, satellite streaks are long). Weights like that invert on a
     night with different seeing, moon or star density.
  2. FOLD-FRACTION BUDGETS. Leave-one-visit-pair-out folds hold ~2k alerts, so a fixed "@500" cutoff
     is 18% of a fold but 1.8% of the pooled night -- which alone inflated every ranker. Budgets are
     therefore a FRACTION of each fold (10% ~ the 1k product out of ~10k alerts; 2% = top-of-list).
  3. SCORE-INFLATION STRESS TEST. Injected detections score ~+0.24 above the night's typical
     detection at matched SNR and matched trail length. That delta is confounded (the comparison set
     is ~99% false positives, and out-scoring FPs is what the CNN is FOR), so it is an UPPER BOUND on
     synthetic inflation, not a measurement of it. We therefore re-rank with the true alerts' scores
     pushed DOWN by delta and check whether the ranker still beats production. A ranker whose gain
     survives the full upper-bound penalty does not depend on the artefact.

Usage:  python fit_ranker.py <rank_table.csv> [<cross_night_table.csv>]
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "outputs/runs/pa_validate")
from build_rank_table import SIGN

FEATS = list(SIGN)
FRACS = [0.02, 0.10]


def fit_logistic(X, y, constrained=True, l2=1.0, cols=None):
    """Sign-constrained logistic regression on standardised features (bounded L-BFGS-B)."""
    n, p = X.shape
    w0 = np.zeros(p + 1)

    def nll(w):
        z = X @ w[:p] + w[p]
        z = np.clip(z, -30, 30)
        return float(np.mean(np.logaddexp(0, z) - y * z) + l2 * np.sum(w[:p] ** 2) / n)

    bounds = [(None, None)] * (p + 1)
    if constrained:
        bounds = [((0, None) if SIGN[f] > 0 else (None, 0)) for f in (cols or FEATS)] + [(None, None)]
    r = minimize(nll, w0, method="L-BFGS-B", bounds=bounds)
    return r.x[:p], r.x[p]


def recall_at_frac(score, y, frac):
    """Recall of true alerts in the top `frac` of THIS set, ranked by `score` (higher = better)."""
    n = len(score)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(-score, kind="stable")[:k]
    t = int(y.sum())
    return (float(y[idx].sum()) / t if t else np.nan), k, t


def group_cv(D, penalty=0.0, verbose=True):
    """Leave-one-visit-pair-out. Train and test share no field. Budgets scale with fold size."""
    groups = sorted(D.group.unique())
    mu, sd = D[FEATS].mean(), D[FEATS].std().replace(0, 1)
    out = {k: {f: [] for f in FRACS} for k in ("production", "logistic", "constrained", "physics", "hedge", "gbm")}
    sizes = []
    for g in groups:
        te = D.group == g
        tr = ~te
        if D.loc[tr, "y"].sum() < 5 or D.loc[te, "y"].sum() < 1:
            continue
        Xtr = ((D.loc[tr, FEATS] - mu) / sd).to_numpy()
        Xte = ((D.loc[te, FEATS] - mu) / sd).to_numpy()
        ytr = D.loc[tr, "y"].to_numpy().astype(float)
        yte = D.loc[te, "y"].to_numpy().astype(bool)
        pscore = D.loc[te, "pscore"].to_numpy().copy()
        if penalty:
            # deployment stress: the HELD-OUT true movers score `penalty` lower than the synthetic
            # ones the model was trained on. Applied to the test fold only, after standardisation.
            for f, col in (("smin", FEATS.index("smin")), ("smax", FEATS.index("smax"))):
                Xte[yte, col] -= penalty / sd[f]
            # production is priorityScore = base + 0.95*score_min + dt_bonus, i.e. PURELY the weakest
            # member's CNN score -- so the same penalty must hit it, or the test is rigged.
            pscore[yte] -= 0.95 * penalty
        scores = {"production": pscore}
        for name, con in (("logistic", False), ("constrained", True)):
            w, b = fit_logistic(Xtr, ytr, constrained=con)
            scores[name] = Xte @ w + b
        # PHYSICS-ONLY: no CNN score anywhere, so it is IMMUNE to score inflation by construction.
        # If the synthetic-vs-real score gap is real, this is the only ranker whose injection-fitted
        # numbers transfer unchanged; the question is what it costs when the gap is zero.
        pi = [i for i, f in enumerate(FEATS) if f not in ("smin", "smax")]
        wp, bp = fit_logistic(Xtr[:, pi], ytr, constrained=True, cols=[FEATS[i] for i in pi])
        scores["physics"] = Xte[:, pi] @ wp + bp
        # HEDGE: interleave the score-based and physics-only orderings (alternate picks). delta is
        # NOT measurable without labelled real movers, so instead of betting the product on a guess,
        # this spends half the budget on each. Its worst case over delta is bounded below by roughly
        # the better half-budget of the two, which is what "trustworthy" means when the nuisance
        # parameter is unknown: maximise the worst case, not the best case.
        def interleave(sa, sb):
            oa = list(np.argsort(-sa, kind="stable")); ob = list(np.argsort(-sb, kind="stable"))
            seen, order = set(), []
            for i in range(len(sa)):
                for o in (oa, ob):
                    while o and o[0] in seen:
                        o.pop(0)
                    if o:
                        k = o.pop(0); seen.add(k); order.append(k)
                if len(order) >= len(sa):
                    break
            rank = np.empty(len(sa)); rank[np.array(order[:len(sa)])] = np.arange(len(order[:len(sa)]))
            return -rank
        scores["hedge"] = interleave(scores["constrained"], scores["physics"])
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08).fit(Xtr, ytr)
            scores["gbm"] = m.predict_proba(Xte)[:, 1]
        except Exception:
            scores.pop("gbm", None)
        sizes.append((int(te.sum()), int(yte.sum())))
        for name, s in scores.items():
            for f in FRACS:
                out[name][f].append(recall_at_frac(s, yte, f)[0])
    if verbose:
        n_med = int(np.median([s[0] for s in sizes]))
        print(f"  folds {len(sizes)} | median fold {n_med:,} alerts, "
              f"{int(np.median([s[1] for s in sizes]))} true | "
              f"budgets: " + ", ".join(f"{int(f*100)}% = {int(round(f*n_med))}" for f in FRACS))
        hdr = "  " + "ranking".ljust(14) + "".join(f"{'top '+str(int(f*100))+'%':>18}" for f in FRACS)
        print(hdr)
        for name in ("production", "logistic", "constrained", "physics", "hedge", "gbm"):
            if not out.get(name) or not out[name][FRACS[0]]:
                continue
            row = "  " + name.ljust(14)
            for f in FRACS:
                v = np.array(out[name][f], float)
                row += f"{100*np.nanmean(v):>12.1f}% +-{100*np.nanstd(v):>4.1f}"
            print(row)
    return out


def main(table, cross=None):
    D = pd.read_csv(table)
    print(f"\n=== {table}: {len(D):,} alerts, {int(D.y.sum())} true, {D.group.nunique()} groups ===")

    print("\n[1] SIGN-CONSTRAINED vs FREE fit (full data, standardised coefficients)")
    mu, sd = D[FEATS].mean(), D[FEATS].std().replace(0, 1)
    X = ((D[FEATS] - mu) / sd).to_numpy(); y = D.y.to_numpy().astype(float)
    wf, _ = fit_logistic(X, y, constrained=False)
    wc, bc = fit_logistic(X, y, constrained=True)
    print(f"  {'feature':<11}{'want':>6}{'free':>9}{'constrained':>13}")
    for i, f in enumerate(FEATS):
        flag = "  <-- WRONG SIGN" if np.sign(wf[i]) != SIGN[f] and abs(wf[i]) > 0.05 else ""
        print(f"  {f:<11}{'+' if SIGN[f]>0 else '-':>6}{wf[i]:>9.2f}{wc[i]:>13.2f}{flag}")

    print("\n[2] LEAVE-ONE-VISIT-PAIR-OUT CV -- recall of true alerts, budget = fraction of fold")
    group_cv(D)

    print("\n[3] STRESS TEST -- held-out true movers' CNN scores pushed down by delta")
    print("    (delta 0.24 = the FULL measured injected-vs-real gap, an upper bound on inflation)")
    for pen in (0.05, 0.10, 0.24):
        print(f"  -- penalty {pen:.2f} --")
        group_cv(D, penalty=pen)

    print("\n[4] STRATIFIED by injected SNR (full-data constrained fit, ranked over the whole night)")
    s = X @ wc + bc
    order = np.argsort(-s, kind="stable"); po = np.argsort(-D.pscore.to_numpy(), kind="stable")
    for lo, hi in [(2, 4), (4, 6), (6, 8), (8, 10)]:
        m = (D.snr_t >= lo) & (D.snr_t < hi)
        if m.sum() == 0:
            print(f"  SNR {lo}-{hi}: no true alerts"); continue
        for nm, o in (("constrained", order), ("production", po)):
            k = max(1, int(0.10 * len(D)))
            top = np.zeros(len(D), bool); top[o[:k]] = True
            r = float((m & top).sum()) / int(m.sum())
            print(f"  SNR {lo:>2}-{hi:<2} n={int(m.sum()):>4} {nm:<12} top10% recall {100*r:>5.1f}%")

    if cross:
        C = pd.read_csv(cross)
        print(f"\n[5] CROSS-NIGHT TRANSFER -> {cross}: {len(C):,} alerts, {int(C.y.sum())} true")
        # Standardise the NEW night with the TRAINING night's mu/sd -- correct for deployment (a
        # frozen model applied to an unseen night), but it means a distribution shift would show up
        # as a transfer failure. Print both so the two are distinguishable.
        Xc = ((C[FEATS] - mu) / sd).to_numpy()
        print(f"  {'feature':<11}{'train mean':>12}{'test mean':>12}{'test sd':>10}")
        for i, f in enumerate(FEATS):
            print(f"  {f:<11}{0.0:>12.2f}{Xc[:, i].mean():>12.2f}{Xc[:, i].std():>10.2f}"
                  + ("   <-- SHIFTED" if abs(Xc[:, i].mean()) > 1.0 else ""))
        sc = Xc @ wc + bc
        yc = C.y.to_numpy().astype(bool)
        for f in FRACS:
            rp, k, t = recall_at_frac(C.pscore.to_numpy(), yc, f)
            rc, _, _ = recall_at_frac(sc, yc, f)
            print(f"  top {int(f*100):>2}% (n={k:,} of {len(C):,}, {t} true): "
                  f"production {100*rp:>5.1f}%   constrained {100*rc:>5.1f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
