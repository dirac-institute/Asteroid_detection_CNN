#!/usr/bin/env python3
"""Fit the alert-ranking likelihood on a REAL night's own false-positive population.

Companion to night_pair_injection.py. That script injected pair-consistent movers into a real
night's repeat-pointing regions and the night was re-detected+linked; here each resulting 2-visit
alert is labelled by truth (both members matched to the SAME injected objID => real mover;
anything else => a false pair drawn from that night's genuine residual population) and the
ranking model is fit and scored.

Model: logit P(real) = a + b*log(score_min) + c*log(chi2) + d*log(mfsnr_min).
The additive form is not arbitrary -- it is the Neyman-Pearson likelihood ratio under conditional
independence of the features given the label, which was measured to hold (within-class
corr(log score, log chi2) = -0.05 real / -0.00 chance, ADCNN/calibration/pair_likelihood.py).

Cross-validation is BY POINTING GROUP: alerts from one pointing share sky, seeing, template and
residual population, so a pair-level split leaks.

The headline comparison is completeness at a fixed nightly alert budget -- with a budget of N
alerts to eyeball or follow up, what fraction of the real movers does each ranking keep?

Usage:
  python -m ADCNN.calibration.fit_alert_ranking --alerts <calib alerts.jsonl> \
      --inject <inject.csv> --dets <calib masked dets csv> [--out report.json]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_ARCSEC = 2.0


def label_alerts(alerts, inject, tol_arcsec=MATCH_ARCSEC):
    """-> (features DataFrame, y bool, group array). An alert is REAL iff both of its epochs
    match the same injected objID within tol_arcsec at that epoch's own visit."""
    from scipy.spatial import cKDTree
    by_visit = {}
    for v, g in inject.groupby("visit"):
        by_visit[int(v)] = (cKDTree(np.column_stack([
            g.ra.to_numpy() * np.cos(np.radians(g.dec.to_numpy())), g.dec.to_numpy()])),
            g.objID.to_numpy())
    tol_deg = tol_arcsec / 3600.0
    rows = []
    for al in alerts:
        eps = al["epochs"]
        ids = []
        for e in eps:
            t = by_visit.get(int(e["visit"]))
            if t is None:
                ids.append(None); continue
            tree, objs = t
            q = [float(e["ra"]) * np.cos(np.radians(float(e["dec"]))), float(e["dec"])]
            dist, idx = tree.query(q, distance_upper_bound=tol_deg)
            ids.append(objs[idx] if np.isfinite(dist) else None)
        real = (len(ids) >= 2 and ids[0] is not None and all(i == ids[0] for i in ids))
        v = al.get("vetting") or {}
        m = al.get("motion") or {}
        o = al.get("orbit") or {}
        # pointing group: injected objIDs are prefixed g<grp>_; fall back to the visit pair
        grp = str(ids[0]).split("_")[0] if real else f"v{eps[0]['visit']}"
        rows.append(dict(real=bool(real), group=grp,
                         score_min=v.get("score_min", np.nan),
                         mfsnr_min=v.get("mfsnr_min", np.nan),
                         chi2=o.get("chi2", np.nan),
                         rate=m.get("rate_degday", np.nan),
                         arc_min=al.get("arcMin", np.nan),
                         prio=al.get("priorityScore", np.nan)))
    d = pd.DataFrame(rows)
    return d


def auc(stat, y):
    from scipy.stats import rankdata
    r = rankdata(stat)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def completeness_at(stat, y, budget):
    order = np.argsort(-stat)
    return float(y[order[:budget]].sum()) / max(int(y.sum()), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True, help="alerts.jsonl from linking the INJECTED night")
    ap.add_argument("--inject", required=True, help="inject.csv used for that night")
    ap.add_argument("--out", default=None)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)

    alerts = [json.loads(l) for l in open(a.alerts)] if os.path.getsize(a.alerts) else []
    if not alerts:
        raise SystemExit("no alerts")
    inj = pd.read_csv(a.inject)
    d = label_alerts(alerts, inj)
    d = d[np.isfinite(d.chi2) & np.isfinite(d.score_min)].reset_index(drop=True)
    y = d.real.to_numpy()
    print(f"[fit] {len(d)} labelled 2v alerts | real {int(y.sum())} ({100*y.mean():.2f}%) "
          f"| false {int((~y).sum())} (this night's own residual population)", flush=True)
    if y.sum() < 30:
        print("[fit] WARNING: very few real pairs -- fit will be unstable", flush=True)

    res = {"n_alerts": len(d), "n_real": int(y.sum()), "alerts": os.path.abspath(a.alerts), "auc": {}}
    ls = np.log(np.clip(d.score_min.to_numpy(), 1e-6, None))
    lc = np.log(np.clip(d.chi2.to_numpy(), 1e-6, None))
    lm = np.log(np.clip(d.mfsnr_min.to_numpy(), 1e-3, None))

    # within-class correlation -- does the additive (naive-Bayes) form hold on THIS null?
    if y.sum() > 5:
        res["corr_within_real"] = float(np.corrcoef(ls[y], lc[y])[0, 1])
    res["corr_within_false"] = float(np.corrcoef(ls[~y], lc[~y])[0, 1])
    print(f"[fit] within-class corr(log score, log chi2): "
          f"real {res.get('corr_within_real', float('nan')):+.3f}  false {res['corr_within_false']:+.3f}",
          flush=True)

    for name, s in (("score", d.score_min.to_numpy()), ("chi2 (neg)", -d.chi2.to_numpy()),
                    ("mfsnr", d.mfsnr_min.to_numpy()), ("priorityScore", d.prio.to_numpy())):
        res["auc"][name] = round(auc(s, y), 4)
        print(f"[fit] AUC {name:16s} {res['auc'][name]:.4f}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    F = np.column_stack([ls, lc, lm])
    groups = d.group.to_numpy()
    ng = len(set(groups))
    combos = {"score+chi2": [0, 1], "score+chi2+mfsnr": [0, 1, 2]}
    oof = {}
    nsp = max(2, min(a.folds, ng))
    for name, cols in combos.items():
        p = np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=nsp).split(F, y, groups=groups):
            if y[tr].sum() < 2:
                continue
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(F[tr][:, cols], y[tr])
            p[te] = lr.predict_proba(F[te][:, cols])[:, 1]
        oof[name] = p
        res["auc"][f"CV {name}"] = round(auc(p, y), 4)
        print(f"[fit] AUC CV {name:20s} {res['auc'][f'CV {name}']:.4f}  ({nsp} folds over {ng} groups)",
              flush=True)

    lrf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(F, y)
    res["coef"] = {"intercept": float(lrf.intercept_[0]), "log_score": float(lrf.coef_[0][0]),
                   "log_chi2": float(lrf.coef_[0][1]), "log_mfsnr": float(lrf.coef_[0][2])}
    print(f"[fit] logit P(real) = {res['coef']['intercept']:.3f} "
          f"+ {res['coef']['log_score']:.3f}*log(score) "
          f"+ {res['coef']['log_chi2']:.3f}*log(chi2) "
          f"+ {res['coef']['log_mfsnr']:.3f}*log(mfsnr)", flush=True)

    print("\n[fit] real movers kept at a fixed nightly alert budget:")
    res["completeness"] = {}
    for b in (100, 250, 500, 1000, 2000, 5000):
        if b >= len(d):
            continue
        row = {"score": round(100 * completeness_at(d.score_min.to_numpy(), y, b), 1),
               "chi2": round(100 * completeness_at(-d.chi2.to_numpy(), y, b), 1),
               "priorityScore": round(100 * completeness_at(d.prio.to_numpy(), y, b), 1),
               "CV score+chi2": round(100 * completeness_at(oof["score+chi2"], y, b), 1),
               "CV all3": round(100 * completeness_at(oof["score+chi2+mfsnr"], y, b), 1)}
        res["completeness"][b] = row
        print(f"   top {b:5d}: " + "  ".join(f"{k} {v:5.1f}%" for k, v in row.items()), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\n[fit] -> {a.out}", flush=True)
    return res


if __name__ == "__main__":
    main()
