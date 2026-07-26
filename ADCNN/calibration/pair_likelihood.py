#!/usr/bin/env python3
"""Is a calibrated (CNN score + orbit-fit chi2) ranking better than either alone? -- MEASUREMENT.

Scientific basis. For a fixed false-positive budget the most powerful ranking of candidate alerts
is by the LIKELIHOOD RATIO LR(x) = p(x | real) / p(x | chance) (Neyman-Pearson). Any monotone
function of LR gives the same ordering, so the question "how do I weight CNN score against chi2"
has a determined answer -- it is not a free parameter to hand-tune. Two facts make it tractable:

  * the per-detection CNN score measures PIXEL morphology of one detection;
  * the pair chi2 measures TWO-EPOCH geometric/orbital consistency.

If those are conditionally independent given the truth label, LR factorises,
LR = LR_score(s_min) * LR_chi2(chi2), i.e. the evidence ADDS in log space. That is testable, and
this script tests it before assuming it.

Labelled data: the 82-field run_lambda injection campaign -- the same evidence the frozen
threshold was selected on. Each cached pair carries (min_score, min_mfsnr, rate, chi2, label),
label='tp' when both members are the same injected object, 'fp' for a chance link.

Everything is cross-validated BY FIELD (GroupKFold on the field id), never by pair: pairs inside
one field share detections, sky and seeing, so a pair-level split leaks and flatters the fit.

CAVEAT, stated up front: run_lambda measures trail lengths anomalously (MISS_AUDIT_V2D.md --
short-end bloom intercept ~1.0 px vs ~5.2 on dev and ~11.4 on real 0630 panels), and chi2 depends
on trail velocity through len_db. The RANKING conclusions (does chi2 add information over score?)
should transfer; the absolute chi2 scale may not. Confirm any shipped ranking on a real night.

Usage:
  python -m ADCNN.calibration.pair_likelihood [--cache-dir ...] [--out report.json]
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
DEFAULT_CACHE = REPO / "ADCNN/pipelines/heliolinc/run_lambda/_nomfsnr_cache"
RATE_LO, RATE_HI = 1.0, 8.0


def load(cache_dir):
    """-> (X dict of feature arrays, y bool 'is real', field id array). Rate-band restricted."""
    rows = []
    seen = {}
    for p in glob.glob(f"{cache_dir}/*_smin0.6_v3exact.json.gz"):
        seen[Path(p).name.split("_")[0]] = p
    for p in glob.glob(f"{cache_dir}/*_smin0.6_v3exact.json"):
        seen[Path(p).name.split("_")[0]] = p
    for k, p in sorted(seen.items()):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            c = json.load(f)
        for r in c["rows"]:
            # (min_score, min_mfsnr, rate, label, n_fp, obj, max_score, min_len, chi2, dpa, dsp, perp)
            rows.append((k, r[0], r[1], r[2], r[3], r[8]))
    if not rows:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    fld = np.array([r[0] for r in rows])
    smin = np.array([r[1] for r in rows], float)
    mfs = np.array([r[2] for r in rows], float)
    rate = np.array([r[3] for r in rows], float)
    y = np.array([r[4] == "tp" for r in rows])
    chi2 = np.array([r[5] for r in rows], float)
    m = (rate >= RATE_LO) & (rate <= RATE_HI) & np.isfinite(chi2)
    return dict(score=smin[m], chi2=chi2[m], mfsnr=mfs[m]), y[m], fld[m]


def auc(score, y):
    """Rank AUC (= P(random real ranked above random chance link)); ties get 0.5 credit."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def completeness_at_budget(rank_stat, y, budget):
    """Fraction of real pairs captured by the top-`budget` alerts -- the operational metric:
    with a fixed nightly follow-up/eyeball budget, how many real movers does this ranking keep?"""
    order = np.argsort(-rank_stat)
    top = order[:budget]
    return float(y[top].sum()) / max(int(y.sum()), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)

    X, y, fld = load(a.cache_dir)
    n, nreal = len(y), int(y.sum())
    print(f"[pair-lr] {n} labelled pairs over {len(set(fld))} fields | real {nreal} "
          f"({100*nreal/n:.2f}%) chance {n - nreal}", flush=True)

    # ---- 1. do the two features carry independent information? -------------------------------
    # Conditional (within-class) correlation is what matters for the naive-Bayes factorisation;
    # a MARGINAL correlation can be large purely because both features separate the classes.
    ls, lc = np.log(np.clip(X["score"], 1e-6, None)), np.log(np.clip(X["chi2"], 1e-6, None))
    r_real = float(np.corrcoef(ls[y], lc[y])[0, 1])
    r_chance = float(np.corrcoef(ls[~y], lc[~y])[0, 1])
    print(f"[pair-lr] within-class corr(log score, log chi2): real {r_real:+.3f}  chance {r_chance:+.3f}"
          f"   (near 0 => evidence adds in log space)", flush=True)

    # ---- 2. single-feature separation --------------------------------------------------------
    singles = {"score (CNN, higher better)": X["score"], "chi2 (geometry, lower better)": -X["chi2"],
               "mfsnr (higher better)": X["mfsnr"]}
    res = {"n_pairs": n, "n_real": nreal, "n_fields": len(set(fld)),
           "corr_within_real": r_real, "corr_within_chance": r_chance, "auc": {}, "completeness": {}}
    for name, s in singles.items():
        res["auc"][name] = round(auc(s, y), 4)
        print(f"[pair-lr] AUC {name:32s} {res['auc'][name]:.4f}", flush=True)

    # ---- 3. combined, cross-validated BY FIELD ----------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    F = np.column_stack([ls, lc, np.log(np.clip(X["mfsnr"], 1e-3, None))])
    combos = {"score+chi2": [0, 1], "score only": [0], "chi2 only": [1], "score+chi2+mfsnr": [0, 1, 2]}
    oof = {}
    for name, cols in combos.items():
        p = np.zeros(n)
        for tr, te in GroupKFold(n_splits=a.folds).split(F, y, groups=fld):
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(F[tr][:, cols], y[tr])
            p[te] = lr.predict_proba(F[te][:, cols])[:, 1]
        oof[name] = p
        res["auc"][f"CV {name}"] = round(auc(p, y), 4)
        print(f"[pair-lr] AUC CV {name:24s} {res['auc'][f'CV {name}']:.4f}", flush=True)
    lr_full = LogisticRegression(max_iter=2000, class_weight="balanced").fit(F[:, [0, 1]], y)
    res["coef_score_chi2"] = {"log_score": float(lr_full.coef_[0][0]),
                              "log_chi2": float(lr_full.coef_[0][1]),
                              "intercept": float(lr_full.intercept_[0])}
    print(f"[pair-lr] fit: logit P(real) = {lr_full.intercept_[0]:.3f} "
          f"+ {lr_full.coef_[0][0]:.3f}*log(score) + {lr_full.coef_[0][1]:.3f}*log(chi2)", flush=True)

    # ---- 4. the operational number: completeness at a fixed alert budget ---------------------
    print("\n[pair-lr] real pairs kept at a fixed nightly alert budget (top-N of the ranking):")
    for budget in (500, 1000, 2000, 5000, 10000):
        if budget > n:
            continue
        row = {}
        for name, s in (("score", X["score"]), ("chi2", -X["chi2"]),
                        ("CV score+chi2", oof["score+chi2"]), ("CV all3", oof["score+chi2+mfsnr"])):
            row[name] = round(100 * completeness_at_budget(s, y, budget), 2)
        res["completeness"][budget] = row
        print(f"   top {budget:6d}: " + "  ".join(f"{k} {v:5.1f}%" for k, v in row.items()), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\n[pair-lr] -> {a.out}", flush=True)
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
