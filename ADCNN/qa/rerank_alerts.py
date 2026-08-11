#!/usr/bin/env python3
"""Re-rank an alert stream by the calibrated reality probability P(real).

Post-hoc by design: it reads alerts.jsonl, computes P(real) from fields the alerts already
carry, annotates and re-sorts. No re-linking (~50 min saved) and no re-imaging -- the cutout
cache is keyed by alert index, so re-running this then re-rendering is the cheap way to change
the operating point. That is exactly the "publish low, cut later on the ranked list" workflow.

Model: logit P(real) = a + b*log(score_min) + c*log(chi2) + d*log(mfsnr_min), coefficients from
ADCNN/calibration/alert_ranking_model.json (fit on a real night's own FP population, see
ALERT_RANKING.md). The additive form is the Neyman-Pearson likelihood ratio under the measured
conditional independence of the features -- not a hand-tuned weighting.

Ordering keeps the demotion class primary by default (a veto-flagged alert never outranks a clean
one -- the FLAG-not-drop contract), with P(real) ordering WITHIN each class. --ignore-class ranks
purely by P(real) for diagnostics.

Usage:
  python -m ADCNN.qa.rerank_alerts --alerts stream/alerts.jsonl [--out stream/alerts_pcal.jsonl]
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
DEFAULT_MODEL = REPO / "ADCNN/calibration/alert_ranking_model.json"


def p_real(alert, coef, domain=None):
    """Calibrated P(real) for one alert; None when a required field is missing/degenerate.

    Features are CLIPPED to the model's calibration domain. This is not cosmetic: mf_snr has a
    numerics blowup on degenerate panels (values to ~1e9), and without clipping log() carries
    those straight into the logit and pins P(real)=1 on exactly the corrupt panels."""
    v = alert.get("vetting") or {}
    o = alert.get("orbit") or {}
    s, c, m = v.get("score_min"), o.get("chi2"), v.get("mfsnr_min")
    if s is None or c is None or m is None:
        return None
    try:
        s, c, m = float(s), float(c), float(m)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (s, c, m)):
        return None
    if domain:
        s = min(max(s, domain["score_min"][0]), domain["score_min"][1])
        c = min(max(c, domain["chi2"][0]), domain["chi2"][1])
        m = min(max(m, domain["mfsnr_min"][0]), domain["mfsnr_min"][1])
    if not (s > 0 and c > 0):
        return None
    m = max(m, 1e-3)
    z = (coef["intercept"] + coef["log_score"] * math.log(s)
         + coef["log_chi2"] * math.log(c) + coef["log_mfsnr"] * math.log(m))
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--out", default=None, help="default: overwrite --alerts in place")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--ignore-class", action="store_true",
                    help="rank purely by P(real), letting veto-flagged alerts outrank clean ones "
                         "(diagnostic only -- breaks the FLAG-not-drop ordering contract)")
    a = ap.parse_args(argv)

    model = json.load(open(a.model))
    coef = model["coef"]
    domain = model.get("domain")
    alerts = [json.loads(l) for l in open(a.alerts)] if os.path.getsize(a.alerts) else []
    if not alerts:
        print("[rerank] 0 alerts", flush=True); return 0

    from ADCNN.linking.rank_alerts import _rank_class
    n_missing = 0
    for al in alerts:
        p = p_real(al, coef, domain)
        if p is None:
            n_missing += 1
        al["ranking"] = {"pReal": p, "model": os.path.basename(a.model),
                         "night_fit": model.get("night")}
    # TIER before pReal. 3+visit tracks have orbit.chi2=None, so P(real) is not computable for them
    # and pReal=None mapped to -1.0 -- sorting the ~100%-purity discovery tier LAST. On 0706 that put
    # the night's only 3+visit alert at rank 4795 of 5790. `priority` is 1 for 3+visit, 2 for 2visit,
    # so keying on it first preserves the tier precedence that priority_score establishes.
    _pr = lambda al: (al["ranking"]["pReal"] if al["ranking"]["pReal"] is not None else -1.0)
    key = ((lambda al: (al.get("priority", 9), -_pr(al)))
           if a.ignore_class else
           (lambda al: (_rank_class(al), al.get("priority", 9), -_pr(al))))
    ranked = sorted(alerts, key=key)

    out = a.out or a.alerts
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for al in ranked:
            f.write(json.dumps(al, separators=(",", ":")) + "\n")
    ps = [al["ranking"]["pReal"] for al in ranked if al["ranking"]["pReal"] is not None]
    if ps:
        import numpy as np
        q = np.percentile(ps, [50, 90, 99])
        n_hi = sum(1 for p in ps if p >= 0.5)
        print(f"[rerank] {len(ranked)} alerts by P(real) (model fit on night {model.get('night')}): "
              f"median {q[0]:.3f} p90 {q[1]:.3f} p99 {q[2]:.3f} | P>=0.5: {n_hi} "
              f"| missing fields: {n_missing} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
