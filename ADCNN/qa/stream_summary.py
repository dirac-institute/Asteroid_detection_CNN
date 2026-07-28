#!/usr/bin/env python3
"""Per-night summary statistics for the alert stream -- the night-over-night trend record.

The contact sheets show WHAT a given night's alerts look like; this shows whether the night
itself is normal. A stream that silently doubles in size, shifts its score distribution, or
tips into one veto class is a pipeline symptom (new artifact family, calibration drift, bad
template) that no single stamp reveals. One small JSON per night, designed to be diffed or
concatenated across nights.

Usage: python -m ADCNN.qa.stream_summary --alerts stream/alerts.jsonl --out stream/stream_summary.json
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np


def _q(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {}
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def static_coverage(alerts, static_catalog, radius_arcmin=1.0):
    """Fraction of alerts sitting on sky where the DRP static catalogue actually has sources.

    The static veto -- the strongest FP discriminator, ~19% of alerts flagged on a covered night --
    is silently a NO-OP wherever DRP coadds do not reach. On 20260629 the catalogue spanned
    RA 214-229 while the night's alerts spanned RA 8-359: 0.1% coverage, so the dominant false
    class (bright template residuals) went entirely unvetoed and only 8 alerts were flagged. That
    is invisible in the class counts alone -- they just look clean -- so record it explicitly.
    The chance-link calibration also has k_uncovered ~10x k_covered (fpp_2v_chance.json)."""
    import numpy as np
    try:
        import pandas as pd
        from scipy.spatial import cKDTree
        sc = (pd.read_parquet(static_catalog) if str(static_catalog).endswith(".parquet")
              else pd.read_csv(static_catalog))
    except Exception as e:
        return {"error": str(e)[:120]}
    def _u(ra, dec):
        r, d = np.radians(ra), np.radians(dec)
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    if not len(sc):
        return {"covered_fraction": 0.0, "n_static": 0}
    t = cKDTree(_u(sc.ra.to_numpy(), sc.dec.to_numpy()))
    pos = np.array([[float(a["epochs"][0]["ra"]), float(a["epochs"][0]["dec"])] for a in alerts])
    rad = 2 * np.sin(np.radians(radius_arcmin / 60.0) / 2)
    n = t.query_ball_point(_u(pos[:, 0], pos[:, 1]), r=rad, return_length=True)
    return {"covered_fraction": round(float((n > 0).mean()), 4), "n_static": int(len(sc)),
            "radius_arcmin": radius_arcmin,
            "_note": "static veto is a NO-OP off DRP coverage; low fraction => the dominant "
                     "template-residual FP class is UNVETOED and fpp k_uncovered (~10x) applies"}


def summarize(alerts_path, static_catalog=None):
    from ADCNN.qa.alert_report import classify
    alerts = [json.loads(l) for l in open(alerts_path)] if os.path.getsize(alerts_path) else []
    n = len(alerts)
    out = {"n_alerts": n, "alerts": os.path.abspath(alerts_path)}
    if not n:
        return out

    cls, tier, status, nep = {}, {}, {}, {}
    chi2, rate, smin, mfsnr, arc, dt = [], [], [], [], [], []
    known = 0
    for al in alerts:
        c = classify(al)[0]
        cls[c] = cls.get(c, 0) + 1
        tier[al.get("confidenceTier", "?")] = tier.get(al.get("confidenceTier", "?"), 0) + 1
        status[al.get("status", "?")] = status.get(al.get("status", "?"), 0) + 1
        k = str(al.get("nEpochs", "?"))
        nep[k] = nep.get(k, 0) + 1
        chi2.append((al.get("orbit") or {}).get("chi2"))
        rate.append((al.get("motion") or {}).get("rate_degday"))
        v = al.get("vetting") or {}
        smin.append(v.get("score_min"))
        mfsnr.append(v.get("mfsnr_min"))
        arc.append(al.get("arcMin"))
        if (al.get("match") or {}).get("obj"):
            known += 1
    out.update({
        "night": alerts[0].get("night"),
        "schema": alerts[0].get("schema"),
        "by_class": cls, "by_tier": tier, "by_status": status, "by_n_epochs": nep,
        "n_known_match": known,
        "chi2": _q(chi2), "rate_degday": _q(rate), "score_min": _q(smin),
        "mfsnr_min": _q(mfsnr), "arc_min": _q(arc),
        "clean_fraction": round(cls.get("CLEAN", 0) / n, 4),
        "static_coverage": (static_coverage(alerts, static_catalog) if static_catalog else None),
        "multi_epoch_fraction": round(sum(v for k, v in nep.items() if k.isdigit() and int(k) >= 3) / n, 4),
    })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--static-catalog", default=None,
                    help="report what fraction of alerts sit on DRP-covered sky (the static veto "
                         "is a silent no-op elsewhere)")
    a = ap.parse_args(argv)
    s = summarize(a.alerts, a.static_catalog)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(s, f, indent=2)
    print(f"[stream-summary] {s['n_alerts']} alerts | classes {s.get('by_class')} | "
          f"chi2 med {s.get('chi2', {}).get('p50')} | rate med {s.get('rate_degday', {}).get('p50')} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
