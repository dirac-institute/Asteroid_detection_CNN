#!/usr/bin/env python3
"""Summarize the mfsnr alert sweep: per (run, config, night) alert load, known-object recovery,
unknown-candidate burden, priorityScore/mfsnr/chi2 distributions, and follow-up search radii.
Feeds ALERT_SWEEP_DECISION.md. Reads the schema-1.1 alerts.jsonl files written by trail_state_link."""
import json, glob, sys
import numpy as np
import pandas as pd


def load_alerts(path):
    rows = []
    for line in open(path):
        a = json.loads(line)
        pred = {p["dt_min"]: p for p in a.get("predict", [])}
        rows.append(dict(
            night=a["night"], status=a["status"], tier=a["tier"],
            priority=a["priority"], priorityScore=a.get("priorityScore", np.nan),
            nEpochs=a["nEpochs"], chi2=a["orbit"]["chi2"],
            score_min=(a.get("vetting") or {}).get("score_min"),
            mfsnr_min=(a.get("vetting") or {}).get("mfsnr_min"),
            match_obj=(a.get("match") or {}).get("obj"),
            rate=a["motion"]["rate_degday"],
            sr30=pred.get(30.0, {}).get("search_radius_arcsec"),
            sr60=pred.get(60.0, {}).get("search_radius_arcsec"),
            sr90=pred.get(90.0, {}).get("search_radius_arcsec"),
        ))
    return pd.DataFrame(rows)


def main():
    runs = sorted(glob.glob("alert_sweep/*/alerts.jsonl"))
    if not runs:
        print("no alerts.jsonl under alert_sweep/*/"); sys.exit(1)
    out = []
    for path in runs:
        cfg = path.split("/")[1]
        df = load_alerts(path)
        if not len(df):
            out.append(dict(config=cfg, night="-", n_alerts=0)); continue
        for night, g in df.groupby("night"):
            new2 = g[(g.status == "NEW") & (g.tier == "2visit")]
            out.append(dict(
                config=cfg, night=night, n_alerts=len(g),
                n_2v_new=len(new2),
                n_3v=int((g.tier == "3+visit").sum()),
                n_known=int(g.match_obj.notna().sum()),
                pscore_p50=round(float(g.priorityScore.median()), 3),
                pscore_p90=round(float(g.priorityScore.quantile(0.9)), 3),
                chi2_p50=round(float(g.chi2.median()), 2) if g.chi2.notna().any() else None,
                mfsnr_p50=round(float(g.mfsnr_min.median()), 1) if g.mfsnr_min.notna().any() else None,
                sr30_p50=round(float(g.sr30.median()), 1) if g.sr30.notna().any() else None,
                sr90_p50=round(float(g.sr90.median()), 1) if g.sr90.notna().any() else None,
            ))
    res = pd.DataFrame(out)
    res.to_csv("alert_sweep/summary.csv", index=False)
    print(res.to_string(index=False))
    # config-level rollup (alert burden per night is THE follow-up budget number)
    print("\n=== per-config rollup (mean per night) ===")
    r = res[res.night != "-"]
    if len(r):
        roll = r.groupby("config").agg(nights=("night", "nunique"), alerts_pn=("n_alerts", "mean"),
                                       new2_pn=("n_2v_new", "mean"), known_total=("n_known", "sum"))
        print(roll.round(1).to_string())


if __name__ == "__main__":
    main()
