#!/usr/bin/env python3
"""Flatten an alerts.jsonl into a one-row-per-alert CSV, for spreadsheets and quick joins.

The JSONL stays the record of truth (nested epochs, vetoes, prediction blocks survive only there);
this is the human-facing view: identity, tier, motion, the gate statistics the alert was admitted
on, and up to four epochs flattened as ep{K}_{mjd,ra,dec,snr,len}. Row order preserves the file's
rank order, and `rank` makes it explicit so a spreadsheet re-sort is recoverable.

    python -m ADCNN.qa.alerts_csv --alerts stream_1k/alerts.jsonl            # -> stream_1k/alerts.csv
    python -m ADCNN.qa.alerts_csv --alerts a.jsonl --out b.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import sys

MAX_EP = 4

FIELDS = ["rank", "alertId", "night", "tier", "status", "nEpochs", "arcMin",
          "rate_degday", "pa_deg", "rate_sigma_degday",
          "chi2", "chi2_source", "pReal", "priorityScore", "confidenceTier",
          "linRmsArcsec", "trailMotionDpaMaxDeg", "speedRatioMax",
          "score_min", "mfsnr_min", "trail_len_px_mean",
          "vetoStationary", "vetoTrain", "nStaticMembers", "pixelVet",
          "match_obj"]
EP_FIELDS = ["mjd", "ra", "dec", "snr", "len_px"]


def _g(a, *ks, d=None):
    x = a
    for k in ks:
        if not isinstance(x, dict):
            return d
        x = x.get(k)
    return d if x is None else x


def flatten(a, rank):
    tl = _g(a, "vetting", "trail_len_px", d=None) or []
    tl = [t for t in tl if t is not None]
    row = {
        "rank": rank,
        "alertId": a.get("alertId"),
        "night": a.get("night"),
        "tier": a.get("tier"),
        "status": a.get("status"),
        "nEpochs": a.get("nEpochs"),
        "arcMin": a.get("arcMin"),
        "rate_degday": _g(a, "motion", "rate_degday"),
        "pa_deg": _g(a, "motion", "pa_deg"),
        "rate_sigma_degday": _g(a, "motion", "rate_sigma_degday"),
        "chi2": _g(a, "orbit", "chi2"),
        "chi2_source": _g(a, "orbit", "chi2_source"),
        "pReal": _g(a, "ranking", "pReal"),
        "priorityScore": a.get("priorityScore"),
        "confidenceTier": a.get("confidenceTier"),
        "linRmsArcsec": _g(a, "geometry", "linRmsArcsec"),
        "trailMotionDpaMaxDeg": _g(a, "geometry", "trailMotionDpaMaxDeg"),
        "speedRatioMax": _g(a, "geometry", "speedRatioMax"),
        "score_min": _g(a, "vetting", "score_min"),
        "mfsnr_min": _g(a, "vetting", "mfsnr_min"),
        "trail_len_px_mean": (sum(tl) / len(tl)) if tl else None,
        "vetoStationary": _g(a, "stationarity", "vetoStationary"),
        "vetoTrain": _g(a, "trainVeto", "vetoTrain"),
        "nStaticMembers": _g(a, "staticVeto", "nStaticMembers"),
        "pixelVet": _g(a, "pixelVet", "verdict"),
        "match_obj": _g(a, "match", "obj") or "",
    }
    eps = a.get("epochs") or []
    for k in range(MAX_EP):
        e = eps[k] if k < len(eps) else {}
        row[f"ep{k}_mjd"] = e.get("mjd")
        row[f"ep{k}_ra"] = e.get("ra")
        row[f"ep{k}_dec"] = e.get("dec")
        row[f"ep{k}_snr"] = e.get("snr")
        row[f"ep{k}_len_px"] = e.get("trail_len_px")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--out", default=None, help="default: alongside --alerts as alerts.csv")
    a = ap.parse_args(argv)
    out = a.out or (a.alerts.rsplit(".", 1)[0] + ".csv")
    cols = FIELDS + [f"ep{k}_{f}" for k in range(MAX_EP) for f in EP_FIELDS]
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for line in open(a.alerts):
            w.writerow(flatten(json.loads(line), n))
            n += 1
    print(f"[alerts-csv] {n} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
