#!/usr/bin/env python3
"""Cross-night quadratic-track scan of the delivered products + its empirical null.

The heliolinx faithful arm (run_multinight.py) answers "do two delivered alerts link under a
heliocentric hypothesis with r >= 1.1 AU"; its grids do not cover the very-close geometry our
1-8 deg/day population implies (only 2 of 6,028 faithful tracklets map to a physical state).
This scan is the geometry-complete complement: for every cross-night alert pair that survives a
coarse motion-extrapolation gate, fit ONE quadratic sky-track through ALL epochs of both alerts
(8 measurements, 6 params for 2v+2v) and score the astrometric RMS. A real object gives ~0.1-0.5"
(the astrometry); a chance pair, arcmin+. Significance comes from an EMPIRICAL NULL: the same
scan with every night-B position offset in dec (kills real coincidences, preserves densities).

MEASURED on the nine delivered nights (2026-08-16): 88,352 gated pairs; 39 under 3" vs null
36-50 (chance-consistent); n<0.5": real 3 vs null 0-2 in 13 draws (none reached 3); tightest
real pair 0.125" vs null minima 0.059-0.583" (1 of 13 nulls tighter). The single standout is
BELOW_THRESHOLD (~8% chance probability), written to candidates.csv rather than discarded.

    python -m ADCNN.analysis.multinight.track_scan          # writes outputs/runs/multinight/
"""
from __future__ import annotations
import collections
import csv
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "outputs/runs/10k_cadence"
OUT = REPO / "outputs/runs/multinight"
NULL_OFFSETS = (0.4, -0.4, 0.8, 0.6, -0.6, 1.0, -1.0, 1.2, -1.2, 1.6, -1.6, 2.0, -2.0)
RMS_FLAG = 0.5          # arcsec; below this a pair is worth a human look
DT_RANGE = (0.5, 3.2)   # days; beyond ~3 nights the coarse linear gate is meaningless at 8 deg/day


def load():
    alerts, recs = {}, {}
    for d in sorted(glob.glob(str(RUNS / "run_night_*"))):
        for a in map(json.loads, open(f"{d}/alerts.jsonl")):
            m = a.get("motion") or {}
            if m.get("rate_degday") is None or m.get("pa_deg") is None:
                continue
            k = (d[-8:], a["alertId"])
            alerts[k] = ([(e["mjd"], e["ra"], e["dec"]) for e in a["epochs"]],
                         m["rate_degday"], m["pa_deg"])
            recs[k] = a
    return alerts, recs


def track_rms(epsA, epsB):
    eps = sorted(epsA + epsB)
    t = np.array([e[0] for e in eps]); t = t - t.mean()
    dec0 = np.mean([e[2] for e in eps]); ra0 = eps[0][1]
    x = np.array([((e[1] - ra0 + 180) % 360 - 180) for e in eps]) * np.cos(np.radians(dec0)) * 3600
    y = np.array([e[2] - dec0 for e in eps]) * 3600
    A = np.vstack([np.ones_like(t), t, t * t]).T
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sqrt(np.mean(rx * rx + ry * ry)))


def scan(alerts, off=0.0):
    bynight = collections.defaultdict(list)
    for k in alerts:
        bynight[k[0]].append(k)
    nights = sorted(bynight)
    out = []
    for i, nA in enumerate(nights):
        for nB in nights[i + 1:]:
            dt = alerts[bynight[nB][0]][0][0][0] - alerts[bynight[nA][0]][0][0][0]
            if not (DT_RANGE[0] < dt < DT_RANGE[1]):
                continue
            Bk = bynight[nB]
            Bpos = np.array([[alerts[k][0][0][1], alerts[k][0][0][2] + off, alerts[k][1]]
                             for k in Bk])
            for kA in bynight[nA]:
                epsA, rate, pa = alerts[kA]
                th = np.radians(pa)
                pra = epsA[0][1] + rate * dt * np.sin(th) / np.cos(np.radians(epsA[0][2]))
                pdec = epsA[0][2] + rate * dt * np.cos(th)
                tol = 1.0 + 0.35 * rate * dt
                d = np.hypot((Bpos[:, 0] - pra) * np.cos(np.radians(pdec)), Bpos[:, 1] - pdec)
                ok = (d < tol) & (Bpos[:, 2] > rate / 2.2) & (Bpos[:, 2] < rate * 2.2)
                for j in np.where(ok)[0]:
                    epsB = [(e[0], e[1], e[2] + off) for e in alerts[Bk[j]][0]]
                    out.append((track_rms(epsA, epsB), kA, Bk[j]))
    return sorted(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alerts, recs = load()
    print(f"[track_scan] {len(alerts)} delivered alerts with motion")
    real = scan(alerts)
    print(f"[track_scan] REAL: {len(real)} gated pairs, "
          f"n<3\"={sum(1 for r, *_ in real if r < 3)}, n<0.5\"={sum(1 for r, *_ in real if r < 0.5)}")
    nulls = []
    for off in NULL_OFFSETS:
        n = scan(alerts, off)
        nulls.append({"offset_deg": off, "n_pairs": len(n),
                      "n_lt3": sum(1 for r, *_ in n if r < 3),
                      "n_lt05": sum(1 for r, *_ in n if r < 0.5),
                      "min_rms": round(n[0][0], 3) if n else None})
        print(f"[track_scan] null {off:+.1f}: min {nulls[-1]['min_rms']}\" n<0.5\" {nulls[-1]['n_lt05']}")

    # the product: ONLY cross-night material, flagged pairs first, with the null beside them
    flagged = [(r, a, b) for r, a, b in real if r < RMS_FLAG]
    n_tighter = sum(1 for x in nulls if x["min_rms"] is not None and flagged
                    and x["min_rms"] < flagged[0][0])
    with open(OUT / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "status", "track_rms_arcsec", "nightA", "alertA", "rateA_degday",
                    "paA_deg", "pRealA", "nightB", "alertB", "rateB_degday", "paB_deg", "pRealB",
                    "null_prob_note"])
        for i, (r, a, b) in enumerate(flagged):
            ra_, rb_ = recs[a], recs[b]
            w.writerow([i, "BELOW_THRESHOLD", f"{r:.3f}",
                        a[0], a[1], f"{ra_['motion']['rate_degday']:.2f}",
                        f"{ra_['motion']['pa_deg']:.1f}",
                        f"{(ra_.get('ranking') or {}).get('pReal', float('nan')):.2f}",
                        b[0], b[1], f"{rb_['motion']['rate_degday']:.2f}",
                        f"{rb_['motion']['pa_deg']:.1f}",
                        f"{(rb_.get('ranking') or {}).get('pReal', float('nan')):.2f}",
                        f"{n_tighter}/{len(nulls)} nulls produced a tighter pair"])
    for i, (r, a, b) in enumerate(flagged):
        cd = OUT / "candidates" / f"cand_{i:03d}"
        cd.mkdir(parents=True, exist_ok=True)
        (cd / "members.json").write_text(json.dumps(
            {"track_rms_arcsec": r, "A": recs[a], "B": recs[b]}, indent=2))
        for night, aid in (a, b):
            for p in glob.glob(str(RUNS / f"run_night_{night}" / "pairs" / f"*_{aid}_*.png")):
                shutil.copy2(p, cd / Path(p).name)
    (OUT / "work").mkdir(exist_ok=True)
    (OUT / "work" / "track_scan_summary.json").write_text(json.dumps({
        "n_alerts": len(alerts), "n_gated_pairs": len(real),
        "real_lt3": sum(1 for r, *_ in real if r < 3),
        "real_lt05": sum(1 for r, *_ in real if r < 0.5),
        "real_min": round(real[0][0], 3) if real else None,
        "flag_threshold_arcsec": RMS_FLAG, "nulls": nulls}, indent=2))
    print(f"[track_scan] {len(flagged)} flagged pair(s) -> {OUT}/candidates.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
