#!/usr/bin/env python3
"""Mine the FULL streams (below the 1k cut) for cross-night chains -- pairs, then triplets.

The delivered-product campaign (track_scan) answered "do two DELIVERED alerts link?" -- verdict
null. This digs below the budget: all ~46k stream alerts per the nine nights, because an object
delivered on one night is often budget-cut (or op-cut) on another; the per-night cut cannot see
that correlation. A 2-alert chain has 2 dof and needs the empirical null; a 3-alert chain
(>= 12 measurements vs 8 cubic params) is selective on its own -- that is what the user asked
for: "three detections in 2+ nights".

Stages (each null-calibrated by dec-offsetting every non-anchor night):
  1. cross-night PAIR scan, quadratic track through all epochs of both alerts;
  2. TRIPLET extension: every pair under EXTEND_RMS is extrapolated along its fitted track to
     the remaining nights; gated third alerts join a cubic re-fit of all >= 12 measurements.

Members are labeled delivered/sub-budget. Product rows land in
outputs/runs/multinight/substream_{pairs,triplets}.csv; nothing outside cross-night material.

    python -m ADCNN.analysis.multinight.mine_substream            # real + nulls, ~tens of min
"""
from __future__ import annotations
import collections
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "outputs/runs/10k_cadence"
OUT = REPO / "outputs/runs/multinight"
DT_PAIR = (0.5, 3.2)      # days between pair members
EXTEND_RMS = 3.0          # arcsec; pairs below this are tried for a third alert
EXTEND_DT = 3.2           # days; max extrapolation beyond the pair's nearest epoch
TRIPLET_TOL = 1.5         # deg; gate radius at the extrapolated third-night position
NULL_OFFSETS = (0.4, -0.4, 0.8, -0.8, 1.2, -1.2)


def load():
    """(night, alertId) -> (epochs, rate, pa, delivered?) for ALL stream alerts."""
    alerts = {}
    for d in sorted(glob.glob(str(RUNS / "run_night_*"))):
        night = d[-8:]
        delivered = {json.loads(l)["alertId"] for l in open(f"{d}/alerts.jsonl")}
        for a in map(json.loads, open(f"{d}/work/stream/alerts.jsonl")):
            m = a.get("motion") or {}
            if m.get("rate_degday") is None or m.get("pa_deg") is None:
                continue
            alerts[(night, a["alertId"])] = (
                [(e["mjd"], e["ra"], e["dec"]) for e in a["epochs"]],
                m["rate_degday"], m["pa_deg"], a["alertId"] in delivered)
    return alerts


def poly_rms(eps_groups, order):
    """RMS (arcsec) of one polynomial sky-track through every epoch of every group."""
    eps = sorted(e for g in eps_groups for e in g)
    t = np.array([e[0] for e in eps]); t = t - t.mean()
    dec0 = np.mean([e[2] for e in eps]); ra0 = eps[0][1]
    x = np.array([((e[1] - ra0 + 180) % 360 - 180) for e in eps]) * np.cos(np.radians(dec0)) * 3600
    y = np.array([e[2] - dec0 for e in eps]) * 3600
    A = np.vstack([t ** k for k in range(order + 1)]).T
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sqrt(np.mean(rx * rx + ry * ry)))


def pair_scan(alerts, off=0.0):
    """Quadratic pair scan; off shifts every LATER night's dec (the empirical null)."""
    bynight = collections.defaultdict(list)
    for k in alerts:
        bynight[k[0]].append(k)
    nights = sorted(bynight)
    pos = {n: np.array([[alerts[k][0][0][1], alerts[k][0][0][2], alerts[k][1]]
                        for k in bynight[n]]) for n in nights}
    out = []
    for i, nA in enumerate(nights):
        for nB in nights[i + 1:]:
            dt = alerts[bynight[nB][0]][0][0][0] - alerts[bynight[nA][0]][0][0][0]
            if not (DT_PAIR[0] < dt < DT_PAIR[1]):
                continue
            B = pos[nB].copy(); B[:, 1] += off
            Bk = bynight[nB]
            for kA in bynight[nA]:
                epsA, rate, pa, _ = alerts[kA]
                th = np.radians(pa)
                pra = epsA[0][1] + rate * dt * np.sin(th) / np.cos(np.radians(epsA[0][2]))
                pdec = epsA[0][2] + rate * dt * np.cos(th)
                tol = 1.0 + 0.35 * rate * dt
                d = np.hypot((B[:, 0] - pra) * np.cos(np.radians(pdec)), B[:, 1] - pdec)
                ok = (d < tol) & (B[:, 2] > rate / 2.2) & (B[:, 2] < rate * 2.2)
                for j in np.where(ok)[0]:
                    epsB = [(e[0], e[1], e[2] + off) for e in alerts[Bk[j]][0]]
                    out.append((poly_rms([epsA, epsB], 2), kA, Bk[j]))
    return sorted(out, key=lambda r: r[0])


def extend_pairs(alerts, pairs, off=0.0):
    """Try a third alert (any OTHER night) along each sub-EXTEND_RMS pair's quadratic track."""
    bynight = collections.defaultdict(list)
    for k in alerts:
        bynight[k[0]].append(k)
    trips = []
    for rms2, kA, kB in pairs:
        if rms2 >= EXTEND_RMS:
            break
        epsA = alerts[kA][0]
        epsB = [(e[0], e[1], e[2] + off) for e in alerts[kB][0]]
        eps = sorted(epsA + epsB)
        t0 = np.mean([e[0] for e in eps])
        t = np.array([e[0] for e in eps]) - t0
        dec0 = np.mean([e[2] for e in eps]); ra0 = eps[0][1]
        x = np.array([((e[1] - ra0 + 180) % 360 - 180) for e in eps]) * np.cos(np.radians(dec0))
        y = np.array([e[2] - dec0 for e in eps])
        A = np.vstack([np.ones_like(t), t, t * t]).T
        cx = np.linalg.lstsq(A, x, rcond=None)[0]
        cy = np.linalg.lstsq(A, y, rcond=None)[0]
        for nC in bynight:
            if nC in (kA[0], kB[0]):
                continue
            tC = alerts[bynight[nC][0]][0][0][0] - t0
            if min(abs(tC - t.min()), abs(tC - t.max())) > EXTEND_DT:
                continue
            pxc = cx[0] + cx[1] * tC + cx[2] * tC * tC
            pyc = cy[0] + cy[1] * tC + cy[2] * tC * tC
            pra = ra0 + pxc / np.cos(np.radians(dec0)); pdec = dec0 + pyc
            offC = off if nC > kA[0] else 0.0          # nulls shift nights AFTER the anchor
            C = np.array([[alerts[k][0][0][1], alerts[k][0][0][2] + offC] for k in bynight[nC]])
            d = np.hypot((C[:, 0] - pra) * np.cos(np.radians(pdec)), C[:, 1] - pdec)
            for j in np.where(d < TRIPLET_TOL)[0]:
                kC = bynight[nC][j]
                epsC = [(e[0], e[1], e[2] + offC) for e in alerts[kC][0]]
                r3 = poly_rms([epsA, epsB, epsC], 3)
                if r3 < 30:
                    trips.append((r3, rms2, kA, kB, kC))
    return sorted(trips, key=lambda r: r[0])


def main(argv=None):
    alerts = load()
    n_del = sum(1 for v in alerts.values() if v[3])
    print(f"[mine] {len(alerts)} stream alerts loaded ({n_del} delivered, "
          f"{len(alerts) - n_del} sub-budget)", flush=True)

    real_pairs = pair_scan(alerts)
    print(f"[mine] REAL pairs: {len(real_pairs)} gated; "
          f"<3\" {sum(1 for r, *_ in real_pairs if r < 3)}, "
          f"<0.5\" {sum(1 for r, *_ in real_pairs if r < 0.5)}, "
          f"min {real_pairs[0][0]:.3f}\"" if real_pairs else "none", flush=True)
    real_trips = extend_pairs(alerts, real_pairs)
    print(f"[mine] REAL triplets: {len(real_trips)}; "
          f"<1\" {sum(1 for r, *_ in real_trips if r < 1)}, "
          f"<0.5\" {sum(1 for r, *_ in real_trips if r < 0.5)}", flush=True)

    nulls = []
    for off in NULL_OFFSETS:
        np_ = pair_scan(alerts, off)
        nt_ = extend_pairs(alerts, np_, off)
        nulls.append({"off": off, "pairs": len(np_),
                      "p_lt05": sum(1 for r, *_ in np_ if r < 0.5),
                      "p_min": round(np_[0][0], 3) if np_ else None,
                      "trips": len(nt_),
                      "t_lt1": sum(1 for r, *_ in nt_ if r < 1),
                      "t_lt05": sum(1 for r, *_ in nt_ if r < 0.5),
                      "t_min": round(nt_[0][0], 3) if nt_ else None})
        print(f"[mine] null {off:+.1f}: pairs<0.5\" {nulls[-1]['p_lt05']} (min {nulls[-1]['p_min']}) "
              f"trips<1\" {nulls[-1]['t_lt1']} (min {nulls[-1]['t_min']})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    def _lab(k):
        return f"{k[0]}:{k[1]}{'*' if alerts[k][3] else ''}"     # * = delivered
    with open(OUT / "substream_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rms_arcsec", "A", "B", "rateA", "rateB", "n_delivered_members"])
        for r, a, b in real_pairs:
            if r < 0.5:
                w.writerow([f"{r:.3f}", _lab(a), _lab(b),
                            f"{alerts[a][1]:.2f}", f"{alerts[b][1]:.2f}",
                            int(alerts[a][3]) + int(alerts[b][3])])
    with open(OUT / "substream_triplets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rms3_arcsec", "pair_rms", "A", "B", "C", "n_nights", "n_delivered_members"])
        for r3, r2, a, b, c in real_trips:
            if r3 < 5:
                w.writerow([f"{r3:.3f}", f"{r2:.3f}", _lab(a), _lab(b), _lab(c),
                            len({a[0], b[0], c[0]}),
                            sum(int(alerts[k][3]) for k in (a, b, c))])
    (OUT / "work" / "substream_summary.json").write_text(json.dumps({
        "n_stream_alerts": len(alerts), "n_delivered": n_del,
        "real": {"pairs": len(real_pairs),
                 "p_lt05": sum(1 for r, *_ in real_pairs if r < 0.5),
                 "p_min": round(real_pairs[0][0], 3) if real_pairs else None,
                 "trips": len(real_trips),
                 "t_lt1": sum(1 for r, *_ in real_trips if r < 1),
                 "t_lt05": sum(1 for r, *_ in real_trips if r < 0.5),
                 "t_min": round(real_trips[0][0], 3) if real_trips else None},
        "nulls": nulls}, indent=2))
    print(f"[mine] wrote substream_pairs.csv / substream_triplets.csv -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
