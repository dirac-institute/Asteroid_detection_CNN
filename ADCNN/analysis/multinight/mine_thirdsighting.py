#!/usr/bin/env python3
"""Close the two holes left by mine_substream's triplet search.

HOLE 1 (span): the triplet extension capped extrapolation at 3.2 d, so triples bridging wider
gaps -- 0706+0710+0713, anything across the 0630->0705 gap -- were never tried. Stage A extends
every sub-3" pair to EVERY other night with a distance-scaled gate.

HOLE 2 (detection-level): a night where the object yielded ONE detection produces no alert at
all, so an alert-only search cannot see it -- yet that single sits in work/dets_merged.csv.
Stage B takes every sub-1" cross-night pair and hunts the OTHER nights' detection tables for a
single detection that (a) joins a cubic re-fit of all 10 measurements at astrometric residuals
and (b) carries a trail LENGTH consistent with the track's rate at that instant (len_db is
frame-independent; trail beta is image-frame and deliberately NOT used -- see
image-frame-vs-sky-PA in memory). Interpolated configurations (single BETWEEN the pair's
nights) are geometrically far stronger than extrapolated ones and are labeled.

Both stages run identically on dec-offset nulls. Products (cross-night material only):
outputs/runs/multinight/thirdsighting_{triplets,singles}.csv + work/thirdsighting_summary.json.

    python -m ADCNN.analysis.multinight.mine_thirdsighting
"""
from __future__ import annotations
import collections
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.analysis.multinight.mine_substream import load as load_alerts, poly_rms, pair_scan

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "outputs/runs/10k_cadence"
OUT = REPO / "outputs/runs/multinight"
NULL_OFFSETS = (0.5, -0.5, 1.0, -1.0)
PAIR_SEED_A = 3.0        # arcsec; pairs below this feed the extended-span triplet stage
PAIR_SEED_B = 1.0        # arcsec; pairs below this feed the detection-level stage
SINGLE_FIT_RMS = 0.5     # arcsec; QUADRATIC through pair + single must fit to this
SINGLE_WINDOW = 2.5      # days beyond the pair span a single may sit (quadratic validity)
TRIP_FIT_RMS = 5.0       # arcsec; report ceiling for stage A (nulls decide significance)
PX_DEGDAY = 0.2 / 3600.0 * 2880.0   # trail px (30 s exposure) -> deg/day: len_db * this


def load_dets():
    """night -> (mjd, ra, dec, len_db) arrays of plausible mover detections."""
    out = {}
    for d in sorted(glob.glob(str(RUNS / "run_night_*"))):
        f = f"{d}/work/dets_merged.csv"
        t = pd.read_csv(f, usecols=["mjd", "ra", "dec", "score", "len_db", "src", "art_frac"],
                        low_memory=False)
        keep = (((t.score >= 0.68) | (t.src.astype(str) == "stack"))
                & (t.len_db >= 5.0) & (t.art_frac.fillna(0) < 0.5))
        t = t[keep]
        out[d[-8:]] = (t.mjd.to_numpy(float), t.ra.to_numpy(float),
                       t.dec.to_numpy(float), t.len_db.to_numpy(float))
        print(f"[3rd] {d[-8:]}: {keep.sum():,} candidate detections kept", flush=True)
    return out


def _quad_coeffs(epsA, epsB):
    eps = sorted(epsA + epsB)
    t0 = np.mean([e[0] for e in eps])
    t = np.array([e[0] for e in eps]) - t0
    dec0 = np.mean([e[2] for e in eps]); ra0 = eps[0][1]
    x = np.array([((e[1] - ra0 + 180) % 360 - 180) for e in eps]) * np.cos(np.radians(dec0))
    y = np.array([e[2] - dec0 for e in eps])
    A = np.vstack([np.ones_like(t), t, t * t]).T
    cx = np.linalg.lstsq(A, x, rcond=None)[0]
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    return t0, ra0, dec0, cx, cy, (t.min(), t.max())


def stage_a(alerts, pairs, off=0.0):
    """Extended-span alert triplets: no gap cap, distance-scaled gate, cubic fit."""
    bynight = collections.defaultdict(list)
    for k in alerts:
        bynight[k[0]].append(k)
    trips = []
    for rms2, kA, kB in pairs:
        if rms2 >= PAIR_SEED_A:
            break
        epsA = alerts[kA][0]
        epsB = [(e[0], e[1], e[2] + off) for e in alerts[kB][0]]
        t0, ra0, dec0, cx, cy, (tlo, thi) = _quad_coeffs(epsA, epsB)
        rate = alerts[kA][1]
        for nC in bynight:
            if nC in (kA[0], kB[0]):
                continue
            tC = alerts[bynight[nC][0]][0][0][0] - t0
            dtx = max(0.0, tlo - tC, tC - thi)          # 0 when interpolating
            tol = min(4.0, 0.3 + (0.15 + 0.05 * rate) * dtx)
            pra = ra0 + (cx[0] + cx[1] * tC + cx[2] * tC * tC) / np.cos(np.radians(dec0))
            pdec = dec0 + cy[0] + cy[1] * tC + cy[2] * tC * tC
            offC = off if nC > kA[0] else 0.0
            C = np.array([[alerts[k][0][0][1], alerts[k][0][0][2] + offC] for k in bynight[nC]])
            d = np.hypot((C[:, 0] - pra) * np.cos(np.radians(pdec)), C[:, 1] - pdec)
            for j in np.where(d < tol)[0]:
                kC = bynight[nC][j]
                epsC = [(e[0], e[1], e[2] + offC) for e in alerts[kC][0]]
                r3 = poly_rms([epsA, epsB, epsC], 3)
                if r3 < TRIP_FIT_RMS:
                    trips.append((r3, rms2, kA, kB, kC, dtx == 0.0))
    return sorted(trips, key=lambda r: r[0])


def stage_b(alerts, dets, pairs, off=0.0):
    """Detection-level third sighting for every tight pair."""
    hits = []
    for rms2, kA, kB in pairs:
        if rms2 >= PAIR_SEED_B:
            break
        epsA = alerts[kA][0]
        epsB = [(e[0], e[1], e[2] + off) for e in alerts[kB][0]]
        t0, ra0, dec0, cx, cy, (tlo, thi) = _quad_coeffs(epsA, epsB)
        rate = alerts[kA][1]
        for nC, (mj, ra, dec, ldb) in dets.items():
            if nC in (kA[0], kB[0]):
                continue
            tC = mj - t0
            if np.median(tC) < tlo - SINGLE_WINDOW or np.median(tC) > thi + SINGLE_WINDOW:
                continue
            dtx = np.maximum(0.0, np.maximum(tlo - tC, tC - thi))
            tol = np.minimum(4.0, 0.3 + (0.15 + 0.05 * rate) * dtx)
            offC = off if nC > kA[0] else 0.0
            pra = ra0 + (cx[0] + cx[1] * tC + cx[2] * tC * tC) / np.cos(np.radians(dec0))
            pdec = dec0 + cy[0] + cy[1] * tC + cy[2] * tC * tC
            d = np.hypot((ra - pra) * np.cos(np.radians(pdec)), (dec + offC) - pdec)
            # trail length must be consistent with the track's instantaneous rate
            trate = np.hypot(cx[1] + 2 * cx[2] * tC, cy[1] + 2 * cy[2] * tC)
            lrate = ldb * PX_DEGDAY
            ok = (d < tol) & (lrate > trate / 2.5) & (lrate < trate * 2.5)
            for j in np.where(ok)[0]:
                single = [(float(mj[j]), float(ra[j]), float(dec[j] + offC))]
                # QUADRATIC, not cubic: 10 measurements vs 6 params = 4 dof. The cubic left 2 dof
                # and MEASURED zero discrimination (real 40,693 hits vs null 22k-48k -- the fit
                # absorbed everything). Quadratic is adequate for real movers over <=3-day spans
                # (delivered pairs fit at 0.1-0.5") and is what makes a single detection count.
                r = poly_rms([epsA, epsB, single], 2)
                if r < SINGLE_FIT_RMS:
                    interp = tlo <= tC[j] <= thi
                    hits.append((r, rms2, kA, kB, nC, float(mj[j]), float(ra[j]),
                                 float(dec[j]), float(ldb[j]), bool(interp)))
    return sorted(hits, key=lambda r: r[0])


def main(argv=None):
    alerts = load_alerts()
    print(f"[3rd] {len(alerts)} stream alerts", flush=True)
    dets = load_dets()

    real_pairs = pair_scan(alerts)
    print(f"[3rd] pairs gated {len(real_pairs)}", flush=True)
    ra_ = stage_a(alerts, real_pairs)
    rb_ = stage_b(alerts, dets, real_pairs)
    print(f"[3rd] REAL: stageA trips {len(ra_)} (<2\" {sum(1 for r, *_ in ra_ if r < 2)}, "
          f"min {ra_[0][0]:.2f}\" )" if ra_ else "[3rd] REAL: stageA none", flush=True)
    print(f"[3rd] REAL: stageB singles {len(rb_)} "
          f"(interp {sum(1 for h in rb_ if h[9])}, min {rb_[0][0]:.2f}\")" if rb_
          else "[3rd] REAL: stageB none", flush=True)

    nulls = []
    for offv in NULL_OFFSETS:
        np_ = pair_scan(alerts, offv)
        na = stage_a(alerts, np_, offv)
        nb = stage_b(alerts, dets, np_, offv)
        nulls.append({"off": offv,
                      "a_n": len(na), "a_lt2": sum(1 for r, *_ in na if r < 2),
                      "a_min": round(na[0][0], 3) if na else None,
                      "b_n": len(nb), "b_interp": sum(1 for h in nb if h[9]),
                      "b_min": round(nb[0][0], 3) if nb else None})
        print(f"[3rd] null {offv:+.1f}: A {nulls[-1]['a_n']} (<2\" {nulls[-1]['a_lt2']}) "
              f"B {nulls[-1]['b_n']} (interp {nulls[-1]['b_interp']})", flush=True)

    def _lab(k):
        return f"{k[0]}:{k[1]}{'*' if alerts[k][3] else ''}"
    with open(OUT / "thirdsighting_triplets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rms3_arcsec", "pair_rms", "A", "B", "C", "interpolated", "n_nights"])
        for r3, r2, a, b, c, interp in ra_:
            w.writerow([f"{r3:.3f}", f"{r2:.3f}", _lab(a), _lab(b), _lab(c),
                        int(interp), len({a[0], b[0], c[0]})])
    with open(OUT / "thirdsighting_singles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rms_arcsec", "pair_rms", "A", "B", "night_single", "mjd", "ra", "dec",
                    "len_db_px", "interpolated"])
        for r, r2, a, b, nC, mj, ra, dec, ldb, interp in rb_:
            w.writerow([f"{r:.3f}", f"{r2:.3f}", _lab(a), _lab(b), nC,
                        f"{mj:.5f}", f"{ra:.6f}", f"{dec:.6f}", f"{ldb:.1f}", int(interp)])
    (OUT / "work" / "thirdsighting_summary.json").write_text(json.dumps({
        "real": {"a_n": len(ra_), "a_lt2": sum(1 for r, *_ in ra_ if r < 2),
                 "a_min": round(ra_[0][0], 3) if ra_ else None,
                 "b_n": len(rb_), "b_interp": sum(1 for h in rb_ if h[9]),
                 "b_min": round(rb_[0][0], 3) if rb_ else None},
        "nulls": nulls}, indent=2))
    print(f"[3rd] wrote thirdsighting_triplets.csv / thirdsighting_singles.csv -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
