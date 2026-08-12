#!/usr/bin/env python3
"""Binned both-epoch detection completeness for ADCNN, the stack, and their union, on ONE injection set.

WHY THE MATCHING IS DONE HERE RATHER THAN READ FROM A TRUTH FILE. truth_stack.csv carries the SAME
detA_ok/detB_ok columns as truth_v2.csv -- it records the ADCNN detections, not the stack's. The stack
side exists only as a raw, unlabelled catalogue. So the only way to compare the systems is to run ONE
procedure over BOTH catalogues.

HOW THE TOLERANCE WAS SETTLED, because it turned out to drive the answer.

  * A FIXED radius reproduces truth_v2's own flags exactly (100.00% epoch A / 99.97% epoch B at 1.0";
    0.5" gives 58.7% vs 70.9%, 2.0" gives 76.1%). So the legacy harness uses a fixed 1.0".
  * But a trail is an EXTENDED source: at 0.2"/px a 60 px trail is 12" long, so a detection centroided
    anywhere but the exact middle sits up to 6" from the centre and a 1" radius misses it. The bias
    grows with length -- precisely the axis this comparison is about. Under fixed 1.0", ADCNN appears
    to COLLAPSE from 43.9% (0-8 px) to 9.9% (44-60 px). That collapse is the matcher.
  * The fix is not a bigger radius, which is arbitrary and length-coupled. It is to ask the question
    directly: is there a detection ON THE TRAIL -- within `perp` of the segment joining its true
    endpoints? That has one genuine positional tolerance and no free length parameter.

VALIDATION. Offset null (shift every true position 20-60", preserving density and field geometry):
0.00% for both systems at both perp values, so nothing here is chance. Coverage parity checked: 20
visits and 165 detectors in both catalogues, 801 shared panels, 5 ADCNN-only panels carrying 0.48% of
its detections, and the stack has MORE detections per panel (480 vs 450) -- so the gap is not a
coverage artifact. Label hygiene: only 0.16% of injections have another injection inside their own
search radius (nearest-neighbour p1 = 9.7" against a max radius of 6.6").

SENSITIVITY, stated because the long-trail bins are where it matters. The overall ADCNN-minus-stack
gap is STABLE at 11.6-13.3 points across every rule tried (fixed 1"/2", half-trail, full-trail,
segment). The per-bin values at 44-60 px are NOT: ADCNN spans 9.9% (fixed 1") to 58.1% (full trail),
the stack 1.4% to 35.3%. Quote the shape and the gap; treat a single long-trail number as method-
dependent at the ~5-point level.
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

V = "outputs/runs/pa_validate"
PIX = 0.2
PERP_ARCSEC = 1.0
L_EDGES = [0, 8, 12, 16, 24, 32, 44, 60]
S_EDGES = [0, 3, 4, 5, 6, 8, 10, 99]


def _unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def radius_det(T, dets, tol_arcsec):
    """Legacy/cross-check matcher: nearest detection within `tol_arcsec` of the epoch position."""
    tol_a = np.broadcast_to(np.asarray(tol_arcsec, float), (len(T),))
    trees = {int(v): cKDTree(_unit(g.ra, g.dec)) for v, g in dets.groupby("visit")}
    out = []
    for ep in ("A", "B"):
        ok = np.zeros(len(T), bool)
        for v, idx in T.groupby(f"visit{ep}").groups.items():
            tr = trees.get(int(v))
            if tr is None:
                continue
            sub = T.loc[idx]; pos = T.index.get_indexer(idx)
            d, _ = tr.query(_unit(sub[f"ra{ep}"], sub[f"dec{ep}"]), k=1)
            ok[pos] = d < 2 * np.sin(np.radians(tol_a[pos] / 3600.0) / 2)
        out.append(ok)
    return np.logical_and(*out)


def segment_det(T, dets, perp_arcsec=PERP_ARCSEC, dra_off=0.0, ddec_off=0.0):
    """PRIMARY matcher: a detection within `perp_arcsec` of the true trail SEGMENT, in both epochs."""
    half = T.L_px.to_numpy() * PIX / 2.0
    pa = np.radians(T.pa.to_numpy())
    out = []
    for ep in ("A", "B"):
        ok = np.zeros(len(T), bool)
        ra0 = T[f"ra{ep}"].to_numpy() + dra_off / 3600.0 / np.cos(np.radians(T[f"dec{ep}"].to_numpy()))
        dec0 = T[f"dec{ep}"].to_numpy() + ddec_off / 3600.0
        for v, idx in T.groupby(f"visit{ep}").groups.items():
            g = dets[dets.visit == int(v)]
            if not len(g):
                continue
            cdg = np.cos(np.radians(g.dec.values))
            tr = cKDTree(np.c_[g.ra.values * cdg, g.dec.values])
            for p in T.index.get_indexer(idx):
                cd = np.cos(np.radians(dec0[p]))
                cx, cy = ra0[p] * cd * 3600.0, dec0[p] * 3600.0
                ux, uy = np.cos(pa[p]), np.sin(pa[p])
                cand = tr.query_ball_point([ra0[p] * cd, dec0[p]],
                                           (half[p] + perp_arcsec + 0.5) / 3600.0)
                if not cand:
                    continue
                gx = g.ra.values[cand] * cd * 3600.0; gy = g.dec.values[cand] * 3600.0
                t = np.clip((gx - cx) * ux + (gy - cy) * uy, -half[p], half[p])
                ok[p] = bool((np.hypot(gx - (cx + t * ux), gy - (cy + t * uy)) < perp_arcsec).any())
        out.append(ok)
    return np.logical_and(*out)


def _table(T, arms, col, edges, label):
    print(f"\n{label:>12} {'n':>6}" + "".join(f"{a:>12}" for a in arms))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((T[col] >= lo) & (T[col] < hi)).to_numpy()
        if m.sum() < 20:
            continue
        print(f"{lo:>5.0f}-{hi:<6.0f}{int(m.sum()):>6}"
              + "".join(f"{100 * arms[a][m].mean():>11.1f}%" for a in arms))
    print(f"{'ALL':>12}{len(T):>6}" + "".join(f"{100 * arms[a].mean():>11.1f}%" for a in arms))


def main():
    T = pd.read_csv(f"{V}/truth_v2.csv").reset_index(drop=True)
    A = pd.read_csv(f"{V}/inj_dets_v2.csv", usecols=["ra", "dec", "visit"])
    S = pd.read_csv(f"{V}/stack_dets_inj.csv", usecols=["ra", "dec", "visit"])
    print(f"injected {len(T):,} movers | adcnn dets {len(A):,} | stack dets {len(S):,}")

    # CONTROL: the legacy fixed-radius path must reproduce the ADCNN truth table it never saw.
    cA = radius_det(T, A, 1.0)
    legacy = (T.detA_ok.fillna(False) & T.detB_ok.fillna(False)).to_numpy(bool)
    agree = (cA == legacy).mean()
    print(f"CONTROL -- fixed 1.0\" reproduces truth_v2's own both-epoch flags: {100*agree:.2f}%")
    if agree < 0.97:
        print("  *** cannot recover the arm we already have -- stop ***")
        return 1

    a, s = segment_det(T, A), segment_det(T, S)
    an, sn = segment_det(T, A, dra_off=40.0), segment_det(T, S, dra_off=40.0)
    print(f"OFFSET NULL (40\" shift): ADCNN {an.mean():.2%}  stack {sn.mean():.2%}  -- chance, must be ~0")
    print(f"legacy fixed-1.0\" would report ADCNN {legacy.mean():.2%}; the trail-aware answer is {a.mean():.2%}")

    arms = {"ADCNN": a, "stack": s, "union": a | s}
    print(f"\nBOTH-EPOCH DETECTION completeness (segment match, perp={PERP_ARCSEC}\") "
          f"-- the ceiling any linker can reach.")
    _table(T, arms, "L_px", L_EDGES, "trail px")
    _table(T, arms, "snr_t", S_EDGES, "SNR")
    print(f"\ncomplementarity: stack-only {int((s & ~a).sum())}, adcnn-only {int((a & ~s).sum())}, "
          f"both {int((a & s).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
