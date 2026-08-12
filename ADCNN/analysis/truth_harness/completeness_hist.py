#!/usr/bin/env python3
"""Binned completeness for ADCNN, the stack, and their union, on ONE injection set.

WHY THE MATCHING IS DONE HERE RATHER THAN READ FROM A TRUTH FILE. truth_stack.csv carries the SAME
detA_ok/detB_ok columns as truth_v2.csv -- it records the ADCNN detections, not the stack's. The stack
side exists only as a raw, unlabelled catalogue (stack_dets_inj.csv). So the only way to compare the
two systems is to run ONE matching procedure over BOTH catalogues, which is what this does. The
procedure is validated by reproducing truth_v2's own detA_ok/detB_ok for the ADCNN catalogue; if that
control fails, nothing below is trustworthy.

LABEL HYGIENE: an object counts as detected in an epoch only if a detection of the RIGHT visit lands
within `tol` of its true position in that epoch. "Detected in both epochs" is the ceiling any linker
can reach, so it is the honest denominator for an end-to-end comparison.
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

V = "outputs/runs/pa_validate"
# MATCHING TOLERANCE. A FIXED radius is wrong here, and the whole existing truth harness uses one.
#
# truth_v2/truth_v3's own detA_ok/detB_ok are reproduced exactly by a fixed 1.0" match (100.00% epoch
# A, 99.97% epoch B; 0.5" gives 58.7% vs 70.9%, 2.0" gives 76.1%). But a TRAIL is an extended source:
# at 0.2"/px a 60 px trail is 12" long, so a detection centroided anywhere but the exact middle sits
# up to 6" from the true centre and a 1" radius misses it. The bias therefore grows with trail length
# -- exactly the axis this comparison is about.
#
# MEASURED: switching to `1.0" + half the trail length` moves ADCNN both-epoch detection from 38.66%
# to 50.66%, and turns an apparent long-trail COLLAPSE (43.9% at 0-8 px down to 9.9% at 44-60 px) into
# a FLAT curve (44.9% -> 51.7%). The collapse was the matcher, not the detector.
#
# That widening is only legitimate if it is not buying chance matches, so it was checked with an
# OFFSET NULL (shift every true position 20-60", preserving density and field geometry, and re-match):
# chance accounts for 0.09% of the ADCNN signal and 0.23% of the stack's, rising to only 1.2% in the
# longest bin. The signal is real.
TOL_ARCSEC = 1.0                 # the legacy fixed convention, kept for the control only
SCALE_WITH_TRAIL = True          # add half the trail length; set False to reproduce the legacy numbers

# Trail length in px and first-epoch SNR. Bin edges follow the injection grid (7 discrete lengths)
# and the SNR band the detector is built for.
L_EDGES = [0, 8, 12, 16, 24, 32, 44, 60]
S_EDGES = [0, 3, 4, 5, 6, 8, 10, 99]


def _unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def detected(T, dets, tol_arcsec):
    """(detA, detB) boolean arrays: is there a detection of the right visit at the true position?"""
    tol_a = np.broadcast_to(np.asarray(tol_arcsec, float), (len(T),))
    trees = {int(v): cKDTree(_unit(g.ra, g.dec)) for v, g in dets.groupby("visit")}
    out = []
    for ep in ("A", "B"):
        ok = np.zeros(len(T), bool)
        for v, idx in T.groupby(f"visit{ep}").groups.items():
            tr = trees.get(int(v))
            if tr is None:
                continue
            sub = T.loc[idx]
            pos = T.index.get_indexer(idx)
            d, _ = tr.query(_unit(sub[f"ra{ep}"], sub[f"dec{ep}"]), k=1)
            ok[pos] = d < 2 * np.sin(np.radians(tol_a[pos] / 3600.0) / 2)
        out.append(ok)
    return out


def _table(T, arms, mask_col, edges, label):
    print(f"\n{label:>12} {'n':>6}" + "".join(f"{a:>12}" for a in arms))
    lo_all = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (T[mask_col] >= lo) & (T[mask_col] < hi)
        if m.sum() < 20:
            continue
        row = f"{lo:>5.0f}-{hi:<6.0f}{int(m.sum()):>6}"
        for a in arms:
            row += f"{100 * T.loc[m, a].mean():>11.1f}%"
        print(row)
        lo_all.append(m)
    row = f"{'ALL':>12}{len(T):>6}"
    for a in arms:
        row += f"{100 * T[a].mean():>11.1f}%"
    print(row)


def main():
    T = pd.read_csv(f"{V}/truth_v2.csv").reset_index(drop=True)
    adcnn = pd.read_csv(f"{V}/inj_dets_v2.csv", usecols=["ra", "dec", "visit"])
    stack = pd.read_csv(f"{V}/stack_dets_inj.csv", usecols=["ra", "dec", "visit"])
    print(f"injected {len(T):,} movers | adcnn dets {len(adcnn):,} | stack dets {len(stack):,}")

    # CONTROL FIRST, in the LEGACY convention: the procedure must reproduce the ADCNN truth table it
    # never saw. If it cannot recover the arm we already have, its stack numbers mean nothing.
    cA, cB = detected(T, adcnn, TOL_ARCSEC)
    tA = T.detA_ok.fillna(False).to_numpy(bool); tB = T.detB_ok.fillna(False).to_numpy(bool)
    print(f"\nCONTROL -- fixed {TOL_ARCSEC}\" matcher vs truth_v2's own ADCNN flags: "
          f"epochA agree {100*(cA==tA).mean():.2f}%  epochB agree {100*(cB==tB).mean():.2f}%")
    if (cA == tA).mean() < 0.97 or (cB == tB).mean() < 0.97:
        print("  *** matcher does NOT reproduce the known answer -- stop ***")
        return 1

    tol = TOL_ARCSEC + (T.L_px.to_numpy() * 0.2 / 2.0 if SCALE_WITH_TRAIL else 0.0)
    aA, aB = detected(T, adcnn, tol)
    sA, sB = detected(T, stack, tol)
    if SCALE_WITH_TRAIL:
        print(f"  using {TOL_ARCSEC}\" + half-trail (offset-null chance: 0.09% ADCNN / 0.23% stack). "
              f"Legacy fixed-{TOL_ARCSEC}\" ADCNN both-epoch would read "
              f"{100*(cA & cB).mean():.2f}% against {100*(aA & aB).mean():.2f}% here.")

    T["ADCNN"] = aA & aB
    T["stack"] = sA & sB                    # NB: bracket access -- `T.stack` is a DataFrame METHOD
    T["union"] = T["ADCNN"] | T["stack"]
    arms = ["ADCNN", "stack", "union"]
    print("\nBOTH-EPOCH DETECTION completeness -- the ceiling any linker can reach.")
    _table(T, arms, "L_px", L_EDGES, "trail px")
    _table(T, arms, "snr_t", S_EDGES, "SNR")
    A_, S_ = T["ADCNN"], T["stack"]
    print(f"\ncomplementarity: stack-only {int((S_ & ~A_).sum())}, "
          f"adcnn-only {int((A_ & ~S_).sum())}, both {int((A_ & S_).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
