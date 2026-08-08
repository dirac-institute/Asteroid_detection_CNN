#!/usr/bin/env python3
"""Merge the STACK's DIA sources into the ADCNN detection catalogue. ALWAYS ON in production.

The two detectors are complementary, measured against injected truth (3,857 movers, rate >1 deg/day,
SNR 2-10; see outputs/runs/pa_validate and the adcnn-vs-stack-truth note):

    DETECTION   ADCNN 38.6%   stack 39.1%   -- a TIE; 9.2% ADCNN-only, 9.7% stack-only
    END-TO-END  ADCNN 18.8%   stack  7.2%   -- 15.1% ADCNN-only, 3.4% stack-only

ADCNN owns the long-trail / faint end (its matched filter measures trails to 0.84-1.03x out to 56px,
while the stack's own Naive/Veres plugins return NaN above ~20px REGARDLESS of brightness). The stack
owns the short-trail / bright end (<=14px, 1-2 deg/day), where PSF detection is optimal. Neither
subsumes the other, so an ADCNN-only product silently forfeits ~3.4 points of real movers end-to-end
(~18% relative) and the merged upper bound is ~22.2%.

ORDER MATTERS: ADCNN detections are ~35% bright-star RING residuals on a real night. Chance-link rate
scales as n1*n2, so merging a clean external catalogue into an uncleaned one compounds contamination.
The rings are removed here (is_dipole, the detection-time flag) BEFORE the union, and the stack side
is already dipole-cut in ingest_diasource.

Every row carries `src` ("adcnn" | "stack") so the provenance of any alert member is recoverable.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def radec_to_unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)


def merge(adcnn_path, stack_path, out_path, dedup_arcsec=1.5, drop_rings=True, verbose=True):
    from scipy.spatial import cKDTree
    A = pd.read_csv(adcnn_path)
    A["src"] = "adcnn"
    n_a0 = len(A)
    if drop_rings and "is_dipole" in A.columns:
        ring = A["is_dipole"].fillna(False).astype(bool)
        A = A[~ring].reset_index(drop=True)
        if verbose:
            print(f"[merge] ADCNN {n_a0:,} -> {len(A):,} after dropping {int(ring.sum()):,} ring "
                  f"detections ({100*ring.mean():.1f}%) -- must precede the union", flush=True)
    try:
        S = pd.read_csv(stack_path)
    except Exception as e:
        print(f"[merge] no stack catalogue ({type(e).__name__}) -- writing ADCNN only (fail-safe)", flush=True)
        A.to_csv(out_path, index=False)
        return A
    S["src"] = "stack"
    # keep only stack detections ADCNN did not already find (same object, one row)
    tol = 2 * np.sin(np.radians(dedup_arcsec / 3600.0) / 2)
    keep = np.ones(len(S), bool)
    for v, g in S.groupby("visit"):
        a = A[A.visit == v]
        if not len(a):
            continue
        t = cKDTree(radec_to_unit(a.ra.to_numpy(), a.dec.to_numpy()))
        d1, _ = t.query(radec_to_unit(g.ra.to_numpy(), g.dec.to_numpy()), k=1)
        keep[g.index.to_numpy()] = d1 >= tol
    S_new = S[keep]
    M = pd.concat([A, S_new], ignore_index=True, sort=False)
    for c, fill in (("art_frac", 0.0), ("is_dipole", False), ("score", 1.0)):
        if c in M.columns:
            M[c] = M[c].fillna(fill)
    M.to_csv(out_path, index=False)
    if verbose:
        nl = int(((M.src == "stack") & (M.get("len_db", 0) >= 6)).sum())
        print(f"[merge] stack {len(S):,} -> {len(S_new):,} NEW (deduped at {dedup_arcsec}\") | "
              f"MERGED {len(M):,} dets ({int((M.src=='adcnn').sum()):,} adcnn + {len(S_new):,} stack, "
              f"{nl:,} stack rows with a linkable trail) -> {out_path}", flush=True)
    return M


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adcnn", required=True, help="ADCNN masked detection catalogue")
    ap.add_argument("--stack", required=True, help="stack catalogue from ingest_diasource")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-arcsec", type=float, default=1.5)
    ap.add_argument("--keep-rings", action="store_true",
                    help="do NOT drop ADCNN is_dipole rows before merging (diagnostic only)")
    a = ap.parse_args(argv)
    merge(a.adcnn, a.stack, a.out, a.dedup_arcsec, drop_rings=not a.keep_rings)


if __name__ == "__main__":
    main()
