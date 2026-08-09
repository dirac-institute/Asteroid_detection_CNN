#!/usr/bin/env python3
"""Merge the STACK's DIA sources into the ADCNN detection catalogue. ALWAYS ON in production.

The two detectors are complementary, measured against injected truth (3,857 movers, rate >1 deg/day,
SNR 2-10, each system run with its OWN code; outputs/runs/pa_validate + the adcnn-vs-stack-truth note):

    DETECTION   ADCNN 38.6%   stack 39.1%   -- a TIE;  9.2% ADCNN-only,  9.7% stack-only
    END-TO-END  ADCNN 18.8%   stack  7.2%           15.1% ADCNN-only,  3.4% stack-only

ADCNN owns the long-trail / faint end: its matched filter recovers trail length at 0.84-1.03x out to
56px, while the stack's OWN ext_trailedSources Naive AND Veres plugins return NaN above ~20px
REGARDLESS of brightness (verified at mag 19.0) -- so the stack detects long trails but cannot hand the
linker usable geometry. The stack owns the short-trail / bright end (<=14px, 1-2 deg/day) where PSF
detection is optimal. Neither subsumes the other: an ADCNN-only product forfeits the ~3.4 points of
real movers only the stack finds (~18% relative), and the merged upper bound is ~22.2%.

ORDER IS MANDATORY: ADCNN detections are ~18-35% bright-star RING residuals on a real night, and the
chance-link rate goes as n1*n2 -- merging a clean external catalogue into an uncleaned one compounds
the contamination. Rings are therefore removed from the ADCNN side BEFORE the union, via the
detection-time `is_dipole` flag when present, else via deep-refcat proximity. If NEITHER is available
this REFUSES to run rather than silently merging into a ring-contaminated catalogue (that silent skip
was a real bug: night catalogues predating detection-time morphology have no is_dipole column).

Every row carries `src` ("adcnn" | "stack"), so the provenance of any alert member is recoverable.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def radec_to_unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)


def merge(adcnn_path, stack_path, out_path, dedup_arcsec=1.5, drop_rings=True, verbose=True,
          refcat=None, refcat_mag_max=21.0, refcat_radius=2.5):
    from scipy.spatial import cKDTree
    from ADCNN.linking.clean_dets import ring_mask
    A = pd.read_csv(adcnn_path)
    A["src"] = "adcnn"
    n_a0 = len(A)
    if drop_rings:
        has_flag = "is_dipole" in A.columns
        if not has_flag and not refcat:
            raise SystemExit(
                "[merge] REFUSING to merge: the ADCNN catalogue has no `is_dipole` column (it was "
                "detected before detection-time morphology existed) and no --refcat was given, so "
                "rings cannot be removed before the union. Chance links go as n1*n2, so merging a "
                "clean catalogue into a ring-contaminated one compounds the contamination. Pass "
                "--refcat <deep mag<21 refcat>, or --keep-rings to override deliberately.")
        ring = ring_mask(A, refcat_path=refcat, radius_arcsec=refcat_radius,
                         mag_max=refcat_mag_max, use_dipole=has_flag, verbose=False)
        A = A[~ring].reset_index(drop=True)
        if verbose:
            how = "is_dipole" if has_flag else f"deep-refcat proximity (mag<{refcat_mag_max}, {refcat_radius}\")"
            print(f"[merge] ADCNN {n_a0:,} -> {len(A):,} after dropping {int(ring.sum()):,} ring "
                  f"detections ({100*ring.mean():.1f}%) via {how} -- precedes the union", flush=True)
    try:
        S = pd.read_csv(stack_path)
    except Exception as e:
        print(f"[merge] no stack catalogue ({type(e).__name__}) -- writing ADCNN only (fail-safe)", flush=True)
        A.to_csv(out_path, index=False)
        return A
    S["src"] = "stack"
    # keep only the stack detections ADCNN did not already find, so one object is one row
    tol = 2 * np.sin(np.radians(dedup_arcsec / 3600.0) / 2)
    keep = np.ones(len(S), bool)
    for v, g in S.groupby("visit"):
        a = A[A.visit == v]
        if not len(a):
            continue
        t = cKDTree(radec_to_unit(a.ra.to_numpy(), a.dec.to_numpy()))
        d1, _ = t.query(radec_to_unit(g.ra.to_numpy(), g.dec.to_numpy()), k=1)
        keep[g.index.to_numpy()] = d1 >= tol
    S_new = S[keep].copy()
    # Stack rows carry no fits_path (ingest_diasource reads a table, not pixels), but every
    # downstream pixel stage (alert_cutouts -> morphology -> sheets/pairs) needs one. Fill it from
    # the ADCNN catalogue's (visit, detector) -> fits_path map: same panel, same file.
    if "fits_path" in A.columns and {"visit", "detector"} <= set(S_new.columns):
        fp = (A.dropna(subset=["fits_path"]).groupby(["visit", "detector"])["fits_path"].first())
        idx = pd.MultiIndex.from_arrays([S_new.visit.astype(int), S_new.detector.astype(int)])
        S_new["fits_path"] = fp.reindex(idx).to_numpy()
        n_bad = int(S_new.fits_path.isna().sum())
        if n_bad:
            print(f"[merge] dropping {n_bad:,} stack rows on panels ADCNN never saw (no fits_path)", flush=True)
            S_new = S_new[S_new.fits_path.notna()]
    M = pd.concat([A, S_new], ignore_index=True, sort=False)
    if "art_frac" in M.columns:
        M["art_frac"] = M["art_frac"].astype(float).fillna(0.0)
    if "is_dipole" in M.columns:
        M["is_dipole"] = M["is_dipole"].fillna(False).astype(bool)
    if "score" in M.columns:
        M["score"] = M["score"].astype(float).fillna(1.0)
    M.to_csv(out_path, index=False)
    if verbose:
        nl = int(((M.src == "stack") & (M.get("len_db", pd.Series(0, index=M.index)) >= 6)).sum())
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
    ap.add_argument("--refcat", default=None,
                    help="deep (mag<21) all-sky refcat. REQUIRED when the ADCNN catalogue predates the "
                         "detection-time is_dipole column, so rings can still be removed before the union")
    ap.add_argument("--refcat-mag-max", type=float, default=21.0)
    ap.add_argument("--refcat-radius", type=float, default=2.5)
    ap.add_argument("--keep-rings", action="store_true",
                    help="do NOT remove ADCNN rings before merging (diagnostic only -- inflates chance links)")
    a = ap.parse_args(argv)
    merge(a.adcnn, a.stack, a.out, a.dedup_arcsec, drop_rings=not a.keep_rings,
          refcat=a.refcat, refcat_mag_max=a.refcat_mag_max, refcat_radius=a.refcat_radius)


if __name__ == "__main__":
    main()
