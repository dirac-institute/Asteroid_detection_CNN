#!/usr/bin/env python3
"""Merge the STACK's DIA sources into the ADCNN detection catalogue. ALWAYS ON in production.

Measured 2026-08-12/13 on 3,857 injected movers with ONE matching procedure over both catalogues
(trail-segment matching; supersedes older figures produced by two different matchers):

    DETECTION both-epoch   ADCNN 49.8%   stack 37.0%   union 53.8%
      (a TIE at 0-8px: 44.5 vs 44.2; diverging with trail length to 49.0 vs 23.1 at 44-60px;
       the stack finds 154 both-epoch movers ADCNN misses)
    END-TO-END at the 1k budget   ADCNN 9.26%   merged 9.28%   -- NEUTRAL (flagship 2.06% vs 2.16%)

The union ceiling exceeds the delivered gain because the stack's unique detections carry no usable
trail geometry: DPDD trailLength is NaN on ~31% of rows and near-PSF on the stack-only population,
and this linker is built on trail-vs-chord agreement. Re-measuring those rows with ADCNN's template
bank cannot unlock them -- the only gate that stops it saturating is ADCNN's own seg+stage-2 chain,
and what passes that gate is what ADCNN already detected (measured: 2 rescuable of 992). Unlocking
needs a trail measurement independent of our segmentation, e.g. running
lsst.meas.extensions.trailedSources ourselves; the DRP does not run it.

BOTH sides are ring-cleaned here, with the same deep-refcat cut, BEFORE the union: the ADCNN side via
is_dipole/proximity, the stack side via proximity alone because its own dipole columns are inert on
the DRP output (dipoleFluxDiff NaN or exactly 0 everywhere; measured 61.2% of stack rows
ring-positioned vs 10.4% chance). The dedup runs against the FULL pre-cleaning ADCNN catalogue so the
stack's copies of deleted rings cannot re-enter as "NEW".

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
    A_rings = None
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
        # KEEP the deleted rings for the dedup below. Dropping them here and then deduping the stack
        # against what REMAINS means the stack's own copies of those rings are never compared to them
        # and sail in as "NEW" -- the merge re-imports through the stack door exactly what the ADCNN
        # cleaning just removed. MEASURED on 0706: 33,941 stack rows (6.06%) sit within 1.5" of an
        # ADCNN row deleted as a ring, and among the stack rows that are actually LINKABLE
        # (len_db >= 6) it is 23.44%. The control -- the same query against the KEPT ADCNN rows --
        # is 0.00%, which is the dedup working as designed on the half it can see.
        A_rings = A[ring].reset_index(drop=True)
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
    # CLEAN THE STACK SIDE TOO, symmetrically, before the union. It was never cleaned at all:
    # ingest_diasource's dipole drop is INERT on the DRP output (dipoleFluxDiff NaN on 72.4% and
    # exactly 0 on the rest -- nine nights, 3,880,041 sources, zero dropped), so the only working
    # ring lever is the same deep-refcat proximity cut the ADCNN side gets. MEASURED on the real
    # 0706 stack catalogue with the production ring_mask and an OFFSET NULL (positions shifted
    # 20-60", the same calibration the ADCNN cut shipped with): 61.2% of stack rows sit within 2.5"
    # of a mag<21 star against a 10.4% chance rate -- a 5.9x excess, 50.8 points of genuine
    # star-locked contamination. On the linkable subset (len_db>=6, score>=0.5) it is 27.3%. The
    # cost side of this cut was already established when it shipped for ADCNN: 2.7% of real movers
    # (offset null), ~20:1. A merge that ring-cleans one side and not the other undoes its own
    # cleaning through the other door.
    if drop_rings and refcat:
        s_ring = ring_mask(S, refcat_path=refcat, radius_arcsec=refcat_radius,
                           mag_max=refcat_mag_max, use_dipole=("is_dipole" in S.columns
                                                               and S.is_dipole.notna().any()),
                           verbose=False)
        S = S[~s_ring].reset_index(drop=True)
        if verbose:
            print(f"[merge] stack {len(S)+int(s_ring.sum()):,} -> {len(S):,} after dropping "
                  f"{int(s_ring.sum()):,} ring-positioned rows ({100*s_ring.mean():.1f}%) via the "
                  f"same deep-refcat cut the ADCNN side gets (measured excess over chance: 5.9x)",
                  flush=True)
    elif drop_rings and verbose:
        print("[merge] WARNING: no --refcat, so the STACK side is merged UNCLEANED -- its own dipole "
              "columns cannot flag anything (measured inert) and 61.2% of its rows are "
              "ring-positioned. Pass --refcat.", flush=True)
    # keep only the stack detections ADCNN did not already find, so one object is one row
    tol = 2 * np.sin(np.radians(dedup_arcsec / 3600.0) / 2)
    keep = np.ones(len(S), bool)
    A_all = (pd.concat([A, A_rings], ignore_index=True, sort=False)
             if A_rings is not None and len(A_rings) else A)
    n_ring_dedup = 0
    for v, g in S.groupby("visit"):
        a = A_all[A_all.visit == v]
        if not len(a):
            continue
        t = cKDTree(radec_to_unit(a.ra.to_numpy(), a.dec.to_numpy()))
        d1, _ = t.query(radec_to_unit(g.ra.to_numpy(), g.dec.to_numpy()), k=1)
        keep[g.index.to_numpy()] = d1 >= tol
    if A_rings is not None and len(A_rings):
        # how many stack rows this rescued us from re-importing, reported rather than silent
        k2 = np.ones(len(S), bool)
        for v, g in S.groupby("visit"):
            a = A[A.visit == v]
            if not len(a):
                continue
            t = cKDTree(radec_to_unit(a.ra.to_numpy(), a.dec.to_numpy()))
            d1, _ = t.query(radec_to_unit(g.ra.to_numpy(), g.dec.to_numpy()), k=1)
            k2[g.index.to_numpy()] = d1 >= tol
        n_ring_dedup = int((k2 & ~keep).sum())
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
    # DO NOT FABRICATE. Each of these fillna values silently disarms a downstream gate:
    #   art_frac -> 0.0  makes --art-frac-max unable to reject a stack row
    #   is_dipole -> False manufactures a column the ADCNN catalogue does not have, so link_2visit
    #                takes its "column exists" branch and prints "0/N ring dets removed" instead of
    #                the honest "no is_dipole column -> no-op"; it also walks straight past the
    #                REFUSAL guard above, which exists precisely to catch a missing is_dipole
    #   score -> 1.0     reintroduces, one layer downstream, the exact max-score trap that
    #                ingest_diasource._reliability was just hardened against: a maxed score is a
    #                maxed P(real) input, and --claim-order preal lets such pairs take real movers'
    #                detections
    # Leave them absent/NaN and SAY SO, so a consumer sees a missing input rather than a benign one.
    for col, why in (("art_frac", "--art-frac-max cannot reject these rows"),
                     ("is_dipole", "the dipole veto cannot veto these rows"),
                     ("score", "P(real) and --score-min have no value for these rows")):
        if col in M.columns:
            n_na = int(M[col].isna().sum())
            if n_na:
                print(f"[merge] {n_na:,} rows carry no `{col}` -- left as NaN, NOT defaulted: {why}",
                      flush=True)
    if "is_dipole" in M.columns:                       # bool dtype cannot hold NaN; use object
        M["is_dipole"] = M["is_dipole"].astype(object)
    M.to_csv(out_path, index=False)
    if verbose:
        nl = int(((M.src == "stack") & (M.get("len_db", pd.Series(0, index=M.index)) >= 6)).sum())
        if n_ring_dedup:
            print(f"[merge] {n_ring_dedup:,} stack rows dropped as duplicates of ADCNN rows that were "
                  f"themselves deleted as rings -- these would have been re-imported", flush=True)
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
