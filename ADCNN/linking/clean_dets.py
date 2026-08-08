#!/usr/bin/env python3
"""Remove bright-star DIPOLE/RING residuals from a DETECTION catalog, before linking or merging.

Doing this at the DETECTION level (not on alerts) is what makes rings unable to form tracklets at all,
and it is a prerequisite for merging with the stack's DIA sources -- otherwise a clean external
catalogue gets polluted by ADCNN's ring detections and the chance-link rate (~ n1*n2) goes up.

Two independent, measured cuts:
  is_dipole   detection-time morphology (ADCNN.inference.catalog._attach_dipole_morphology ->
              ADCNN.qa.alert_morphology.ripple_flag): catches BRIGHT rings, catalogue-free, so it
              works where no refcat reaches. Absent column = skipped (older catalogues).
  proximity   within --radius of a refcat star brighter than --mag-max. THE primary lever, but only
              when the refcat is DEEP: the residual rings sit on mag 19-21 stars, so a mag<19
              catalogue catches ~0% of them. Measured on 20260706 with an offset null (positions
              shifted 20-60", preserving footprint but decorrelating from stars, so the null rate IS
              the chance cost a real mover pays): mag<21 @2.5" removes 55.8% of the product's ring
              contamination at 2.7% cost to real movers (~20:1). See build_static_refcat --mag-max 21.

Usage:
  python -m ADCNN.linking.clean_dets --dets adcnn_dets_masked.csv --refcat bright_refcat.parquet \
      --out adcnn_dets_clean.csv [--radius 2.5] [--mag-max 21] [--no-dipole]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def radec_to_unit(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], -1)


def ring_mask(dets: pd.DataFrame, refcat_path=None, radius_arcsec=2.5, mag_max=21.0,
              use_dipole=True, verbose=True):
    """-> boolean mask of detections that are bright-star ring/dipole residuals."""
    n = len(dets)
    dip = np.zeros(n, bool)
    if use_dipole and "is_dipole" in dets.columns:
        dip = dets["is_dipole"].fillna(False).astype(bool).to_numpy()
    elif use_dipole and verbose:
        print("[clean-dets] no is_dipole column (older catalogue) -- morphology cut skipped", flush=True)
    prox = np.zeros(n, bool)
    if refcat_path:
        from scipy.spatial import cKDTree
        rc = pd.read_parquet(refcat_path)
        rc = rc[np.isfinite(rc["mag"]) & (rc["mag"] < mag_max)]
        if len(rc):
            if verbose and rc["mag"].max() < mag_max - 0.5:
                print(f"[clean-dets] WARNING: refcat only reaches mag {rc['mag'].max():.1f} < requested "
                      f"{mag_max} -- rings on fainter stars will SURVIVE (rebuild with --mag-max 21)",
                      flush=True)
            tree = cKDTree(radec_to_unit(rc.ra.to_numpy(), rc.dec.to_numpy()))
            chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)
            d1, _ = tree.query(radec_to_unit(dets.ra.to_numpy(), dets.dec.to_numpy()), k=1)
            prox = d1 < chord
    ring = dip | prox
    if verbose:
        print(f"[clean-dets] {n:,} dets -> ring {int(ring.sum()):,} ({100*ring.mean():.1f}%) "
              f"[morphology {int(dip.sum()):,}, proximity {int(prox.sum()):,}, "
              f"both {int((dip & prox).sum()):,}]", flush=True)
    return ring


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refcat", default=None, help="all-sky refcat parquet (ra,dec,mag); build DEEP "
                    "(--mag-max 21) or the residual rings survive")
    ap.add_argument("--radius", type=float, default=2.5, help="proximity radius (arcsec)")
    ap.add_argument("--mag-max", type=float, default=21.0)
    ap.add_argument("--no-dipole", action="store_true", help="skip the is_dipole morphology cut")
    ap.add_argument("--flag-only", action="store_true", help="write an is_ring column instead of dropping")
    a = ap.parse_args(argv)
    d = pd.read_csv(a.dets)
    ring = ring_mask(d, a.refcat, a.radius, a.mag_max, use_dipole=not a.no_dipole)
    if a.flag_only:
        d["is_ring"] = ring
        out = d
    else:
        out = d[~ring].reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print(f"[clean-dets] wrote {len(out):,} dets -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
