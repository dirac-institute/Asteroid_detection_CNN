#!/usr/bin/env python3
"""Rebuild the CONTROL arm (segmentation trail length) from a matched-filter catalogue.

Detection was measured BIT-IDENTICAL between the two arms (0.00% of objects change epoch-A state,
0.02% epoch-B), because the matched filter only rewrites `length`/`beta` AFTER detection. So the two
arms differ solely in those two columns and in the sky endpoints derived from them. `inject_mf.py`
preserves `length_seg`/`beta_seg`, which means the control can be reconstructed offline instead of
costing a second 2-hour GPU run -- and it is a STRONGER control than two separate runs, because the
detections are the same by construction rather than by verification.

SELF-CONSISTENCY GATE. The reconstruction is only trustworthy if the endpoint recomputation
reproduces what the injector itself wrote. So before emitting the control arm, this recomputes the
endpoints from the MF `length`/`beta` -- the arm already present in the file -- and requires them to
match the stored ra0/dec0/ra1/dec1. If that fails, the WCS handling here differs from the injector's
and the reconstruction must NOT be used; queue a real control run instead.

Endpoints must be offset in PIXELS and pushed through each panel's WCS. Treating the image-frame
`beta` as a sky position angle rotates every trail by its detector's orientation -- the bug that once
made dpa_tm disagree by ~48 deg and appeared to kill 81% of true pairs.

Usage:  python reconstruct_control.py <in_dets.csv> <out_dets.csv> [manifest.csv]
"""
import sys

import numpy as np
import pandas as pd
from astropy.wcs import WCS

sys.path.insert(0, ".")
from ADCNN.inference.diffim_io import open_diffim

TOL_ARCSEC = 1e-3      # endpoint agreement required of the self-consistency gate


def endpoints(df, man, lcol, bcol):
    """Sky endpoints for each detection from (x, y, length, beta) via that panel's WCS."""
    ra0 = np.full(len(df), np.nan); dec0 = ra0.copy(); ra1 = ra0.copy(); dec1 = ra0.copy()
    for (v, d), g in df.groupby(["visit", "detector"]):
        r = man[(man.visit == v) & (man.detector == d)]
        if not len(r):
            continue
        try:
            with open_diffim(r.fits_path.iloc[0], memmap=False) as h:
                w = WCS(h[1].header)
        except Exception:
            continue
        br = np.radians(g[bcol].to_numpy(float)); Lp = np.clip(g[lcol].to_numpy(float), 0, None)
        hdx = 0.5 * Lp * np.cos(br); hdy = 0.5 * Lp * np.sin(br)
        xy = g[["x", "y"]].to_numpy(float)
        s0 = w.all_pix2world(np.stack([xy[:, 0] - hdx, xy[:, 1] - hdy], 1), 0)
        s1 = w.all_pix2world(np.stack([xy[:, 0] + hdx, xy[:, 1] + hdy], 1), 0)
        i = g.index.to_numpy()
        ra0[df.index.get_indexer(i)] = s0[:, 0]; dec0[df.index.get_indexer(i)] = s0[:, 1]
        ra1[df.index.get_indexer(i)] = s1[:, 0]; dec1[df.index.get_indexer(i)] = s1[:, 1]
    return ra0, dec0, ra1, dec1


def main(src, dst, manifest="outputs/runs/10k_cadence/run_night_20260706/work/manifest.csv"):
    D = pd.read_csv(src)
    man = pd.read_csv(manifest)
    for c in ("length_seg", "beta_seg"):
        if c not in D.columns:
            raise SystemExit(f"[reconstruct] {src} has no `{c}` -- it predates the preservation fix. "
                             f"A reconstructed control is impossible; run a real control instead.")

    print("[gate] self-consistency: recompute the MF arm's endpoints and compare to what the "
          "injector wrote")
    a0, d0, a1, d1 = endpoints(D, man, "length", "beta")
    ok = np.isfinite(a0) & np.isfinite(D.ra0.to_numpy())
    worst = max(np.nanmax(np.abs(a0[ok] - D.ra0.to_numpy()[ok])),
                np.nanmax(np.abs(d0[ok] - D.dec0.to_numpy()[ok])),
                np.nanmax(np.abs(a1[ok] - D.ra1.to_numpy()[ok])),
                np.nanmax(np.abs(d1[ok] - D.dec1.to_numpy()[ok]))) * 3600.0
    print(f"[gate] compared {ok.sum():,} detections | worst endpoint disagreement {worst:.2e} arcsec")
    if worst > TOL_ARCSEC:
        raise SystemExit(f"[gate] FAIL: endpoint recomputation does not reproduce the injector "
                         f"({worst:.3g}\" > {TOL_ARCSEC}\"). Do NOT use a reconstructed control.")
    print(f"[gate] PASS -- the recomputation reproduces the injector, so the same code applied to "
          f"length_seg/beta_seg is trustworthy")

    C = D.copy()
    C["length"] = D["length_seg"]; C["beta"] = D["beta_seg"]; C["len_db"] = D["length_seg"]
    a0, d0, a1, d1 = endpoints(C, man, "length", "beta")
    C["ra0"], C["dec0"], C["ra1"], C["dec1"] = a0, d0, a1, d1
    C = C[np.isfinite(C.ra0)].reset_index(drop=True)
    C.to_csv(dst, index=False)
    print(f"[reconstruct] wrote {dst}  n={len(C):,}  (MF arm: {len(D):,})")
    print(f"[reconstruct] median length  MF {D.length.median():.2f}px  vs  seg {C.length.median():.2f}px")


if __name__ == "__main__":
    main(*sys.argv[1:])
