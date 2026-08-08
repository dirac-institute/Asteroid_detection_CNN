#!/usr/bin/env python3
"""Score each alert's SHAPE from its zoom stamps and flag bright-star difference-image DIPOLES.

The dominant morphology-blind FP in the ranked stream is the bright-star diffim dipole -- a round,
radially-symmetric black/white bullseye (positive lobe + adjacent negative lobe) from an imperfectly
subtracted moderately-bright star (PSF/kernel mismatch). P(real) ranks by per-detection SNR + orbit
chi2, so it is blind to this: a round dipole scores like a real source. A real asteroid is instead a
LINEAR, one-signed trail. Two catalog-free features on the already-cut zoom stamps separate them:

  elongation  eigenvalue ratio of the positive-signal 2nd moments  (trail high, dipole ~1)
  dipole      -min/max of the smoothed stamp, NO-DATA EXCLUDED     (dipole ~1, trail low)

Measured (20260705): rings (elong<1.6, dipole>0.55) and trails (elong>2.2, dipole<0.45) form clean,
separated clusters, and bright-star PROXIMITY does NOT separate them (the associating stars are mag
16-18, dense enough that ~26% of real trails sit near one) -- so shape, not proximity, is the filter.

CRITICAL: the dipole score must EXCLUDE no-data (exactly-0) pixels first, or a masked/off-detector
patch blows up -min/max and masquerades as a dipole. `nodata_frac` is reported separately.

Output: an npz aligned to the alerts (one row per alert, by file position) with
  elong, dipole, nodata, peak_sn, ripple(bool)
`ripple = elong<--elong-max & dipole>--dipole-min & peak_sn>--peak-min`.

Usage:
  python -m ADCNN.qa.alert_morphology --alerts alerts.jsonl --cutouts cutouts.npz --out morph.npz
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def _features(stamp, win=28, dwin=11):
    """(elongation, dipole, nodata_frac, peak_sn, neg_ratio, neg_blob) for one zoom stamp.

    `dipole` = -min/max (roundness-era peak measure). `neg_ratio` and `neg_blob` are the DMTN-007
    style dipole measure: integrated NEGATIVE-lobe flux vs positive-lobe flux, and the largest
    coherent negative blob, both in a SMALL window centred on the detection. A real transient is
    one-signed (neg_ratio ~ 0); a dipole has an adjacent coherent negative lobe -- and this catches
    ELONGATED dipoles (stretched +/- lobes) that the roundness test misses. The small window keeps a
    bright-star dipole in a stamp CORNER from vetoing a real trail whose own centre is clean."""
    from scipy.ndimage import gaussian_filter, label
    K = stamp.shape[0]; c = K // 2
    a = stamp[c - win:c + win + 1, c - win:c + win + 1].astype(np.float32)
    nod = (a == 0.0)
    ndfrac = float(nod.mean())
    valid = a[~nod]
    if valid.size < 50:
        return 1.0, 0.0, ndfrac, 0.0, 0.0, 0.0
    med = np.median(valid); mad = 1.4826 * np.median(np.abs(valid - med)) + 1e-6
    b = (a - med) / mad; b[nod] = 0.0
    yy, xx = np.mgrid[-win:win + 1, -win:win + 1].astype(np.float32)
    w = np.clip(b, 0, None); w[w < 3] = 0; s = w.sum()
    if s < 1e-3:
        elong = 1.0
    else:
        mx = (w * xx).sum() / s; my = (w * yy).sum() / s
        cxx = (w * (xx - mx) ** 2).sum() / s; cyy = (w * (yy - my) ** 2).sum() / s
        cxy = (w * (xx - mx) * (yy - my)).sum() / s
        tr = cxx + cyy; disc = max(tr * tr / 4 - (cxx * cyy - cxy * cxy), 0) ** 0.5
        elong = ((tr / 2 + disc) / max(tr / 2 - disc, 1e-3)) ** 0.5
    g = gaussian_filter(b, 1.2)
    peak = float(g.max())
    dipole = float(np.clip((-g.min()) / max(peak, 1e-3), 0, 2))
    # centred integrated-lobe dipole measure (small window)
    gc = g[win - dwin:win + dwin + 1, win - dwin:win + dwin + 1]
    pos = gc > 3; neg = gc < -3
    posflux = gc[pos].sum() if pos.any() else 1e-3
    negflux = -gc[neg].sum() if neg.any() else 0.0
    neg_ratio = float(negflux / max(posflux, 1e-3))
    lab, n = label(neg)
    neg_blob = float(max([(lab == k).sum() for k in range(1, n + 1)], default=0))
    return float(elong), dipole, ndfrac, peak, neg_ratio, neg_blob


def ripple_flag(E, D, SN, NR, NB, *, elong_max=1.6, dipole_min=0.55, peak_min=4.0,
                neg_ratio_min=0.20, neg_blob_min=2.0):
    """The bright-star DIPOLE/RING boolean from the morphology features -- ONE definition shared by
    the post-link QA veto (`compute`) and the DETECTION-TIME per-detection flag
    (`ADCNN.inference.catalog.panel_to_catalog_rows`), so the same stamp scores identically whether it
    is vetoed before linking (as a catalog `is_dipole` column) or after (on the alert stamps).
    round_dip = round bright-star ring; lobe_dip = DMTN-007 centred negative lobe (catches elongated
    dipoles the round test misses; no peak-SN gate -- the coherent neg-blob IS the signal)."""
    E = np.asarray(E, float); D = np.asarray(D, float); SN = np.asarray(SN, float)
    NR = np.asarray(NR, float); NB = np.asarray(NB, float)
    fin = np.isfinite(E)
    round_dip = fin & (SN > peak_min) & (E < elong_max) & (D > dipole_min)
    lobe_dip = fin & (NR > neg_ratio_min) & (NB >= neg_blob_min) & (E < 3.0)
    return round_dip | lobe_dip


def compute(alerts_path, cutouts_npz, out_npz, elong_max=1.6, dipole_min=0.55, peak_min=4.0,
            neg_ratio_min=0.20, neg_blob_min=2.0):
    n_alerts = sum(1 for _ in open(alerts_path))
    z = np.load(cutouts_npz)
    S = z["stamps"].astype(np.float32); al = z["alert"]
    # first zoom stamp per alert (epoch 0); both epochs would only average, epoch-0 is enough
    first = {}
    for i, a in enumerate(al):
        first.setdefault(int(a), i)
    E = np.full(n_alerts, np.nan, np.float32); D = np.full(n_alerts, np.nan, np.float32)
    ND = np.full(n_alerts, np.nan, np.float32); SN = np.full(n_alerts, np.nan, np.float32)
    NR = np.full(n_alerts, np.nan, np.float32); NB = np.full(n_alerts, np.nan, np.float32)
    for ai in range(n_alerts):
        i = first.get(ai)
        if i is None:
            continue
        E[ai], D[ai], ND[ai], SN[ai], NR[ai], NB[ai] = _features(S[i])
    ripple = ripple_flag(E, D, SN, NR, NB, elong_max=elong_max, dipole_min=dipole_min,
                         peak_min=peak_min, neg_ratio_min=neg_ratio_min, neg_blob_min=neg_blob_min)
    round_dip = np.isfinite(E) & (SN > peak_min) & (E < elong_max) & (D > dipole_min)
    lobe_dip = ripple & ~round_dip                                   # for the reporting split only
    np.savez(out_npz, elong=E, dipole=D, nodata=ND, peak_sn=SN, neg_ratio=NR, neg_blob=NB,
             ripple=ripple, params=np.array([elong_max, dipole_min, peak_min, neg_ratio_min,
                                             neg_blob_min], np.float32))
    print(f"[morph] {n_alerts} scored -> {out_npz}  |  dipoles: {int(ripple.sum())} "
          f"(round {int(round_dip.sum())}, lobe {int(lobe_dip.sum())})", flush=True)
    return out_npz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--cutouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--elong-max", type=float, default=1.6)
    ap.add_argument("--dipole-min", type=float, default=0.55)
    ap.add_argument("--peak-min", type=float, default=4.0)
    ap.add_argument("--neg-ratio-min", type=float, default=0.20, help="centred neg-lobe/pos-lobe flux ratio to flag a dipole")
    ap.add_argument("--neg-blob-min", type=float, default=2.0, help="min coherent negative-blob size (px) for the lobe measure "
                    "(2px: real trails are one-signed so neg_blob~0; measured on 0706 to catch the 41 elongated dipole EVADERS "
                    "that slip between the round test and the old nb>=4 lobe test, at zero cost to 612 clean trails)")
    a = ap.parse_args(argv)
    compute(a.alerts, a.cutouts, a.out, a.elong_max, a.dipole_min, a.peak_min, a.neg_ratio_min, a.neg_blob_min)


if __name__ == "__main__":
    sys.exit(main())
