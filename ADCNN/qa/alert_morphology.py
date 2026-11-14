"""Flag bright-star difference-image DIPOLES from the SHAPE of a stamp.

The bright-star diffim dipole is a round, radially-symmetric black/white bullseye (positive lobe +
adjacent negative lobe) from an imperfectly subtracted moderately-bright star (PSF/kernel mismatch).
A real asteroid is instead a LINEAR, one-signed trail. Two catalog-free features on the stamp
separate them:

  elongation  eigenvalue ratio of the positive-signal 2nd moments  (trail high, dipole ~1)
  dipole      -min/max of the smoothed stamp, NO-DATA EXCLUDED     (dipole ~1, trail low)

Measured (20260705): rings (elong<1.6, dipole>0.55) and trails (elong>2.2, dipole<0.45) form clean,
separated clusters, and bright-star PROXIMITY does NOT separate them (the associating stars are mag
16-18, dense enough that ~26% of real trails sit near one) -- so shape, not proximity, is the filter.

CRITICAL: the dipole score must EXCLUDE no-data (exactly-0) pixels first, or a masked/off-detector
patch blows up -min/max and masquerades as a dipole.
"""
from __future__ import annotations
import numpy as np


def _features(stamp, win=28, dwin=11):
    """(elongation, dipole, peak_sn, neg_ratio, neg_blob) for one zoom stamp.

    `dipole` = -min/max of the smoothed stamp. `neg_ratio` and `neg_blob` are the DMTN-007
    style dipole measure: integrated NEGATIVE-lobe flux vs positive-lobe flux, and the largest
    coherent negative blob, both in a SMALL window centred on the detection. A real transient is
    one-signed (neg_ratio ~ 0); a dipole has an adjacent coherent negative lobe -- and this catches
    ELONGATED dipoles (stretched +/- lobes) that the roundness test misses. The small window keeps a
    bright-star dipole in a stamp CORNER from vetoing a real trail whose own centre is clean."""
    from scipy.ndimage import gaussian_filter, label
    K = stamp.shape[0]; c = K // 2
    a = stamp[c - win:c + win + 1, c - win:c + win + 1].astype(np.float32)
    nod = (a == 0.0)
    valid = a[~nod]
    if valid.size < 50:
        return 1.0, 0.0, 0.0, 0.0, 0.0
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
    return float(elong), dipole, peak, neg_ratio, neg_blob


ELONG_MAX, DIPOLE_MIN, PEAK_MIN = 1.6, 0.55, 4.0      # round bright-star ring
NEG_RATIO_MIN, NEG_BLOB_MIN = 0.20, 2.0               # DMTN-007 centred negative lobe


def ripple_flag(E, D, SN, NR, NB):
    """The bright-star DIPOLE/RING boolean from the morphology features: the DETECTION-TIME
    per-detection flag (`ADCNN.inference.catalog._attach_dipole_morphology`).
    round_dip = round bright-star ring; lobe_dip = DMTN-007 centred negative lobe (catches elongated
    dipoles the round test misses; no peak-SN gate -- the coherent neg-blob IS the signal)."""
    E = np.asarray(E, float); D = np.asarray(D, float); SN = np.asarray(SN, float)
    NR = np.asarray(NR, float); NB = np.asarray(NB, float)
    fin = np.isfinite(E)
    round_dip = fin & (SN > PEAK_MIN) & (E < ELONG_MAX) & (D > DIPOLE_MIN)
    lobe_dip = fin & (NR > NEG_RATIO_MIN) & (NB >= NEG_BLOB_MIN) & (E < 3.0)
    return round_dip | lobe_dip

