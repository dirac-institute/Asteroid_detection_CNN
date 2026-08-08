#!/usr/bin/env python3
"""Apply the 1k/1.5k THRESHOLD op-point to an existing 2-visit stream -> artifact-clean survivors.

This is the op-point selector (NOT a rank cut). Because chi2 filters AFTER the orbit solve, the
chi2<=chi2_2v_max set is a strict SUBSET of the loosely-linked stream, so filtering here is IDENTICAL
to linking at the op. Drops applied (all TP-safe metadata; the morphology dipole veto runs AFTER this
on the survivors' stamps):
  - orbit chi2 > chi2_2v_max            (the volume knob; also removes poorly-fit dipole chance-links)
  - mfsnr < mfsnr_min_2v, mean trail len < len_db_min, rate outside [rate_lo,rate_hi]  (base op gates)
  - CONFIDENT-FP veto classes           (both-endpoint / recurring stationary, static-line, train)
  - member on an instrument-artifact MASK  (m_SPIKE / m_CROSSTALK / m_SAT_TEMPLATE / m_SAT): the diffim
    flags these pixels as diffraction-spike / crosstalk / saturated-star residuals -- bright-star
    artifacts, TP-safe (a real trail rarely sits exactly on one). Needs the masked dets for the flags.
  KEPT + labelled: single-counterpart stationary (86% chance coincidence -> dropping sheds real slow
  movers). Count is whatever the thresholds yield -- that is the point.

Usage:
  python -m ADCNN.qa.filter_op --alerts stream/alerts.jsonl --dets adcnn_dets_masked.csv \
      --op op_2v_stream_1k.json --out survivors.jsonl
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np

ARTIFACT_MASKS = ("m_SPIKE", "m_CROSSTALK", "m_SAT_TEMPLATE", "m_SAT")


def _G(a, *ks, d=None):
    x = a
    for k in ks:
        if not isinstance(x, dict):
            return d
        x = x.get(k)
    return d if x is None else x


def _member_artifact(alerts, dets_path):
    """-> set of alert indices whose EITHER member detection lands on an instrument-artifact mask."""
    import pandas as pd
    from scipy.spatial import cKDTree
    cols = ["visit", "detector", "ra", "dec"] + list(ARTIFACT_MASKS)
    d = pd.read_csv(dets_path, usecols=lambda c: c in cols)
    have = [m for m in ARTIFACT_MASKS if m in d.columns]
    if not have:
        return set()
    trees = {}
    for (v, det), g in d.groupby(["visit", "detector"]):
        cd = np.cos(np.radians(g.dec.values))
        trees[(int(v), int(det))] = (cKDTree(np.c_[g.ra.values * cd, g.dec.values]),
                                     g[have].to_numpy().any(axis=1))
    hit = set()
    tol = 1.5 / 3600.0
    for ai, a in enumerate(alerts):
        for ep in a["epochs"]:
            t = trees.get((int(ep["visit"]), int(ep["detector"])))
            if t is None:
                continue
            tree, flag = t
            dist, i = tree.query([ep["ra"] * np.cos(np.radians(ep["dec"])), ep["dec"]],
                                 distance_upper_bound=tol)
            if np.isfinite(dist) and flag[int(i)]:
                hit.add(ai); break
    return hit


def _near_bright_star(alerts, refcat_path, radius_arcsec, mag_max):
    """-> set of alert indices with EITHER member within `radius_arcsec` of a star brighter than
    `mag_max`. These are the bright-star PSF/decorrelation residuals (dipoles, RINGS, spikes) that
    are centred on bright stars -- a shape-based veto can't fully clear the faint tail, but proximity
    to the catalogued star does. NB: this DROPS real movers that transit near a bright star (measured
    non-trivial), so it is an explicit purity-over-completeness choice."""
    import pandas as pd
    from scipy.spatial import cKDTree
    rc = pd.read_parquet(refcat_path)
    rc = rc[rc.mag < mag_max]
    if len(rc) == 0:
        return set()

    def uv(ra, dec):
        r = np.radians(ra); d = np.radians(dec)
        return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    tree = cKDTree(uv(rc.ra.values, rc.dec.values))
    chord = 2 * np.sin(np.radians(radius_arcsec / 3600.0) / 2)          # angular radius -> chord
    hit = set()
    for ai, a in enumerate(alerts):
        for ep in a["epochs"]:
            if tree.query_ball_point(uv(np.array([ep["ra"]]), np.array([ep["dec"]]))[0], chord,
                                     return_length=True):
                hit.add(ai); break
    return hit


def filter_stream(alerts_path, dets_path, op_path, out_path, refcat_path=None):
    from ADCNN.qa.select_clean import _confident_fp
    op = json.load(open(op_path))
    CHI2, MF, LEN = op["chi2_2v_max"], op["mfsnr_min_2v"], op["len_db_min"]
    RLO, RHI = op["rate_lo_2v"], op["rate_hi_2v"]
    PROX_R = op.get("bright_star_radius_arcsec", 4.0)
    PROX_M = op.get("bright_star_mag_max", 16.0)
    alerts = [json.loads(l) for l in open(alerts_path)]
    artifact = _member_artifact(alerts, dets_path) if dets_path else set()
    near_star = (_near_bright_star(alerts, refcat_path, PROX_R, PROX_M)
                 if refcat_path and op.get("bright_star_proximity") else set())

    def keep(ai, a):
        return (_confident_fp(a) is None and ai not in artifact and ai not in near_star
                and _G(a, "orbit", "chi2", d=99) <= CHI2
                and _G(a, "vetting", "mfsnr_min", d=0) >= MF
                and np.mean(_G(a, "vetting", "trail_len_px", d=[0]) or [0]) >= LEN
                and RLO <= _G(a, "motion", "rate_degday", d=0) <= RHI)
    surv = [a for ai, a in enumerate(alerts) if keep(ai, a)]
    surv.sort(key=lambda a: -(_G(a, "ranking", "pReal", d=-1)))        # pReal order for display only
    with open(out_path, "w") as f:
        for a in surv:
            f.write(json.dumps(a) + "\n")
    print(f"[filter_op] chi2<={CHI2} + veto-drop + mask-artifact({len(artifact)}) + "
          f"bright-star({len(near_star)}): {len(surv)} survivors of {len(alerts)} -> {out_path}",
          flush=True)
    return len(surv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--dets", required=True, help="masked dets CSV (for the instrument-artifact mask flags)")
    ap.add_argument("--op", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refcat", default=None, help="all-sky bright-star refcat parquet (ra,dec,mag) "
                    "for the bright-star proximity veto; used only if the op enables it")
    a = ap.parse_args(argv)
    filter_stream(a.alerts, a.dets, a.op, a.out, a.refcat)


if __name__ == "__main__":
    sys.exit(main())
