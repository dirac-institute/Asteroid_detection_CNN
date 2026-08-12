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
        return set(), False
    trees = {}
    for (v, det), g in d.groupby(["visit", "detector"]):
        cd = np.cos(np.radians(g.dec.values))
        # `.any()` on a NaN is TRUE. Stack rows carry no m_* columns at all (100% NaN after the
        # merge), so every one of them was flagged as sitting on an instrument-artifact mask and any
        # alert with a stack member was dropped. MEASURED on the real 0706 merged product: zeroing
        # the stack's m_* values takes the flag count 460 -> 427 and admits the one stack-member
        # alert that reaches the op. A missing mask is NOT a set mask -- fill with False first.
        trees[(int(v), int(det))] = (cKDTree(np.c_[g.ra.values * cd, g.dec.values]),
                                     g[have].fillna(False).astype(bool).to_numpy().any(axis=1))
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
    return hit, True


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


def filter_stream(alerts_path, dets_path, op_path, out_path, refcat_path=None, allow_unranked=False):
    from ADCNN.qa.select_clean import _confident_fp
    op = json.load(open(op_path))
    CHI2, MF, LEN = op["chi2_2v_max"], op["mfsnr_min_2v"], op["len_db_min"]
    RLO, RHI = op["rate_lo_2v"], op["rate_hi_2v"]
    # Defaults are the PRE deep-refcat-fix values, so an op-point that enables proximity but omits
    # the depth keys silently loses 99.5% of the veto (MEASURED: 182 -> 1 flagged of 200 alerts).
    # Refuse rather than degrade.
    if op.get("bright_star_proximity") and not ({"bright_star_radius_arcsec",
                                                 "bright_star_mag_max"} <= set(op)):
        raise SystemExit(
            "[filter_op] op enables bright_star_proximity but omits bright_star_radius_arcsec / "
            "bright_star_mag_max. The built-in defaults (4 arcsec, mag<16) predate the deep-refcat "
            "fix and lose ~99.5% of the veto silently. Set both keys explicitly.")
    PROX_R = op.get("bright_star_radius_arcsec", 4.0)
    PROX_M = op.get("bright_star_mag_max", 16.0)
    alerts = [json.loads(l) for l in open(alerts_path)]
    artifact, _have_masks = _member_artifact(alerts, dets_path) if dets_path else (set(), False)
    # A veto enabled in the op but handed no catalogue is a NO-OP that prints "(0)" -- textually
    # identical to "0 flagged". MEASURED on 200 real alerts: with --refcat 10 survivors, without it
    # 84 (8.4x larger, ~88% ring-contaminated). This exact bug shipped once. Fail, do not degrade.
    if op.get("bright_star_proximity") and not refcat_path:
        raise SystemExit(
            "[filter_op] op enables bright_star_proximity but no --refcat was given, so the veto "
            "would silently do nothing (the product would be ~8x larger and heavily ring-"
            "contaminated). Pass --refcat <deep mag<21 refcat>, or disable the veto in the op.")
    near_star = (_near_bright_star(alerts, refcat_path, PROX_R, PROX_M)
                 if refcat_path and op.get("bright_star_proximity") else set())

    def keep(ai, a):
        # chi2 is None for 3+visit tracks -- they have no 2-visit orbit solve -- and `_G`'s default
        # fires on None, so every one of them scored 99 and failed an 8.0 gate. That is a
        # default-on-missing failure, not a threshold decision: all 52 real 3+visit alerts across the
        # nine delivered nights were dropped, 39 of them blocked by this gate ALONE, and every
        # delivered night reads multi_epoch_fraction = 0.0. The 3-sighting tier is the discovery tier
        # (purity ~1.00 vs 0.17-0.56 for 2-sighting), and 7 of 9 nights deliver UNDER the 1000 budget
        # (4,182 of 9,000 slots filled), so admitting them displaces nothing on those nights.
        # rerank_alerts already handles this same field the same way; filter_op never got the fix.
        _c = _G(a, "orbit", "chi2", d=None)
        return (_confident_fp(a) is None and ai not in artifact and ai not in near_star
                and (_c is None or float(_c) <= CHI2)
                and _G(a, "vetting", "mfsnr_min", d=0) >= MF
                and np.mean(_G(a, "vetting", "trail_len_px", d=[0]) or [0]) >= LEN
                and RLO <= _G(a, "motion", "rate_degday", d=0) <= RHI)
    surv = [a for ai, a in enumerate(alerts) if keep(ai, a)]
    # RANK BY (class, -pReal) -- the same key rank_alerts uses, so FLAG-not-drop survives the sort.
    #
    # Two defects lived here. (1) With `-pReal` alone and a stream carrying NO `ranking` key, every
    # sort key was identical, Python's stable sort was a NO-OP, and the output silently kept the
    # linker's chi2 order. MEASURED: all nine delivered nights have pReal=None for 100% of alerts,
    # and only 24 of the top-100 a vetter sees were the true top-100 by P(real). The symptom was in
    # plain sight -- every rendered file is named `pNA`. (2) Dropping the class term inverts
    # FLAG-not-drop: applying `-pReal` alone to a stream that DOES carry pReal put 52 veto-flagged
    # alerts in the top 100, best at rank 6. The class order survived only BECAUSE the sort was inert,
    # so these two must be fixed together.
    from ADCNN.linking.rank_alerts import _rank_class
    _n_pr = sum(1 for a in surv if _G(a, "ranking", "pReal", d=None) is not None)
    if surv and _n_pr == 0 and not allow_unranked:
        raise SystemExit(
            f"[filter_op] REFUSING to emit an unranked product: none of {len(surv)} survivors carry "
            f"ranking.pReal, so a pReal sort is a silent no-op and the output would keep the linker's "
            f"chi2 order. Run ADCNN.qa.rerank_alerts on the input stream first (P(real) is computable "
            f"from the alert fields), or pass --allow-unranked to accept chi2 order deliberately.")
    if surv and _n_pr < len(surv):
        print(f"[filter_op] WARNING: {len(surv)-_n_pr} of {len(surv)} survivors lack ranking.pReal "
              f"and will sort last within their class", flush=True)
    # TIER before pReal, matching rerank_alerts: pReal is not computable for a 3+visit track (no
    # 2-visit chi2), so keying on pReal alone sorts the ~100%-purity discovery tier LAST.
    surv.sort(key=lambda a: (_rank_class(a), a.get("priority", 9),
                             -(_G(a, "ranking", "pReal", d=-1))))
    with open(out_path, "w") as f:
        for a in surv:
            f.write(json.dumps(a) + "\n")
    if not _have_masks:
        # This warning referenced an undefined name and raised NameError in EXACTLY the case it
        # exists for (dets with no m_* columns) -- after the survivors file was already written, so a
        # caller guarding on `[ -s out ]` consumed a product whose mask veto had silently done
        # nothing. _member_artifact now reports whether the columns were there.
        print("[filter_op] WARNING: dets carry no m_* mask columns -- the mask-artifact veto is OFF, "
              "not '0 flagged'", flush=True)
    print(f"[filter_op] chi2<={CHI2} + veto-drop + mask-artifact({len(artifact)}) + "
          f"bright-star({len(near_star)}): {len(surv)} survivors of {len(alerts)} -> {out_path}",
          flush=True)
    return len(surv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--dets", required=True, help="masked dets CSV (for the instrument-artifact mask flags)")
    ap.add_argument("--op", required=True)
    ap.add_argument("--allow-unranked", action="store_true",
                    help="emit in linker chi2 order when no alert carries ranking.pReal (default: refuse)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--refcat", default=None, help="all-sky bright-star refcat parquet (ra,dec,mag) "
                    "for the bright-star proximity veto; used only if the op enables it")
    a = ap.parse_args(argv)
    filter_stream(a.alerts, a.dets, a.op, a.out, a.refcat, allow_unranked=a.allow_unranked)


if __name__ == "__main__":
    sys.exit(main())
