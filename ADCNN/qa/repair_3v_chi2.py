#!/usr/bin/env python3
"""Post-hoc: populate `orbit.chi2` on 3+visit alerts that were linked before the 2026-08-14 fix.

WHY. link_2visit used to compute `pair_chi2` only for `n_ep == 2` and write NaN for everything else,
so every 3+visit alert shipped with `orbit.chi2 = null` -- no orbit statistic at all on the tier that
sorts to the TOP of the delivered product. link_2visit now populates it (pair_chi2 sorts by mjd and
reads iloc[0]/iloc[-1], so on a 3+ member set it scores the OUTER pair -- the widest arc). This tool
applies the same computation to nights already built, without relinking.

IT DOES NOT CHANGE WHICH ALERTS ARE DELIVERED. The 3+visit tier is exempt from the chi2 gate
(filter_op._TIER_EXEMPT), and the delivery sort key is (class, -pReal) with pReal null for the tier.
So this is metadata only: no re-filter, no re-render, no cutout regeneration. That is what makes a
post-hoc repair safe here -- verify with --check before and after if in doubt.

WHY THE TIER IS NOT GATED ON THE VALUE IT NOW CARRIES. MEASURED on the six delivered 3+visit alerts
of 20260710/20260711: gating at the shipped chi2<=10 drops FOUR, and the two killed by pair_chi2's
hard pre-gate are exactly the SHORT-TRAIL ones (len_db 6.8-12.2 px, dpa_tt 24.4 and 26.3 against a
15 deg cut). That cut is 2-visit calibration -- _PA_S gives sigma_PA ~17 deg at 6 px, so
sigma(dpa_tt) ~24 deg and 15 deg rejects ~1 sigma of REAL short-trail scatter. Gating would
preferentially delete FAINT 3-sighting detections, the opposite of the tier's purpose.

    python -m ADCNN.qa.repair_3v_chi2 --night-dir outputs/runs/10k_cadence/run_night_20260710
    python -m ADCNN.qa.repair_3v_chi2 --night-dir ... --check      # report only, write nothing
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

NEED = ["ra", "dec", "mjd", "ra0", "dec0", "ra1", "dec1", "mf_snr", "src", "len_db",
        "visit", "detector", "score"]
MATCH_TOL_ARCSEC = 1.0      # a member must land on ITS detection, not a neighbour


def _members(d, alert, tol_arcsec=MATCH_TOL_ARCSEC):
    """Row indices in `d` for this alert's epochs, or None if any member cannot be matched.

    Matched within (visit, detector) then nearest on sky, with an explicit separation ceiling: a
    silent mismatch would compute chi2 for a DIFFERENT track and there would be nothing to show it.
    """
    idx, worst = [], 0.0
    for e in alert["epochs"]:
        sub = d[(d.visit == e["visit"]) & (d.detector == e["detector"])]
        if not len(sub):
            return None, np.inf
        k = ((sub.ra - e["ra"]) ** 2 + (sub.dec - e["dec"]) ** 2).idxmin()
        sep = 3600.0 * float(np.hypot((sub.ra[k] - e["ra"]) * np.cos(np.radians(e["dec"])),
                                      sub.dec[k] - e["dec"]))
        if sep > tol_arcsec:
            return None, sep
        idx.append(k); worst = max(worst, sep)
    return idx, worst


def repair(night_dir, check=False, exptime=30.0):
    from ADCNN.linking.link_2visit import pair_chi2, ADM_KEYS
    nd = str(night_dir).rstrip("/")
    dets = os.path.join(nd, "dets_merged.csv")
    if not os.path.exists(dets):
        print(f"[repair3v] {nd}: no dets_merged.csv -- cannot recover trail endpoints, skipping")
        return 0
    d = pd.read_csv(dets, usecols=lambda c: c in NEED, low_memory=False)
    total = 0
    for rel in ("stream/alerts.jsonl", "stream_1k/alerts.jsonl"):
        p = os.path.join(nd, rel)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        al = [json.loads(l) for l in open(p)]
        tri = [a for a in al if int(a.get("nEpochs", 2)) >= 3]
        # IDEMPOTENT. A pre-gated track legitimately ends up with chi2 null (inf is not JSON),
        # so "chi2 is None" alone would re-do it on every run and report work it is not doing.
        # `chi2_source` is the marker that this alert has BEEN through the computation.
        todo = [a for a in tri
                if (a.get("orbit") or {}).get("chi2") is None
                and not (a.get("orbit") or {}).get("chi2_source")]
        n_fixed = n_unmatched = 0
        vals = []
        for a in todo:
            idx, sep = _members(d, a)
            if idx is None:
                n_unmatched += 1
                print(f"[repair3v]   {a.get('alertId')}: member unmatched (sep {sep:.2f}\") -- LEFT AS NULL")
                continue
            g = d.loc[idx].sort_values("mjd")
            c2, ci = pair_chi2(g, exptime)
            vals.append((a.get("alertId"), c2))
            if not check:
                orb = a.setdefault("orbit", {})
                orb["chi2"] = None if not np.isfinite(c2) else float(c2)
                orb["chi2_source"] = ("outer-pair PRE-GATED (dpa_tm/dpa_tt/dspeed -> inf), post-hoc"
                                      if not np.isfinite(c2) else "outer-pair, post-hoc")
                for k, v in (("a_au", ci.get("a")), ("ecc", ci.get("e"))):
                    orb[k] = None if v is None or not np.isfinite(v) else float(v)
                for k in ADM_KEYS:
                    v = ci.get(k)
                    orb[k] = None if v is None or not np.isfinite(v) else float(v)
            n_fixed += 1
        print(f"[repair3v] {rel}: {len(al):,} alerts, {len(tri)} are 3+visit, "
              f"{len(todo)} lacked chi2 -> {n_fixed} computed"
              + (f", {n_unmatched} unmatched" if n_unmatched else "")
              + (" (CHECK ONLY, nothing written)" if check else ""))
        for aid, c2 in vals:
            print(f"[repair3v]     {aid}: chi2={'inf (pre-gated)' if not np.isfinite(c2) else f'{c2:.3g}'}")
        if not check and n_fixed:
            # atomic: a half-written alerts.jsonl would take the night's whole product with it
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                for a in al:
                    f.write(json.dumps(a) + "\n")
            os.replace(tmp, p)
        total += n_fixed
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night-dir", required=True, nargs="+")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--exptime", type=float, default=30.0)
    a = ap.parse_args()
    n = 0
    for nd in a.night_dir:
        print(f"=== {nd}")
        n += repair(nd, check=a.check, exptime=a.exptime)
    print(f"[repair3v] {'would populate' if a.check else 'populated'} chi2 on {n} alert record(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
