#!/usr/bin/env python3
"""Select the top-N ripple-free alerts by P(real) and reindex the cutout cache to match.

Given a pReal-ordered top-K alerts.jsonl, its cutout cache, and the morphology flags
(ADCNN.qa.alert_morphology), keep the first N alerts that are NOT flagged as bright-star dipole
rings, and emit a new alerts.jsonl + a cutout cache reindexed to the kept set (the cache is keyed by
alert POSITION, so it must be re-sliced, not just truncated). The result is the 1k-cadence product:
~N ripple-clean, pReal-ranked alerts for visual vetting.

Usage:
  python -m ADCNN.qa.select_clean --alerts topk.jsonl --morph morph.npz --cutouts topk_cutouts.npz \
      --n 1000 --out-alerts alerts.jsonl --out-cutouts cutouts.npz
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def _confident_fp(a):
    """CONFIDENTLY-false-positive flag classes to drop for a pure sample. NOT single-counterpart
    stationary: that flag fires on ONE source within 3" of ONE endpoint, which in a dense field is
    86% chance coincidence (median n_counterparts=1) -- dropping it would shed real slow movers
    (rank-0 20260705 = a real 1.0 deg/d mover flagged this way). Only the unambiguous cases:
      both-endpoint stationary   -- two recurring sources linked (a chance link of two statics)
      n_counterparts >= 3        -- an endpoint that clearly recurs across visits (a real static)
      static-line (staticVeto)   -- a catalogued static source is a member
      train (trainVeto)          -- a satellite-train / static-line member set
    """
    st = a.get("stationarity") or {}
    e1 = st.get("e1") or {}; e2 = st.get("e2") or {}
    if st.get("vetoStationary"):
        if e1.get("counterpart") and e2.get("counterpart"):
            return "stationary-both"
        if (e1.get("n_counterparts") or 0) + (e2.get("n_counterparts") or 0) >= 3:
            return "stationary-recurring"
    if (a.get("staticVeto") or {}).get("nStaticMembers", 0) > 0:
        return "static-line"
    if (a.get("trainVeto") or {}).get("vetoTrain"):
        return "train"
    return None


def select(alerts_path, morph_npz, cutouts_npz, n, out_alerts, out_cutouts, mode="confident"):
    lines = open(alerts_path).read().splitlines()
    alerts = [json.loads(l) for l in lines]
    K = len(lines)
    # VALIDATE THE INPUT CACHE FIRST. This function reindexes by POSITION and then stamps a fresh
    # fingerprint computed from the alerts file -- so without this check a stale or permuted cache is
    # laundered into a product every downstream guard accepts. Pixel-proven on real 0710 stamps: 20 of
    # 30 alerts carried the wrong alert's pixels (mean 66.66% of pixels differing) and the guard
    # PASSED. Before the fingerprint existed the row-by-row fallback caught it, so this was a
    # regression the fingerprint introduced, not a pre-existing hole.
    from ADCNN.qa.alert_sheets import _assert_cache_matches
    _assert_cache_matches(alerts_path, cutouts_npz, K)
    mo = np.load(morph_npz)
    ripple = mo["ripple"]
    if len(ripple) != K:
        raise SystemExit(f"morph has {len(ripple)} rows but alerts has {K}")
    from collections import Counter
    dropped = Counter()

    def drop(i):
        if ripple[i]:
            dropped["ring"] += 1; return True
        if mode == "rings":
            return False
        a = alerts[i]
        if mode == "clean" and (a.get("stationarity") or {}).get("vetoStationary"):
            dropped["stationary-any"] += 1; return True   # aggressive: any stationary
        fp = _confident_fp(a)
        if fp:
            dropped[fp] += 1; return True
        return False

    # alerts.jsonl is pReal-ordered, so the first N survivors ARE the top-N clean-enough by pReal.
    # DROP first, then count -- so the product is always N, never silently short.
    kept = [i for i in range(K) if not drop(i)][:n]
    remap = {old: new for new, old in enumerate(kept)}
    scanned = kept[-1] + 1 if kept else K
    # write filtered alerts (pReal order preserved)
    with open(out_alerts, "w") as f:
        for old in kept:
            f.write(lines[old] + "\n")
    # reindex the cutout cache: keep only kept alerts, remap their index to the new position
    z = dict(np.load(cutouts_npz))
    keptset = set(kept)

    def _sub(idx_key, arrays):
        idx = z[idx_key]
        m = np.array([int(a) in keptset for a in idx])
        newidx = np.array([remap[int(a)] for a in idx[m]], np.int32)
        order = np.argsort(newidx, kind="stable")
        out = {}
        for k in arrays:
            out[k] = z[k][m][order]
        out[idx_key] = newidx[order]
        return out

    zoom_keys = [k for k in ("stamps", "epoch", "ok", "zoom_ends", "visit", "detector") if k in z]
    wide_keys = [k for k in ("wide", "wide_pos", "wide_apx", "wide_ok", "wide_ends") if k in z]
    newz = {}
    newz.update(_sub("alert", zoom_keys))
    newz.update(_sub("wide_alert", wide_keys))
    np.savez_compressed(out_cutouts, **newz)
    # meta so the render cache-match guard passes
    src_meta = os.path.splitext(cutouts_npz)[0] + "_meta.json"
    base = json.load(open(src_meta)) if os.path.exists(src_meta) else {}
    base.update(dict(n_alerts=len(kept), n_zoom=len(newz["alert"]), n_wide=len(newz["wide_alert"]),
                     alerts=os.path.abspath(out_alerts)))
    # The carried-over sidecar names the SOURCE sequence; this cache is the reindexed subset, so its
    # fingerprint must be recomputed or every renderer would refuse the product we just built.
    from ADCNN.qa.cache_identity import epoch_digest, FINGERPRINT_VERSION
    base["alerts_fingerprint"] = epoch_digest([json.loads(lines[_o]) for _o in kept])
    base["fingerprint_version"] = FINGERPRINT_VERSION
    json.dump(base, open(os.path.splitext(out_cutouts)[0] + "_meta.json", "w"), indent=2)
    drops = "  ".join(f"{k}:{v}" for k, v in sorted(dropped.items()))
    print(f"[select] mode={mode}: kept {len(kept)} of top {K} (scanned {scanned}; dropped {drops or 'none'}) "
          f"-> {out_alerts}", flush=True)
    return len(kept)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True, help="pReal-ordered top-K alerts.jsonl")
    ap.add_argument("--morph", required=True)
    ap.add_argument("--cutouts", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out-alerts", required=True)
    ap.add_argument("--out-cutouts", required=True)
    ap.add_argument("--mode", choices=["rings", "confident", "clean"], default="confident",
                    help="rings=dipoles only; confident=+both-endpoint/recurring stationary+static-line"
                         "+train (default, purity with ~no completeness loss); clean=+ALL stationary "
                         "(aggressive, sheds real slow movers)")
    a = ap.parse_args(argv)
    select(a.alerts, a.morph, a.cutouts, a.n, a.out_alerts, a.out_cutouts, a.mode)


if __name__ == "__main__":
    sys.exit(main())
