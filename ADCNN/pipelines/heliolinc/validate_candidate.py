"""Validate same-night trail-state tracks for one night:
  1. Re-link the night, list surviving tracks (tier 3+visit / 2visit), dump member detections.
  2. RECOVERY report for target known objects (e.g. 2018 BJ1, 2025 NY2): how many of each object's
     known sightings ADCNN detected, whether >=2 fell in linkable visits, and whether the linker
     output a matching track (and at which tier).
  3. Randomized-trail-angle NULL test: scramble each trail's PA, re-link + physical_check N times,
     count surviving tracks -> false-link rate for this night (run at the SAME npt/min-epochs used
     for discovery; this is the defensibility gate for the 2-visit tier).

Usage:
  python -m ADCNN.pipelines.heliolinc.validate_candidate --run run_2v_0706 --night 60863 \
      --npt 2 --min-epochs 2 --pa-tol 20 --pa-tol-2v 10 --targets "2018 BJ1" "2025 NY2"
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")
sys.path.insert(0, str(REPO))
from ADCNN.linking.link_2visit import link, physical_check, fit_residual, crossmatch


def scramble(df, rng):
    """Keep each detection's position + trail half-length; randomize the trail ANGLE (breaks the
    trail-PA-vs-motion correlation that real movers have)."""
    o = df.copy()
    cd = np.cos(np.radians(o.dec.to_numpy()))
    half_len = 0.5 * np.hypot((o.ra1 - o.ra0) * cd, o.dec1 - o.dec0).to_numpy()
    ang = rng.uniform(0, np.pi, len(o))
    dx = half_len * np.cos(ang) / np.where(cd == 0, 1, cd); dy = half_len * np.sin(ang)
    o["ra0"] = o.ra - dx; o["ra1"] = o.ra + dx; o["dec0"] = o.dec - dy; o["dec1"] = o.dec + dy
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(OUTPUTS / "runs/run_band"), help="run dir with adcnn_dets_masked.csv + known.csv")
    ap.add_argument("--dets", default=None, help="override masked dets CSV (default <run>/adcnn_dets_masked.csv)")
    ap.add_argument("--known", default=None, help="override known.csv (default <run>/known.csv)")
    ap.add_argument("--night", type=int, required=True, help="integer night = floor(mjd-0.5)")
    ap.add_argument("--npt", type=int, default=2)
    ap.add_argument("--min-epochs", type=int, default=2)
    ap.add_argument("--pa-tol", type=float, default=20.0)
    ap.add_argument("--pa-tol-2v", type=float, default=10.0)
    ap.add_argument("--max-rms", type=float, default=1.0)
    ap.add_argument("--art-frac-max", type=float, default=0.3)
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--ntrial", type=int, default=50)
    ap.add_argument("--targets", nargs="*", default=["2018 BJ1", "2025 NY2"])
    a = ap.parse_args()

    dets = a.dets or f"{a.run}/adcnn_dets_masked.csv"
    known_path = a.known or f"{a.run}/known.csv"
    d = pd.read_csv(dets)
    if "art_frac" in d and a.art_frac_max > 0:
        d = d[d.art_frac < a.art_frac_max]
    if "len_db" in d and a.len_db_min > 0:
        d = d[d.len_db >= a.len_db_min]
    if "score" in d and a.score_min > 0:
        d = d[d.score >= a.score_min]
    d = d.reset_index(drop=True)
    d["night"] = np.floor(d.mjd - 0.5).astype(int)
    dn = d[d.night == a.night].reset_index(drop=True)
    known = pd.read_csv(known_path)
    pc = dict(pa_tol_deg=a.pa_tol, lin_rms_arcsec=a.max_rms, min_epochs=a.min_epochs, pa_tol_2v_deg=a.pa_tol_2v)
    print(f"night {a.night}: {len(dn)} dets after cuts | npt {a.npt} min-epochs {a.min_epochs} "
          f"pa-tol {a.pa_tol}/{a.pa_tol_2v}(2v)\n")

    # 1. surviving tracks
    labels, tracks = link(dn, npt=a.npt, min_visits=a.npt)
    passed = []
    for members in tracks:
        ok, info, n_ep = physical_check(dn, members, **pc)
        if ok:
            passed.append((members, info, n_ep))
    print(f"{len(tracks)} raw tracks -> {len(passed)} pass physical_check "
          f"({sum(n == 2 for _, _, n in passed)} 2visit, {sum(n >= 3 for _, _, n in passed)} 3+visit)\n")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for members, info, n_ep in passed:
        g = dn.iloc[members].sort_values("mjd")
        rms, speed = fit_residual(dn, members)
        obj, frac = crossmatch(dn, members, known, 5.0, 0.02)
        tier = "2visit" if n_ep == 2 else "3+visit"
        print(f"=== [{tier}] {info} | speed {speed:.3f} deg/day rms {rms:.3f}\" match='{obj}' frac={frac}")
        cols = [c for c in ["visit", "detector", "mjd", "ra", "dec", "len_db", "score", "snr", "art_frac"] if c in g.columns]
        print(g[cols].to_string(index=False), "\n")

    # 2. RECOVERY report for target known objects
    matched_objs = {crossmatch(dn, m, known, 5.0, 0.02)[0] for m, _, _ in passed}
    print("=== RECOVERY of target objects ===")
    kn = known.copy(); kn["ObjID"] = kn.ObjID.astype(str)
    for tgt in a.targets:
        ke = kn[kn.ObjID == tgt]
        if not len(ke):
            print(f"  {tgt}: not in known.csv for this night"); continue
        # ADCNN detections near each known sighting (5", 0.02 d)
        det_visits = set()
        for _, r in ke.iterrows():
            sep = np.hypot((dn.ra - r.ra) * np.cos(np.radians(r.dec)), dn.dec - r.dec) * 3600
            near = dn[(sep <= 5.0) & (np.abs(dn.mjd - r.mjd) <= 0.02)]
            if len(near):
                det_visits.add(int(r.visit))
        rec = "RECOVERED" if tgt in matched_objs else "not linked"
        print(f"  {tgt}: {len(ke)} known sightings | ADCNN detected {len(det_visits)}/{len(ke)} "
              f"(visits {sorted(det_visits)}) | linker: {rec}")
    print()

    # 3. NULL test (scramble trail angles)
    rng = np.random.default_rng(12345)
    counts = []
    for _ in range(a.ntrial):
        s = scramble(dn, rng)
        _, tr = link(s, npt=a.npt, min_visits=a.npt)
        counts.append(sum(physical_check(s, m, **pc)[0] for m in tr))
    counts = np.array(counts)
    print(f"NULL TEST (scrambled trail angles, {a.ntrial} trials): mean {counts.mean():.3f} "
          f"surviving tracks/trial, max {counts.max()}, trials with >=1: {(counts >= 1).sum()}/{a.ntrial}")


if __name__ == "__main__":
    main()
