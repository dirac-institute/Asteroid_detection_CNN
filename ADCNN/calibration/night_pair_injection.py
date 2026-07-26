#!/usr/bin/env python3
"""Build a labelled 2-visit calibration set on a REAL night: inject pair-consistent movers into
the night's own repeat-pointing regions.

Why (pair-ranking-null-problem): to rank alerts by a likelihood ratio you must model
p(features | NOT a real mover). Fitting that on off-ecliptic injection fields gets the wrong
answer, because their negatives are faint random chance links, whereas a real survey night's
negatives are STRUCTURED residuals -- static template subtractions, satellite trains -- that are
bright and morphologically real, and therefore score high on exactly the photometric features
(CNN score, mf_snr) that separate the classes in the sparse-field null. Measured on 20260630:
CNN score ranks validated alerts near the BOTTOM of an 11k stream while orbit-fit chi2 puts them
in the top 2%. The only way to settle the weighting is to make the negative class the real
night's own FP population -- i.e. inject into that night's pixels and run the real linker.

What this does: a night's visits do NOT tile one field; 20260630 spans 11 h and 18 pointings, so
injecting over the global sky bounding box lands almost everything in empty gaps between disjoint
pointings (measured: 6000 objects -> 6 sightings, none paired). Instead this groups visits by
BORESIGHT, keeps groups that actually admit a 2-visit link (>=2 visits separated by a gap inside
the linker window -- a 1-minute repeat cannot show 1-8 deg/day motion, a 92-minute one is outside
it), and runs sim_orbits per group so objects land on real panels covered by BOTH visits.

Output: one inject CSV for the whole night (objIDs made unique per group) + a truth CSV, ready
for `detect_night --inject`, then mask_flags -> link_2visit -> label -> fit.

Usage:
  python -m ADCNN.calibration.night_pair_injection --manifest <wcs-annotated manifest> \
      --dets <night dets csv, for per-visit MJD> --out-dir outputs/runs/run_calib_<night> \
      [--n-objects 3000] [--gap-min 5] [--gap-max 75]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
PIXSCALE = 0.2


def _panel_center(wcs_json):
    from astropy.io import fits
    from astropy.wcs import WCS
    h = fits.Header()
    for k, v in json.loads(wcs_json).items():
        if k in ("COMMENT", "HISTORY") or v is None:
            continue
        h[k] = v
    return WCS(h).all_pix2world([[2036, 2000]], 0)[0]


def visit_table(manifest, dets, sample=25):
    """-> DataFrame(visit, ra, dec, npanel, mjd): per-visit boresight and epoch."""
    m = pd.read_csv(manifest)
    if "wcs_json" not in m.columns:
        raise SystemExit("manifest lacks wcs_json -- run annotate_manifest_wcs.py first")
    rows = []
    for v, g in m.groupby("visit"):
        c = np.array([_panel_center(s) for s in g.wcs_json.head(sample)])
        rows.append((int(v), float(np.median(c[:, 0])), float(np.median(c[:, 1])), len(g)))
    d = pd.DataFrame(rows, columns=["visit", "ra", "dec", "npanel"])
    mj = pd.read_csv(dets, usecols=["visit", "mjd"]).groupby("visit").mjd.median()
    d["mjd"] = d.visit.map(mj)
    return d.dropna(subset=["mjd"]).sort_values("visit").reset_index(drop=True)


def pointing_groups(d, sep_deg=1.0):
    """Cluster visits by boresight (single-link within sep_deg)."""
    d = d.copy(); d["grp"] = -1
    g = 0
    for i in d.index:
        if d.at[i, "grp"] >= 0:
            continue
        sep = np.hypot((d.ra - d.at[i, "ra"]) * np.cos(np.radians(d.dec)), d.dec - d.at[i, "dec"])
        d.loc[sep < sep_deg, "grp"] = g
        g += 1
    return d


def usable_groups(d, gap_min, gap_max):
    """Groups that can actually yield a 2-visit link: >=2 visits whose epoch gap is inside the
    linker's window. Too short => a 1-8 deg/day mover has not moved measurably; too long => the
    pair is outside max_arc_2v_min and the linker will never form it."""
    out = []
    for gg, sub in d.groupby("grp"):
        if len(sub) < 2:
            continue
        gaps = [(b - a) * 1440 for a, b in zip(sorted(sub.mjd)[:-1], sorted(sub.mjd)[1:])]
        best = min(gaps)
        if gap_min <= best <= gap_max:
            out.append((int(gg), sub, best))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="WCS-annotated manifest of the night")
    ap.add_argument("--dets", required=True, help="the night's dets csv (per-visit MJD source)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-objects", type=int, default=3000, help="objects per pointing group")
    ap.add_argument("--gap-min", type=float, default=5.0, help="min visit gap to be linkable (min)")
    ap.add_argument("--gap-max", type=float, default=75.0, help="max visit gap (the 2v window)")
    ap.add_argument("--rate-min", type=float, default=1.0)
    ap.add_argument("--rate-max", type=float, default=8.0)
    ap.add_argument("--snr-min", type=float, default=2.0)
    ap.add_argument("--snr-max", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = pointing_groups(visit_table(a.manifest, a.dets))
    groups = usable_groups(d, a.gap_min, a.gap_max)
    print(f"[calib] {len(d)} visits, {d.grp.nunique()} pointings -> {len(groups)} linkable groups "
          f"(gap in [{a.gap_min},{a.gap_max}] min)", flush=True)
    if not groups:
        raise SystemExit("[calib] no pointing group admits a 2-visit link on this night")

    man = pd.read_csv(a.manifest)
    injs, truths = [], []
    for gg, sub, gap in groups:
        vis = set(sub.visit)
        sm = man[man.visit.isin(vis)]
        mpath = out / f"manifest_g{gg}.csv"
        sm.to_csv(mpath, index=False)
        ipath, tpath = out / f"inject_g{gg}.csv", out / f"truth_g{gg}.csv"
        print(f"[calib] group {gg}: {len(vis)} visits, gap {gap:.0f} min, {len(sm)} panels", flush=True)
        cmd = [sys.executable, str(REPO / "ADCNN/pipelines/heliolinc/sim_orbits.py"),
               "--manifest", str(mpath), "--out-inject", str(ipath), "--out-truth", str(tpath),
               "--n-objects", str(a.n_objects), "--rate-min", str(a.rate_min),
               "--rate-max", str(a.rate_max), "--snr-min", str(a.snr_min),
               "--snr-max", str(a.snr_max), "--seed", str(a.seed + gg)]
        subprocess.run(cmd, check=True)
        if ipath.exists():
            di = pd.read_csv(ipath); di["objID"] = f"g{gg}_" + di.objID.astype(str)
            injs.append(di)
            if tpath.exists():
                dt = pd.read_csv(tpath); dt["objID"] = f"g{gg}_" + dt.objID.astype(str)
                truths.append(dt)

    inj = pd.concat(injs, ignore_index=True)
    inj.to_csv(out / "inject.csv", index=False)
    n_per = inj.groupby("objID").size()
    paired = int((n_per >= 2).sum())
    print(f"\n[calib] TOTAL {len(inj)} sightings, {n_per.size} objects, {paired} with >=2 sightings "
          f"-> {out}/inject.csv", flush=True)
    if truths:
        pd.concat(truths, ignore_index=True).to_csv(out / "truth.csv", index=False)
    if paired < 100:
        print("[calib] WARNING: few paired objects -- raise --n-objects or check group selection",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
