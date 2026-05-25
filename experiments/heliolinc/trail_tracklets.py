"""Build HelioLinC tracklets directly from TRAILED detections — one trail = one tracklet.

A single trailed exposure already encodes a state vector: the trail's two endpoints are the
object's positions at exposure-start and exposure-end (T_exp apart), i.e. position + on-sky
velocity from ONE visit. So we can skip make_tracklets' "≥2 detections per night within
maxtime" pairing and synthesize the pairdets.csv + pairs.txt that HelioLinC consumes directly:

  per trailed detection (ra,dec @ mjd; endpoints ra0,dec0 / ra1,dec1):
    - two pairdet rows at mjd ∓ T_exp/2  (the endpoints, with interpolated observer XYZ)
    - one "T" tracklet in pairs.txt linking them
  the head/tail sense is unknown, so emit BOTH orderings (A→B and B→A); the wrong-direction
  tracklet simply won't cluster into a consistent (r,rdot) orbit and is dropped by heliolinc.

This unlocks the ~half of object-nights that have only a single coverage (where pair-tracklets
fail) and tightens linking via the per-trail velocity — most valuable for fast movers/NEOs.

Two entry points:
  build_tracklet_files(dets, earth_file, out_dir, exptime_s=30)   # dets must carry endpoints
  validate-on-truth (CLI): synthesize idealized trails from a known-object ephemeris and link.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
HL = REPO / "experiments/heliolinc"
COLFORMAT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"


def parse_earth_vectors(path):
    """Parse a JPL Horizons VECTORS table -> (mjd[N], XYZ[N,3] km, heliocentric)."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    try:
        i0 = next(i for i, l in enumerate(lines) if "$$SOE" in l) + 1
        i1 = next(i for i, l in enumerate(lines) if "$$EOE" in l)
    except StopIteration:
        raise ValueError("no $$SOE/$$EOE block in Earth file")
    import re
    mjds, xyz = [], []
    block = lines[i0:i1]
    j = 0
    while j < len(block):
        head = block[j].strip()
        jd = float(head.split("=")[0].strip())          # JDTDB at start of each 4-line record
        xline = block[j + 1]
        def g(tag):
            m = re.search(rf"{tag}\s*=\s*([-+]?[\d.]+[ED][-+]?\d+)", xline)
            return float(m.group(1).replace("D", "E"))
        xyz.append([g("X"), g("Y"), g("Z")])
        mjds.append(jd - 2400000.5)
        j += 4
    return np.array(mjds), np.array(xyz)


def observer_xyz(earth_mjd, earth_xyz, mjd):
    """Linear-interpolate heliocentric observer (Earth) position [km] at mjd (geocentric approx)."""
    return np.stack([np.interp(mjd, earth_mjd, earth_xyz[:, k]) for k in range(3)], axis=-1)


def build_tracklet_files(dets, earth_file, out_dir, *, exptime_s=30.0, both_orderings=True,
                         mag_col="mag", band_col="band", obscode="I11"):
    """dets: DataFrame with mjd, ra, dec, ra0, dec0, ra1, dec1 (endpoints, deg).
    Writes pairdets.csv + pairs.txt + colformat.txt into out_dir (HelioLinC inputs)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emjd, exyz = parse_earth_vectors(earth_file)
    half = (exptime_s / 86400.0) / 2.0

    pd_rows = []       # pairdet rows
    pair_blocks = []   # ("T ..." header, [member indices])
    idx = 0
    for _, d in dets.iterrows():
        mag = float(d.get(mag_col, 21.0)); band = str(d.get(band_col, "r"))[:1] or "r"
        ends = [(d.ra0, d.dec0, d.mjd - half), (d.ra1, d.dec1, d.mjd + half)]
        ids = []
        for (ra, dec, t) in ends:
            ox, oy, oz = observer_xyz(emjd, exyz, t)
            pd_rows.append(dict(MJD=f"{t:.7f}", RA=f"{ra:.7f}", Dec=f"{dec:.7f}",
                                observerX=f"{ox:.3f}", observerY=f"{oy:.3f}", observerZ=f"{oz:.3f}",
                                stringID=idx, mag=f"{mag:.3f}", band=band, obscode=obscode, origindex=idx))
            ids.append(idx); idx += 1
        a, b = ids
        orderings = [(a, b, ends[0], ends[1])]
        if both_orderings:
            orderings.append((b, a, ends[1], ends[0]))
        for i0, i1, e0, e1 in orderings:
            pair_blocks.append((f"T {i0} {i1} {e0[0]:.6f} {e0[1]:.6f} {e1[0]:.6f} {e1[1]:.6f} 2",
                                [i0, i1]))

    cols = ["MJD", "RA", "Dec", "observerX", "observerY", "observerZ", "stringID", "mag",
            "band", "obscode", "origindex"]
    pdf = pd.DataFrame(pd_rows)[cols]
    with open(out_dir / "pairdets.csv", "w") as f:
        f.write("#" + ",".join(cols) + "\n")
        pdf.to_csv(f, header=False, index=False)
    with open(out_dir / "pairs.txt", "w") as f:
        for head, members in pair_blocks:
            f.write(head + "\n")
            for m in members:
                f.write(f"{m}\n")
    (out_dir / "colformat.txt").write_text(COLFORMAT)
    print(f"[trail->tracklets] {len(dets)} trails -> {len(pd_rows)} pairdets, "
          f"{len(pair_blocks)} tracklets -> {out_dir}", flush=True)
    return out_dir


def synth_trails_from_ephemeris(known, exptime_s=30.0):
    """Controlled validation: turn each ephemeris sighting into an idealized trail using the
    object's local sky-motion (finite-difference of its own ephemeris). Returns dets with
    endpoints. This tests the LINKING side (does heliolinc cluster single-trail tracklets?)."""
    half = (exptime_s / 86400.0) / 2.0
    out = []
    for oid, g in known.sort_values("mjd").groupby("ObjID"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        t = g.mjd.to_numpy(); ra = g.ra.to_numpy(); dec = g.dec.to_numpy()
        cosd = np.cos(np.radians(dec))
        # local rate via central finite difference on the ephemeris (deg/day)
        dradt = np.gradient(ra, t); ddecdt = np.gradient(dec, t)
        for i in range(len(g)):
            out.append(dict(ObjID=oid, mjd=t[i], ra=ra[i], dec=dec[i],
                            ra0=ra[i] - dradt[i] * half, dec0=dec[i] - ddecdt[i] * half,
                            ra1=ra[i] + dradt[i] * half, dec1=dec[i] + ddecdt[i] * half,
                            mag=float(g.get("mag", pd.Series([21.0])).iloc[i]) if "mag" in g else 21.0,
                            band="r"))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--known", default=str(HL / "run_wide/known.csv"))
    ap.add_argument("--earth", default=str(HL / "run_disco/Earth1day2020s_02a.txt"))
    ap.add_argument("--out", default=str(HL / "run_trail_validate"))
    ap.add_argument("--min-nights", type=int, default=3, help="keep objects seen on >= this many nights")
    ap.add_argument("--single-night-only", action="store_true",
                    help="keep only object-nights with a SINGLE sighting (the regime where pair-tracklets fail)")
    ap.add_argument("--exptime", type=float, default=30.0)
    a = ap.parse_args()

    known = pd.read_csv(a.known)
    known["night"] = np.floor(known.mjd - 0.5).astype(int)
    nn = known.groupby("ObjID").night.transform("nunique")
    known = known[nn >= a.min_nights]
    if a.single_night_only:
        known = known[known.groupby(["ObjID", "night"])["mjd"].transform("size") == 1]
    dets = synth_trails_from_ephemeris(known, a.exptime).reset_index(drop=True)
    print(f"[validate] {dets.ObjID.nunique()} objects, {len(dets)} sightings -> trail-tracklets", flush=True)
    build_tracklet_files(dets, a.earth, a.out, exptime_s=a.exptime)
    # truth map: pairdet index k (0..2N-1) -> ObjID of its source trail (det k//2)
    tmap = pd.DataFrame({"pairdet": range(2 * len(dets)),
                         "ObjID": np.repeat(dets.ObjID.to_numpy(), 2)})
    tmap.to_csv(Path(a.out) / "truthmap.csv", index=False)


if __name__ == "__main__":
    main()
