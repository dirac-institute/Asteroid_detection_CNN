"""Crossmatch HelioLinC-linked tracks against a known-object catalog -> CONFIRMED + NEW.

A linked track (a cluster in ``lr.csv``, one row per clustered detection) is a real moving
object whose orbit was consistent across multiple nights (the linker's minobsnights gate;
2 for fast NEOs). We label each track by matching its
detections, in (RA, Dec) and time, to a catalog of *known* object sightings:

  * CONFIRMED  -- a single known ObjID accounts for >= `min_frac` of the track's detections
                  (we re-discovered a catalogued asteroid from ADCNN detections alone).
  * NEW        -- a quality track (PURE / low RMS / multi-night) that matches no known object:
                  a candidate previously-undiscovered asteroid, to be vetted.

Default known catalog = the per-sighting truth for the test field (built from the DP2 SSObject
ephemerides). Point ``--known`` at a wider catalog (e.g. an MPC/SSObject export) for a real
discovery search. Matching is purely positional+temporal, so it works whether or not the linked
detections carry any truth label.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
# the per-sighting truth with MJD (built by build_catalog.py): ObjID, mjd, ra, dec — known DP2 objects
DEFAULT_KNOWN = REPO / "ADCNN/pipelines/heliolinc/run_truth/truth_dets.csv"


def _sky_sep_arcsec(ra1, dec1, ra2, dec2):
    dra = ((ra1 - ra2 + 180.0) % 360.0) - 180.0   # wrap to [-180,180] so RA 0/360 seam doesn't blow up
    return np.hypot(dra * np.cos(np.radians(dec2)), dec1 - dec2) * 3600.0


def label_track(track: pd.DataFrame, known: pd.DataFrame, tol_arcsec: float, tol_day: float):
    """Return (dominant ObjID, match fraction) for one track, or (None, 0.0) if unmatched."""
    hits = []
    for _, d in track.iterrows():
        cand = known[np.abs(known.mjd - d.MJD) < tol_day]
        if not len(cand):
            continue
        sep = _sky_sep_arcsec(d.RA, d.Dec, cand.ra.values, cand.dec.values)
        j = int(np.argmin(sep))
        if sep[j] <= tol_arcsec:
            hits.append(cand.ObjID.values[j])
    if not hits:
        return None, 0.0
    vc = pd.Series(hits).value_counts()
    return vc.index[0], vc.iloc[0] / len(track)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir holding lr.csv + lr_rms.csv")
    ap.add_argument("--known", default=str(DEFAULT_KNOWN), help="known-object sightings catalog (ObjID,ra,dec,mjd)")
    ap.add_argument("--tol-arcsec", type=float, default=3.0)
    ap.add_argument("--tol-day", type=float, default=0.02, help="time match window (~30 min)")
    ap.add_argument("--min-frac", type=float, default=0.5, help="min fraction of a track's dets matching one ObjID")
    a = ap.parse_args()

    run = Path(a.run)
    lr = pd.read_csv(run / "lr.csv")
    lr.columns = [c.lstrip("#") for c in lr.columns]
    rms = pd.read_csv(run / "lr_rms.csv")
    rms.columns = [c.lstrip("#") for c in rms.columns]

    known = pd.read_csv(a.known).dropna(subset=["ra", "dec"])
    if "mjd" not in known.columns:  # truth file carries MJD under a different name in some builds
        raise ValueError(f"known catalog needs an 'mjd' column; has {list(known.columns)}")

    confirmed, new = [], []
    for cl, track in lr.groupby("clusternum"):
        obj, frac = label_track(track, known, a.tol_arcsec, a.tol_day)
        q = rms[rms.clusternum == cl]
        rating = q.rating.iloc[0] if len(q) and "rating" in q.columns else "?"
        nnights = int(q.obsnights.iloc[0]) if len(q) and "obsnights" in q.columns else track.MJD.apply(
            lambda m: int(np.floor(m - 0.5))).nunique()
        rec = dict(cluster=int(cl), ndet=len(track), nnights=nnights, rating=rating,
                   obj=obj, match_frac=round(frac, 2))
        (confirmed if (obj is not None and frac >= a.min_frac) else new).append(rec)

    cdf, ndf = pd.DataFrame(confirmed), pd.DataFrame(new)
    n_known_obj = cdf.obj.nunique() if len(cdf) else 0
    print(f"\n=== crossmatch ({len(lr.clusternum.unique())} linked tracks) ===")
    print(f"CONFIRMED (known) : {len(cdf)} tracks -> {n_known_obj} distinct known asteroids re-discovered")
    print(f"NEW candidates    : {len(ndf)} tracks (no known match)")
    if len(cdf):
        cdf.to_csv(run / "confirmed.csv", index=False)
        print("\n-- confirmed (sample) --")
        print(cdf.sort_values("ndet", ascending=False).head(10).to_string(index=False))
    if len(ndf):
        ndf.to_csv(run / "new_candidates.csv", index=False)
        print("\n-- NEW candidates (vet these) --")
        print(ndf.sort_values(["nnights", "ndet"], ascending=False).to_string(index=False))
    print("\nCROSSMATCH DONE")


if __name__ == "__main__":
    main()
