"""Classify HelioLinC-linked tracks into KNOWN / RECOVERED-MISSED / NEW-CANDIDATE, with an orbit-
quality gate that separates real objects from FP chance-clusters.

A real moving object fits an orbit to a small residual (the catalogued asteroids re-discovered in
run_diasrc had posRMS ~180-1000 km, totRMS < a few thousand); FP chance-alignments sit at
40,000-50,000 km. So a track that matches NOTHING known AND is clean (posRMS < --maxposrms,
obsnights >= --minnights) is a credible previously-UNKNOWN asteroid candidate; an unmatched track
with huge RMS is just spurious.

  KNOWN            : matches known.csv (Rubin's catalogued SSObjects) >= min_frac
  RECOVERED-MISSED : matches missed_truth.csv (real MPC objs Rubin failed to link) but not known.csv
  NEW-CANDIDATE    : matches neither AND passes the orbit-quality gate  <-- the discovery output
  SPURIOUS         : matches neither AND fails the gate (FP chance-cluster)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _sep_arcsec(ra1, dec1, ra2, dec2):
    return np.hypot((ra1 - ra2) * np.cos(np.radians(dec2)), dec1 - dec2) * 3600.0


def match(track, cat, tol_arcsec, tol_day, min_frac):
    if cat is None or not len(cat):
        return None, 0.0
    hits = []
    for _, d in track.iterrows():
        c = cat[np.abs(cat.mjd - d.MJD) < tol_day]
        if not len(c):
            continue
        sep = _sep_arcsec(d.RA, d.Dec, c.ra.values, c.dec.values)
        j = int(np.argmin(sep))
        if sep[j] <= tol_arcsec:
            hits.append(c.ObjID.values[j])
    if not hits:
        return None, 0.0
    vc = pd.Series(hits).value_counts()
    return vc.index[0], vc.iloc[0] / len(track)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--known", required=True, help="catalogued SSObject sightings (ObjID,mjd,ra,dec)")
    ap.add_argument("--missed", default=None, help="ss_object_unassociated sightings (optional)")
    ap.add_argument("--tol-arcsec", type=float, default=3.0)
    ap.add_argument("--tol-day", type=float, default=0.02)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--maxposrms", type=float, default=2000.0, help="orbit-quality gate (km) for NEW")
    ap.add_argument("--minnights", type=int, default=3)
    a = ap.parse_args()
    run = Path(a.run)

    lr = pd.read_csv(run / "lr.csv"); lr.columns = [c.lstrip("#").strip() for c in lr.columns]
    rms = pd.read_csv(run / "lr_rms.csv"); rms.columns = [c.lstrip("#").strip() for c in rms.columns]
    known = pd.read_csv(a.known).dropna(subset=["ra", "dec"])
    missed = pd.read_csv(a.missed).dropna(subset=["ra", "dec"]) if a.missed and Path(a.missed).exists() else None

    rows = []
    for cl, tr in lr.groupby("clusternum"):
        q = rms[rms.clusternum == cl]
        posrms = float(q.posRMS.iloc[0]) if len(q) else np.nan
        nnts = int(q.obsnights.iloc[0]) if len(q) and "obsnights" in q else -1
        rating = q.rating.iloc[0] if len(q) and "rating" in q else "?"
        ko, kf = match(tr, known, a.tol_arcsec, a.tol_day, a.min_frac)
        mo, mf = match(tr, missed, a.tol_arcsec, a.tol_day, a.min_frac) if missed is not None else (None, 0.0)
        clean = (posrms < a.maxposrms) and (nnts >= a.minnights)
        if ko is not None and kf >= a.min_frac:
            cls = "KNOWN"
        elif mo is not None and mf >= a.min_frac:
            cls = "RECOVERED-MISSED"
        elif clean:
            cls = "NEW-CANDIDATE"
        else:
            cls = "SPURIOUS"
        rows.append(dict(cluster=int(cl), ndet=len(tr), nnights=nnts, rating=rating,
                         posRMS=round(posrms, 1), cls=cls,
                         known_obj=ko, known_frac=round(kf, 2), missed_obj=mo, missed_frac=round(mf, 2)))
    r = pd.DataFrame(rows)
    print(f"\n=== classify {len(r)} linked tracks (gate: posRMS<{a.maxposrms:.0f}km, >={a.minnights}nt) ===")
    if not len(r):
        print("no linked tracks to classify"); return
    print(r.cls.value_counts().to_string())
    print(f"\nKNOWN re-discovered     : {r[r.cls=='KNOWN'].known_obj.nunique()} distinct asteroids")
    if missed is not None:
        print(f"RECOVERED Rubin-MISSED  : {r[r.cls=='RECOVERED-MISSED'].missed_obj.nunique()} distinct asteroids")
    nc = r[r.cls == "NEW-CANDIDATE"].sort_values("posRMS")
    print(f"NEW (unknown) candidates: {len(nc)} clean unmatched tracks")
    r.to_csv(run / "classified.csv", index=False)
    if len(nc):
        nc.to_csv(run / "new_clean.csv", index=False)
        print("\n-- NEW candidates (clean orbit, matches nothing known) --")
        print(nc[["cluster", "ndet", "nnights", "rating", "posRMS"]].head(20).to_string(index=False))
    print(f"\n-> {run}/classified.csv" + (f", {run}/new_clean.csv" if len(nc) else ""))


if __name__ == "__main__":
    main()
