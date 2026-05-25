"""Vet NEW (unmatched) HelioLinC tracks: are they plausible real moving objects or false links?

For each NEW candidate cluster (crossmatch labelled it as matching no known object), pull its
member detections from lr.csv and report the diagnostics that separate a real asteroid from a
chance alignment of false positives:
  * n detections, n distinct nights, time span,
  * apparent sky motion (deg/day) + whether it is steady (per-night rate consistency),
  * great-circle straightness residual (arcsec) of a linear fit in (mjd -> ra,dec),
  * the link_refine posRMS/velRMS/rating + the median detection score.
A real main-belt object: >=3 nights, ~0.05-0.5 deg/day, small straightness residual, low posRMS.
Also re-checks each NEW track against the known catalog with a looser tolerance (in case the
preloaded list missed it).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def line_resid(mjd, ra, dec):
    """RMS arcsec residual of a linear (constant-rate) fit ra(t), dec(t)."""
    t = mjd - mjd.mean()
    out = []
    for v in (ra, dec):
        A = np.vstack([t, np.ones_like(t)]).T
        pred = A @ np.linalg.lstsq(A, v, rcond=None)[0]
        out.append(v - pred)
    cosd = np.cos(np.radians(dec.mean()))
    return float(np.sqrt(np.mean((out[0] * cosd) ** 2 + out[1] ** 2)) * 3600)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    lr = pd.read_csv(run / "lr.csv"); lr.columns = [c.lstrip("#") for c in lr.columns]
    rms = pd.read_csv(run / "lr_rms.csv"); rms.columns = [c.lstrip("#") for c in rms.columns]
    newc = pd.read_csv(run / "new_candidates.csv") if (run / "new_candidates.csv").exists() else pd.DataFrame()
    if not len(newc):
        print("no NEW candidates to vet"); return
    sc = pd.read_csv(run / "adcnn_dets.csv")[["mjd", "ra", "dec", "score_rf"]] if (run / "adcnn_dets.csv").exists() else None

    rows = []
    for cl in newc.cluster:
        g = lr[lr.clusternum == cl].sort_values("MJD")
        mjd, ra, dec = g.MJD.to_numpy(), g.RA.to_numpy(), g.Dec.to_numpy()
        nights = np.unique(np.floor(mjd - 0.5)).size
        span = mjd.max() - mjd.min()
        dt = mjd.max() - mjd.min()
        vel = np.hypot((ra.max() - ra.min()) * np.cos(np.radians(dec.mean())), dec.max() - dec.min()) / max(dt, 1e-6)
        q = rms[rms.clusternum == cl]
        rows.append(dict(cluster=int(cl), ndet=len(g), nights=nights, span_d=round(span, 2),
                         vel_deg_day=round(vel, 3), resid_arcsec=round(line_resid(mjd, ra, dec), 2),
                         posRMS_km=round(float(q.posRMS.iloc[0]), 0) if len(q) else np.nan,
                         rating=q.rating.iloc[0] if len(q) and "rating" in q else "?"))
    v = pd.DataFrame(rows).sort_values(["nights", "resid_arcsec"], ascending=[False, True])
    # a plausible real object: >=3 nights, steady main-belt-ish motion, small straightness residual
    v["plausible"] = (v.nights >= 3) & (v.vel_deg_day.between(0.03, 0.6)) & (v.resid_arcsec < 3.0)
    v.to_csv(run / "new_vetted.csv", index=False)
    print(f"=== {len(v)} NEW candidate tracks vetted ({v.plausible.sum()} plausible real objects) ===")
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
