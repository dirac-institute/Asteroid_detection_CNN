"""Place synthetic moving objects (NEO-like) across the REAL visit sequence of an off-ecliptic field,
producing a per-sighting injection catalog (+ objID truth) for orbit-driven injection into the real
difference images (which carry the genuine FP population and -- off-ecliptic -- no real asteroids).

Each object: a random sky start in the field footprint + a constant on-sky velocity (rate deg/day, PA)
sampled from a NEO-like distribution + a magnitude. For each visit (real MJD) we propagate the object
(linear same-night motion), find which detector panel of that visit covers it (via the panel WCS), and
emit one injection row: ra, dec, x, y, trail_length(px)=rate*exptime/pixscale, beta(image deg)=motion
direction in the pixel frame, mag, objID. Truth = objID-level rate/mag/sightings. Positions are NOT tied
to a real orbit through the real sky (the field is only a realistic-FP substrate) -- per the design,
the images are real, the orbit geometry is synthetic-but-self-consistent.
"""
from __future__ import annotations
import argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

PIXSCALE = 0.2  # arcsec/px


def read_panels(manifest):
    """Return per-panel (visit, detector, mjd, wcs, nx, ny) + per-visit mjd."""
    m = pd.read_csv(manifest)
    panels = []
    for _, r in m.iterrows():
        try:
            with fits.open(r.fits_path, memmap=True) as h:
                hdr = h[1].header; h0 = h[0].header
                w = WCS(hdr); nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
                mjd = h0.get("MJD-AVG") or h0.get("MJD-OBS")
                if mjd is None:
                    da = h0.get("DATE-AVG"); mjd = Time(da, format="isot").mjd if da else np.nan
            c = w.all_pix2world([[nx/2, ny/2]], 0)[0]
            panels.append(dict(visit=int(r.visit), detector=int(r.detector), mjd=float(mjd),
                               wcs=w, nx=nx, ny=ny, cra=float(c[0]), cdec=float(c[1])))
        except Exception:
            continue
    return panels


def footprint(panels):
    cras = [p["cra"] for p in panels]; cdecs = [p["cdec"] for p in panels]
    return min(cras), max(cras), min(cdecs), max(cdecs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-inject", required=True, help="per-sighting injection catalog csv")
    ap.add_argument("--out-truth", required=True, help="objID-level truth csv")
    ap.add_argument("--n-objects", type=int, default=300)
    ap.add_argument("--rate-min", type=float, default=1.0, help="deg/day (log-uniform); >=1 = the trailed regime the pipeline keeps")
    ap.add_argument("--rate-max", type=float, default=8.0)
    ap.add_argument("--mag-min", type=float, default=20.0)
    ap.add_argument("--mag-max", type=float, default=24.5)
    ap.add_argument("--exptime", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    panels = read_panels(a.manifest)
    visits = sorted({p["visit"] for p in panels})
    vmjd = {v: np.median([p["mjd"] for p in panels if p["visit"] == v]) for v in visits}
    by_visit = {v: [p for p in panels if p["visit"] == v] for v in visits}
    ra0, ra1, dec0, dec1 = footprint(panels)
    t0 = min(vmjd.values())
    print(f"[orbits] {len(panels)} panels, {len(visits)} visits, footprint ra[{ra0:.2f},{ra1:.2f}] "
          f"dec[{dec0:.2f},{dec1:.2f}] span {(max(vmjd.values())-t0)*24:.1f}h", flush=True)

    def panel_for(v, ra, dec):
        cd = np.cos(np.radians(dec))
        for p in by_visit[v]:
            # only invert the (SIP-distorted) WCS for panels whose centre is nearby (a detector is
            # ~0.2 deg); all_world2pix diverges for far points.
            if np.hypot((p["cra"] - ra) * cd, p["cdec"] - dec) > 0.25:
                continue
            try:
                x, y = p["wcs"].all_world2pix([[ra, dec]], 0)[0]
            except Exception:
                continue
            if 0 <= x < p["nx"] and 0 <= y < p["ny"]:
                return p, float(x), float(y)
        return None, None, None

    # REAL same-night apparition-count distribution for FAST (>=1 deg/day) NEOs, measured by Sorcha
    # (Granvik orbits x real DP2 cadence): ~half of fast-NEO apparition-nights are SINGLE (no chance of
    # any link), ~22% exactly 2 (only 2-sighting can recover), ~21% are >=3 (3-sighting possible). We
    # impose this per object as a contiguous same-night sub-window of k real visits -> the cadence cap
    # is modelled, so completeness reflects "some asteroids are only ever seen twice".
    KDIST_K = np.array([1, 2, 3, 4, 5, 6])
    KDIST_P = np.array([0.50, 0.22, 0.15, 0.05, 0.03, 0.05]); KDIST_P /= KDIST_P.sum()
    nv = len(visits)

    inj_rows, truth_rows = [], []
    dt_sub = a.exptime / 86400.0  # for beta (intra-exposure motion direction in pixel frame)
    for oid in range(a.n_objects):
        # start near footprint centre-ish (margin) so it stays in for several epochs
        sra = rng.uniform(ra0 + 0.1, ra1 - 0.1); sdec = rng.uniform(dec0 + 0.1, dec1 - 0.1)
        rate = float(np.exp(rng.uniform(np.log(a.rate_min), np.log(a.rate_max))))  # deg/day, log-uniform
        pa = rng.uniform(0, 2*np.pi)
        cd = np.cos(np.radians(sdec))
        vx = rate*np.cos(pa)/cd; vy = rate*np.sin(pa)  # deg/day in (ra, dec)
        mag = float(rng.uniform(a.mag_min, a.mag_max))
        trail_px = rate * a.exptime/86400.0 / (PIXSCALE/3600.0)  # deg/exposure -> px
        # sample this object's same-night apparition count k (cadence cap), then a contiguous window of
        # k real visits; the object is only present (injected) during those k epochs.
        k = int(rng.choice(KDIST_K, p=KDIST_P)); k = min(k, nv)
        s = int(rng.integers(0, nv - k + 1))
        obj_visits = visits[s:s + k]
        t0o = vmjd[obj_visits[0]]
        oid_s = f"SNEO{oid:05d}"; n_sight = 0
        for v in obj_visits:
            t = vmjd[v]; ra = sra + vx*(t - t0o); dec = sdec + vy*(t - t0o)
            p, x, y = panel_for(v, ra, dec)
            if p is None:
                continue
            # beta = motion direction in the pixel frame (intra-exposure step)
            ra_b = ra + vx*dt_sub; dec_b = dec + vy*dt_sub
            xb, yb = p["wcs"].all_world2pix([[ra_b, dec_b]], 0)[0]
            beta = float(np.degrees(np.arctan2(yb - y, xb - x)) % 360.0)
            inj_rows.append(dict(objID=oid_s, visit=v, detector=p["detector"], mjd=t,
                                 ra=float(ra), dec=float(dec), x=x, y=y,
                                 trail_length=float(trail_px), beta=beta, mag=mag))
            n_sight += 1
        truth_rows.append(dict(objID=oid_s, rate_degday=rate, pa_deg=float(np.degrees(pa)),
                               mag=mag, trail_px=float(trail_px), k_observable=int(len(obj_visits)),
                               n_sightings=n_sight, ra0=sra, dec0=sdec))
    inj = pd.DataFrame(inj_rows); truth = pd.DataFrame(truth_rows)
    Path(a.out_inject).parent.mkdir(parents=True, exist_ok=True)
    inj.to_csv(a.out_inject, index=False); truth.to_csv(a.out_truth, index=False)
    ns = truth.n_sightings
    print(f"[orbits] {len(truth)} objects -> {len(inj)} sightings | n_sightings: "
          f">=2 {int((ns>=2).sum())}, >=3 {int((ns>=3).sum())}, >=5 {int((ns>=5).sum())}, max {int(ns.max())}", flush=True)
    print(f"[orbits] panels with injections: {inj.groupby(['visit','detector']).ngroups} | -> {a.out_inject}", flush=True)


if __name__ == "__main__":
    main()
