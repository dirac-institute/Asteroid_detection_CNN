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
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

PIXSCALE = 0.2  # arcsec/px



def _wcs_from_json(s):
    """astropy WCS from a manifest `wcs_json` column (annotate_manifest_wcs.py: Butler SkyWcs ->
    getFitsMetadata FITS-approximation cards as a JSON dict). None if missing/invalid/non-celestial."""
    if not isinstance(s, str) or not s.strip():
        return None
    from astropy.wcs import WCS as _W
    try:
        h = fits.Header()
        for k, v in json.loads(s).items():
            if k in ("COMMENT", "HISTORY") or v is None:
                continue
            h[k] = v
        w = _W(h)
        return w if w.has_celestial else None
    except Exception:
        return None


def _wcs_any(hdr):
    """WCS from a diffim header: primary FITS-WCS (DP2 stage4) or the alternate 'A' key if it is
    celestial. RAISES if neither is a sky WCS -- newer DRP outputs (e.g. DM-53195) keep the exact
    SkyWcs only in archive HDUs and write 'A' as a CTYPE='PIXEL' bookkeeping transform; silently
    using that produced pixel-valued 'sky' coordinates. For those, annotate the manifest with
    wcs_json (annotate_manifest_wcs.py) -- self-consistent across inject+detect+link."""
    from astropy.wcs import WCS as _W
    try:
        w = _W(hdr)
        if w.has_celestial:
            return w
    except Exception:
        pass
    w = _W(hdr, key="A")
    if not w.has_celestial:
        raise ValueError("no celestial WCS in FITS header (annotate manifest with wcs_json)")
    return w

def read_panels(manifest):
    """Return per-panel (visit, detector, mjd, wcs, nx, ny) + per-visit mjd."""
    m = pd.read_csv(manifest)
    panels = []
    nbad = 0
    for _, r in m.iterrows():
        try:
            wj = _wcs_from_json(getattr(r, "wcs_json", None))
            with fits.open(r.fits_path, memmap=True) as h:
                hdr = h[1].header; h0 = h[0].header
                nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
                w = wj or _wcs_any(hdr)
                mjd = h0.get("MJD-AVG") or h0.get("MJD-OBS")
                if mjd is None:
                    da = h0.get("DATE-AVG"); mjd = Time(da, format="isot").mjd if da else np.nan
            c = w.all_pix2world([[nx/2, ny/2]], 0)[0]
            # coordinate sanity: a sky position, not pixels (the failure mode wcs_json exists for)
            if not (np.isfinite(c).all() and -90.0 <= float(c[1]) <= 90.0):
                nbad += 1; continue
            panels.append(dict(visit=int(r.visit), detector=int(r.detector), mjd=float(mjd),
                               wcs=w, nx=nx, ny=ny, cra=float(c[0]), cdec=float(c[1])))
        except Exception:
            nbad += 1
            continue
    if nbad:
        print(f"[orbits] WARNING: {nbad}/{len(m)} panels skipped (unreadable or no celestial WCS)", flush=True)
    if not panels:
        raise SystemExit("[orbits] FATAL: 0 usable panels -- no celestial WCS? (annotate_manifest_wcs.py)")
    print(f"[orbits] {len(panels)}/{len(m)} manifest panels usable", flush=True)
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
    # FAINT regime: sample the target DETECTION-SNR (log-uniform) down to ~2 -- the sub-5sigma fast movers
    # ADCNN targets -- and convert to a magnitude via the single-visit 5sigma depth + trail SNR dilution.
    ap.add_argument("--snr-min", type=float, default=2.0, help="faintest target detection-SNR (the point)")
    ap.add_argument("--snr-max", type=float, default=30.0)
    ap.add_argument("--m5", type=float, default=24.0, help="nominal single-visit 5sigma POINT-source depth (mag)")
    ap.add_argument("--psf-fwhm-px", type=float, default=3.77, help="PSF FWHM (px); trail dilution = sqrt(trail/FWHM)")
    ap.add_argument("--mag-min", type=float, default=18.0, help="bright clamp on the SNR-derived magnitude")
    ap.add_argument("--mag-max", type=float, default=27.0, help="faint clamp on the SNR-derived magnitude")
    ap.add_argument("--retime-map", default=None, help="retime_cadence.py output: visit,mjd_retimed (re-stamp times)")
    ap.add_argument("--exptime", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    panels = read_panels(a.manifest)
    visits = sorted({p["visit"] for p in panels})
    vmjd = {v: np.median([p["mjd"] for p in panels if p["visit"] == v]) for v in visits}
    if a.retime_map:                          # re-stamp each visit to the realistic same-night cadence
        rmap = pd.read_csv(a.retime_map)
        rm = dict(zip(rmap.visit.astype(int), rmap.mjd_retimed.astype(float)))
        miss = [v for v in visits if v not in rm]
        if miss:
            raise SystemExit(f"[orbits] retime-map missing {len(miss)} visits (e.g. {miss[:3]})")
        vmjd = {v: rm[v] for v in visits}
        print(f"[orbits] re-timed {len(visits)} visits from {a.retime_map}", flush=True)
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
        mra = min(0.1, max(0.0, (ra1 - ra0) * 0.25))    # adaptive margin: narrow footprints (field
        mde = min(0.1, max(0.0, (dec1 - dec0) * 0.25))  # slivers) collapsed the fixed 0.1-deg margin
        sra = rng.uniform(ra0 + mra, ra1 - mra); sdec = rng.uniform(dec0 + mde, dec1 - mde)
        rate = float(np.exp(rng.uniform(np.log(a.rate_min), np.log(a.rate_max))))  # deg/day, log-uniform
        pa = rng.uniform(0, 2*np.pi)
        cd = np.cos(np.radians(sdec))
        vx = rate*np.cos(pa)/cd; vy = rate*np.sin(pa)  # deg/day in (ra, dec)
        trail_px = rate * a.exptime/86400.0 / (PIXSCALE/3600.0)  # deg/exposure -> px
        # target detection-SNR (log-uniform, down to ~2) -> magnitude via depth + trail dilution:
        # a trail of length L spreads flux over ~max(1, L/FWHM) PSF footprints, so detection SNR for an
        # optimally-filtered trail ~ point-SNR / sqrt(N_psf). Point-SNR=5 at m5 => for target SNR:
        #   mag = m5 - 2.5*log10( SNR * sqrt(N_psf) / 5 ).
        snr_t = float(np.exp(rng.uniform(np.log(a.snr_min), np.log(a.snr_max))))
        n_psf = max(1.0, trail_px / a.psf_fwhm_px)
        mag = a.m5 - 2.5*np.log10(snr_t * np.sqrt(n_psf) / 5.0)
        mag = float(np.clip(mag, a.mag_min, a.mag_max))
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
                                 trail_length=float(trail_px), beta=beta, mag=mag, snr_target=snr_t))
            n_sight += 1
        truth_rows.append(dict(objID=oid_s, rate_degday=rate, pa_deg=float(np.degrees(pa)),
                               mag=mag, snr_target=snr_t, trail_px=float(trail_px),
                               k_observable=int(len(obj_visits)),
                               n_sightings=n_sight, ra0=sra, dec0=sdec))
    inj = pd.DataFrame(inj_rows); truth = pd.DataFrame(truth_rows)
    if inj.empty:  # fail loud BEFORE writing: 0 sightings = upstream geometry/WCS problem, not a result
        raise SystemExit(f"[orbits] FATAL: {len(truth)} objects -> 0 sightings (footprint/WCS broken?)")
    Path(a.out_inject).parent.mkdir(parents=True, exist_ok=True)
    inj.to_csv(a.out_inject, index=False); truth.to_csv(a.out_truth, index=False)
    ns = truth.n_sightings
    st = truth.snr_target
    print(f"[orbits] {len(truth)} objects -> {len(inj)} sightings | n_sightings: "
          f">=2 {int((ns>=2).sum())}, >=3 {int((ns>=3).sum())}, >=5 {int((ns>=5).sum())}, max {int(ns.max())}", flush=True)
    print(f"[orbits] target-SNR: <5 {int((st<5).sum())} ([2,5) faint), 5-10 {int(((st>=5)&(st<10)).sum())}, "
          f">=10 {int((st>=10).sum())} | mag [{truth.mag.min():.1f},{truth.mag.max():.1f}]", flush=True)
    print(f"[orbits] panels with injections: {inj.groupby(['visit','detector']).ngroups} | -> {a.out_inject}", flush=True)


if __name__ == "__main__":
    main()
