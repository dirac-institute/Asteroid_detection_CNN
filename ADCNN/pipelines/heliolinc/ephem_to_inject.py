"""Convert a SORCHA main-output catalogue (Granvik NEOs propagated through the REAL DP2 cadence of a
single off-ecliptic field) into the per-sighting injection catalogue (inject.csv) the trail injector and
detectors consume -- the MULTI-NIGHT, orbit-driven analogue of sim_orbits.py.

Unlike sim_orbits (linear same-night motion, synthetic geometry), here every sighting comes from a real
2-body orbit propagated by Sorcha, so the SAME ObjID appears at self-consistent positions across MULTIPLE
NIGHTS -> the injected objects form genuine multi-night arcs that exercise heliolinc's orbital linking
faithfully. The apparent magnitude is Sorcha's `trailedSourceMag` (H + distance + phase), so the faint
end is physical (the faint-H / distant-geometry tail), not imposed. We re-derive the DETECTION-SNR in OUR
diffims from that magnitude + the single-visit depth + trail dilution (same model as sim_orbits) purely
for SNR-binned reporting; it does not change the injected pixels (those are set by mag).

Sorcha FieldID == our visit id (build_pointing_db wrote observationId = visit.id). For each in-window
sighting we map (RA_deg, Dec_deg) to the diffim panel that covers it via the real per-panel WCS.
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



def _wcs_any(hdr):
    """WCS from a diffim header: primary FITS-WCS (DP2 stage4) or the alternate 'A' WCS that newer
    LSST DRP outputs write (exact SkyWcs lives in archive HDUs; 'A' is the FITS approximation --
    self-consistent for inject+detect+link, which all use the same transform)."""
    from astropy.wcs import WCS as _W
    try:
        w = _W(hdr)
        if w.has_celestial:
            return w
    except Exception:
        pass
    return _W(hdr, key="A")

def read_panels(manifest):
    """Per-panel (visit, detector, mjd, wcs, nx, ny, centre) from the diffim manifest."""
    m = pd.read_csv(manifest)
    panels = []
    for _, r in m.iterrows():
        try:
            with fits.open(r.fits_path, memmap=True) as h:
                hdr = h[1].header; h0 = h[0].header
                w = _wcs_any(hdr); nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
                mjd = h0.get("MJD-AVG") or h0.get("MJD-OBS")
                if mjd is None:
                    da = h0.get("DATE-AVG"); mjd = Time(da, format="isot").mjd if da else np.nan
            c = w.all_pix2world([[nx / 2, ny / 2]], 0)[0]
            panels.append(dict(visit=int(r.visit), detector=int(r.detector), mjd=float(mjd),
                               wcs=w, nx=nx, ny=ny, cra=float(c[0]), cdec=float(c[1])))
        except Exception:
            continue
    return panels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sorcha", required=True, help="Sorcha main-output csv (ObjID, FieldID, RA_deg, Dec_deg, rates, trailedSourceMag, ...)")
    ap.add_argument("--manifest", required=True, help="diffim manifest (visit,detector,fits_path) for the window")
    ap.add_argument("--out-inject", required=True)
    ap.add_argument("--out-truth", required=True)
    ap.add_argument("--mjd-min", type=float, default=None, help="window start (TAI MJD); default = all")
    ap.add_argument("--mjd-max", type=float, default=None, help="window end (TAI MJD)")
    ap.add_argument("--m5", type=float, default=24.0, help="nominal single-visit 5sigma POINT-source depth (mag) for SNR reporting")
    ap.add_argument("--psf-fwhm-px", type=float, default=3.77)
    ap.add_argument("--exptime", type=float, default=30.0)
    a = ap.parse_args()

    s = pd.read_csv(a.sorcha)
    # tolerate Sorcha column-name variants
    ren = {"fieldMJD_TAI": "mjd", "RA_deg": "ra", "Dec_deg": "dec",
           "RARateCosDec_deg_day": "vra_cosd", "DecRate_deg_day": "vdec",
           "trailedSourceMag": "mag", "FieldID": "visit"}
    s = s.rename(columns={k: v for k, v in ren.items() if k in s.columns})
    need = {"ObjID", "visit", "mjd", "ra", "dec", "vra_cosd", "vdec", "mag"}
    miss = need - set(s.columns)
    if miss:
        raise SystemExit(f"[ephem2inj] Sorcha output missing columns {miss}; have {list(s.columns)[:20]}")
    if a.mjd_min is not None:
        s = s[(s.mjd >= a.mjd_min) & (s.mjd <= a.mjd_max)]
    s = s.reset_index(drop=True)
    print(f"[ephem2inj] {len(s)} in-window sightings, {s.ObjID.nunique()} distinct objects", flush=True)

    panels = read_panels(a.manifest)
    visits = sorted({p["visit"] for p in panels})
    by_visit = {v: [p for p in panels if p["visit"] == v] for v in visits}
    print(f"[ephem2inj] {len(panels)} panels over {len(visits)} manifest visits", flush=True)

    def panel_for(v, ra, dec):
        if v not in by_visit:
            return None, None, None
        cd = np.cos(np.radians(dec))
        for p in by_visit[v]:
            if np.hypot((p["cra"] - ra) * cd, p["cdec"] - dec) > 0.25:
                continue
            try:
                x, y = p["wcs"].all_world2pix([[ra, dec]], 0)[0]
            except Exception:
                continue
            if 0 <= x < p["nx"] and 0 <= y < p["ny"]:
                return p, float(x), float(y)
        return None, None, None

    dt_sub = a.exptime / 86400.0
    inj_rows = []
    for _, r in s.iterrows():
        v = int(r.visit)
        p, x, y = panel_for(v, float(r.ra), float(r.dec))
        if p is None:
            continue
        rate = float(np.hypot(r.vra_cosd, r.vdec))                       # deg/day on-sky
        trail_px = rate * dt_sub / (PIXSCALE / 3600.0)                   # deg/exposure -> px
        # beta: pixel-frame motion direction (step along the on-sky velocity over dt_sub)
        cd = np.cos(np.radians(float(r.dec)))
        ra_b = float(r.ra) + (r.vra_cosd / cd) * dt_sub
        dec_b = float(r.dec) + r.vdec * dt_sub
        try:
            xb, yb = p["wcs"].all_world2pix([[ra_b, dec_b]], 0)[0]
            beta = float(np.degrees(np.arctan2(yb - y, xb - x)) % 360.0)
        except Exception:
            beta = 0.0
        mag = float(r.mag)
        n_psf = max(1.0, trail_px / a.psf_fwhm_px)
        snr_t = 5.0 * 10.0 ** ((a.m5 - mag) / 2.5) / np.sqrt(n_psf)      # detection-SNR in our diffims
        inj_rows.append(dict(objID=str(r.ObjID), visit=v, detector=p["detector"], mjd=float(r.mjd),
                             ra=float(r.ra), dec=float(r.dec), x=x, y=y, trail_length=trail_px,
                             beta=beta, mag=mag, snr_target=float(snr_t), rate_degday=rate))
    inj = pd.DataFrame(inj_rows)
    if inj.empty:
        raise SystemExit("[ephem2inj] no sightings landed on a manifest panel -- check field/window match")
    # per-object truth: sighting count, NIGHT count (multi-night arc length), rate, faintest mag
    inj["night"] = np.floor(inj.mjd - 0.5).astype(int)
    g = inj.groupby("objID")
    truth = pd.DataFrame(dict(
        n_sightings=g.size(),
        n_nights=g.night.nunique(),
        rate_degday=g.rate_degday.median(),
        mag=g.mag.median(),
        snr_target=g.snr_target.median(),
        snr_min=g.snr_target.min(),
        trail_px=g.trail_length.median(),
    )).reset_index()
    Path(a.out_inject).parent.mkdir(parents=True, exist_ok=True)
    inj.drop(columns=["night"]).to_csv(a.out_inject, index=False)
    truth.to_csv(a.out_truth, index=False)
    ns, nn, st = truth.n_sightings, truth.n_nights, truth.snr_target
    print(f"[ephem2inj] {len(truth)} objects -> {len(inj)} sightings on panels", flush=True)
    print(f"[ephem2inj] NIGHTS/object: >=2 {int((nn>=2).sum())}, >=3 {int((nn>=3).sum())} (heliolinc needs >=3), "
          f"max {int(nn.max())}", flush=True)
    print(f"[ephem2inj] sightings/object: >=2 {int((ns>=2).sum())}, >=6 {int((ns>=6).sum())} (>=3 tracklets) | "
          f"target-SNR <5 {int((st<5).sum())} faint, 5-10 {int(((st>=5)&(st<10)).sum())}, >=10 {int((st>=10).sum())}", flush=True)
    print(f"[ephem2inj] mag [{truth.mag.min():.1f},{truth.mag.max():.1f}] | rate [{truth.rate_degday.min():.2f},{truth.rate_degday.max():.2f}] deg/day", flush=True)


if __name__ == "__main__":
    main()
