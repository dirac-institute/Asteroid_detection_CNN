"""Pixel-level stationarity vet for the same-night 2-visit alert stream (schema 1.3 `pixelVet`).

The 2v false-link wall is ~88% STRUCTURED: static subtraction artifacts that leave flux at the same
sky position in other same-night visits (INVESTIGATION_2V_CONFIDENCE.md sections 2/7/8, audited
2026-07-02). This stage re-opens the difference-image pixels at each alert endpoint and asks the one
question a catalog cannot: *is there static flux at this exact position in the OTHER same-night
visits, where a real mover has already left?*

FORMAL KILL RULE (the audit's central correction -- the pre-audit grid-search maxima false-flagged
24% of blind positions at 3 sigma):
  statistic  = `snr_at0`: forced trail-capsule photometry at the EXACT catalog position, NO grid
               search, summing only mask-clean pixels (BADBITS = BAD/SAT/INTRP/CR/EDGE/SUSPECT/
               NO_DATA/UNMASKEDNAN excluded -- the 000010 retraction: 8.3->3.3 sigma once the
               BAD/INTRP defect flux was dropped);
  validity   = the mover must have MOVED OUT of the capsule: expected displacement
               rate*dt > L/2 + halfw + margin (else the test would kill the mover itself);
  stacking   = visits combine as snr_comb = sum(flux)/sqrt(sum(var)) across all VALID covering
               same-night visits (the sub-5-sigma design point, doc section 8: single-visit kill
               power collapses to 7% for mf_snr<7 members; sqrt(N) stacking is how depth returns);
  kill       = (snr_comb >= kill_sigma) OR (any single valid visit >= kill_sigma) -- the OR keeps a
               legitimate 6-sigma single-visit static from being DILUTED by quiet visits
               (6/sqrt(3) = 3.5 would wrongly survive a combined-only rule);
  flag       = [flag_sigma, kill_sigma) or a defect-dominated capsule (badfrac > 0.5, which may
               only FLAG, never kill, and never enters the combined stack).

FLAG-not-drop: a vetoed alert is still published -- write_alerts demotes it below every clean alert
(the measured true-mover cost of the full veto stack is 3-5%/alert; 0/6 kills on real movers).
Verdicts: CLEAN | STATIC_E1 | STATIC_E2 | STATIC_BOTH | FLAGGED | CONFIRMED | NO_COVERAGE.
CONFIRMED is the opportunistic bonus: forced photometry at the PREDICTED position in a third visit
shows the mover arriving (>= conf_sigma, mask-clean, aperture widened to the prediction error --
still no grid search). `confident` = CLEAN/CONFIRMED + no catalog stationarity veto + fpp
perAlertShare <= --confident-fpp-max.

Runs AFTER link_2visit (needs its alerts.jsonl + the dets catalog's fits_path column for panel
lookup); reads pixels via ADCNN.inference.diffim_io (local or S3, in-memory). Gracefully no-ops
when the dets catalog has no fits_path (catalog-only runs). None of this touches the frozen
op-point: it annotates and re-ranks, it never drops.

  python -m ADCNN.linking.pixel_vet --alerts RUN/alerts.jsonl --dets RUN/adcnn_dets_masked.csv --in-place
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ADCNN.linking.rank_alerts import write_alerts

PXSCALE = 0.2                 # arcsec/px (LSSTCam)
BADBITS = ("BAD", "SAT", "INTRP", "CR", "EDGE", "SUSPECT", "NO_DATA", "UNMASKEDNAN")
MIN_GOOD_PX = 5               # a capsule with fewer mask-clean px carries no measurement
BADFRAC_MAX = 0.5             # defect-dominated capsule: may FLAG, never kill, never stacks
COVER_RADIUS_ARCMIN = 10.0    # panel lookup: dets of the test visit within this radius name the panels


def _unitvec(ra, dec):
    ra, dec = np.radians(np.atleast_1d(ra)), np.radians(np.atleast_1d(dec))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


class PanelStore:
    """LRU cache of difference-image panels (~200 MB each) + their BADBITS mask value.

    get(path) -> (img float32, var float32, mask int, wcs, badval) where badval is the bitwise-OR of
    this panel's BADBITS planes (mask-plane bit assignments vary per file -- read from MP_* header)."""

    def __init__(self, max_panels=5):
        self.max_panels = max_panels
        self._store = {}

    def get(self, path):
        if path not in self._store:
            from astropy.wcs import WCS
            from ADCNN.inference.diffim_io import open_diffim
            while len(self._store) >= self.max_panels:
                self._store.pop(next(iter(self._store)))
            with open_diffim(path) as h:
                mdef = {k[3:]: int(v) for k, v in h[2].header.items() if k.startswith("MP_")}
                badval = 0
                for name in BADBITS:
                    if name in mdef:
                        badval |= 1 << mdef[name]
                self._store[path] = (h[1].data.astype(np.float32), h[3].data.astype(np.float32),
                                     np.asarray(h[2].data), WCS(h[1].header), badval)
        return self._store[path]


def forced_at0(panel, ra, dec, rate_degday, pa_deg, *, exptime_s=30.0, halfw_px=2.0):
    """Forced trail-capsule photometry at the EXACT (ra, dec) -- the audited kill statistic.

    Capsule = the trail the alert's own motion predicts (length rate*exptime, orientation pa, width
    halfw), summing ONLY mask-clean finite pixels (BADBITS excluded). NO grid search: the null of
    this statistic is properly calibrated (audit stage B: at0 median ~0 sigma vs grid-max 2.5), so
    kill/flag thresholds mean what they say. Returns dict(flux, var, snr, badfrac, n_good, n_tot,
    valid) or None when the position is off-panel / the cutout degenerate. `valid` = enough clean
    px AND not defect-dominated (an invalid measurement may still FLAG, it can never kill/stack)."""
    img, var, mask, wcs, badval = panel
    try:
        x0, y0 = wcs.world_to_pixel_values(ra, dec)
    except Exception:
        return None
    x0, y0 = float(x0), float(y0)
    ny, nx = img.shape
    if not (np.isfinite(x0) and np.isfinite(y0) and 0 <= x0 < nx and 0 <= y0 < ny):
        return None
    # unit vector along the trail PA, in pixel coordinates (via a small world-step through the WCS)
    dd = 1e-4
    ra1 = ra + dd * np.sin(np.radians(pa_deg)) / max(np.cos(np.radians(dec)), 1e-9)
    dec1 = dec + dd * np.cos(np.radians(pa_deg))
    x1, y1 = wcs.world_to_pixel_values(ra1, dec1)
    ux, uy = float(x1) - x0, float(y1) - y0
    n = np.hypot(ux, uy)
    if not np.isfinite(n) or n <= 0:
        return None
    ux, uy = ux / n, uy / n
    L = max(float(rate_degday) * exptime_s / 86400.0 * 3600.0 / PXSCALE, 2.0)   # trail length, px
    r = int(np.ceil(L / 2 + halfw_px + 2))
    xlo, xhi = max(0, int(x0) - r), min(nx, int(x0) + r + 1)
    ylo, yhi = max(0, int(y0) - r), min(ny, int(y0) + r + 1)
    if xhi - xlo < 3 or yhi - ylo < 3:
        return None
    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    t = np.clip((xx - x0) * ux + (yy - y0) * uy, -L / 2, L / 2)
    dist = np.hypot(xx - x0 - t * ux, yy - y0 - t * uy)
    sel = dist <= halfw_px
    im, vv, mm = img[ylo:yhi, xlo:xhi][sel], var[ylo:yhi, xlo:xhi][sel], mask[ylo:yhi, xlo:xhi][sel]
    finite = np.isfinite(im) & np.isfinite(vv) & (vv > 0)
    if finite.sum() == 0:
        return None
    bad = (np.bitwise_and(mm, badval) != 0)
    good = finite & ~bad
    badfrac = float(bad[finite].mean())
    if good.sum() < MIN_GOOD_PX:
        return None
    F, V = float(im[good].sum()), float(vv[good].sum())
    return dict(flux=F, var=V, snr=round(float(F / np.sqrt(V)), 2), badfrac=round(badfrac, 2),
                n_good=int(good.sum()), n_tot=int(finite.sum()),
                valid=bool(badfrac <= BADFRAC_MAX))


class NightPixels:
    """Per-visit panel lookup + forced photometry over one night's dets catalog."""

    def __init__(self, dets, max_panels=5):
        self.store = PanelStore(max_panels)
        self.trees = {int(v): (cKDTree(_unitvec(g.ra.to_numpy(), g.dec.to_numpy())),
                               g.fits_path.to_numpy())
                      for v, g in dets.groupby("visit")}
        self.vmjd = dets.groupby("visit").mjd.median().to_dict()
        self.vnight = {int(v): int(np.floor(m - 0.5)) for v, m in self.vmjd.items()}

    def visits_for_night(self, night):
        return sorted(v for v, n in self.vnight.items() if n == int(night))

    def measure(self, visit, ra, dec, rate, pa, *, exptime_s, halfw_px):
        """forced_at0 on the first panel of `visit` that covers (ra, dec); None = no coverage."""
        if int(visit) not in self.trees:
            return None
        tree, paths = self.trees[int(visit)]
        idx = tree.query_ball_point(_unitvec(ra, dec)[0],
                                    r=2 * np.sin(np.radians(COVER_RADIUS_ARCMIN / 60) / 2))
        for p in dict.fromkeys(paths[i] for i in idx):
            r = forced_at0(self.store.get(p), ra, dec, rate, pa,
                           exptime_s=exptime_s, halfw_px=halfw_px)
            if r is not None:
                return r
        return None


def _stat_epoch(np_, epoch, rate, pa, night, *, exptime_s, halfw_px,
                margin_arcsec, kill_sigma, flag_sigma, max_stat_visits):
    """All-valid-visit STAT test for ONE alert endpoint. Returns the per-epoch pixelVet entry."""
    ra, dec, mjd0, v0 = epoch["ra"], epoch["dec"], epoch["mjd"], epoch["visit"]
    L_as = max(float(rate) * exptime_s / 86400.0 * 3600.0, 2.0 * PXSCALE)      # trail length, arcsec
    guard_as = L_as / 2 + halfw_px * PXSCALE + margin_arcsec                   # mover must be OUT
    tests = []
    cands = [v for v in np_.visits_for_night(night) if v != int(v0)]
    cands.sort(key=lambda v: abs(np_.vmjd[v] - mjd0))
    n_guard_skipped = 0
    for v in cands:
        dt_day = abs(np_.vmjd[v] - mjd0)
        disp_as = float(rate) * dt_day * 3600.0
        if disp_as <= guard_as:            # mover may still be inside the capsule: test INVALID
            n_guard_skipped += 1
            continue
        if len(tests) >= max_stat_visits:
            break
        m = np_.measure(v, ra, dec, rate, pa, exptime_s=exptime_s, halfw_px=halfw_px)
        if m is None:
            continue
        tests.append(dict(visit=int(v), dtMin=round(dt_day * 1440.0, 2),
                          expectedDispArcsec=round(disp_as, 1), snrAt0=m["snr"],
                          badfrac=m["badfrac"], nGoodPx=m["n_good"], valid=m["valid"],
                          _flux=m["flux"], _var=m["var"]))
    valid = [t for t in tests if t["valid"]]
    snr_comb = snr_max = None
    if valid:
        F, V = sum(t["_flux"] for t in valid), sum(t["_var"] for t in valid)
        snr_comb = round(float(F / np.sqrt(V)), 2)
        snr_max = max(t["snrAt0"] for t in valid)
    static = bool(valid and (snr_comb >= kill_sigma or snr_max >= kill_sigma))
    # flag zone: [flag, kill) on the clean stats, or a defect-dominated capsule at >= flag
    defect_flag = any((not t["valid"]) and t["snrAt0"] >= flag_sigma for t in tests)
    flag_zone = bool((not static) and ((valid and max(snr_comb, snr_max) >= flag_sigma)
                                       or defect_flag))
    for t in tests:
        t.pop("_flux"), t.pop("_var")
    return dict(visit=int(v0), nValid=len(valid), snrCombined=snr_comb, snrMaxSingle=snr_max,
                guardArcsec=round(guard_as, 1), nGuardSkipped=n_guard_skipped,
                static=static, flagZone=flag_zone, tests=tests)


def _conf_test(np_, alert, night, *, exptime_s, halfw_px, margin_arcsec, conf_sigma,
               max_conf_err_arcsec=1.0):
    """Opportunistic third-visit CONFIRMATION at the PREDICTED position (never demotes on a miss).

    Linear extrapolation from the last epoch; the capsule half-width is widened to the 2-point
    prediction error at that lag (still forced at ONE position -- no grid search, so a false confirm
    needs a real >=conf_sigma clean source at a blind position). Visits whose prediction error
    exceeds `max_conf_err_arcsec` are skipped (aperture would be noise); positions within the guard
    distance of either member endpoint are skipped (a static artifact there must not self-confirm)."""
    mo = alert["motion"]
    rate, pa, vra, vdec = mo["rate_degday"], mo["pa_deg"], mo["dra_degday"], mo["ddec_degday"]
    eps = sorted(alert["epochs"], key=lambda e: e["mjd"])
    ra_ref, dec_ref, mjd_ref = eps[-1]["ra"], eps[-1]["dec"], eps[-1]["mjd"]
    arc_day = max(eps[-1]["mjd"] - eps[0]["mjd"], 1.0 / 86400.0)
    member_visits = {int(e["visit"]) for e in eps}
    L_as = max(float(rate) * exptime_s / 86400.0 * 3600.0, 2.0 * PXSCALE)
    guard_as = L_as / 2 + halfw_px * PXSCALE + margin_arcsec
    cosd = max(np.cos(np.radians(dec_ref)), 1e-9)
    out = dict(tested=0, confirmed=False, visit=None, snr=None, errArcsec=None)
    for v in np_.visits_for_night(night):
        if int(v) in member_visits:
            continue
        dt = np_.vmjd[v] - mjd_ref
        # 2-point lever-arm prediction error (rank_alerts._predict linear term, 0.4" floor)
        err_as = 0.4 * float(np.sqrt(1.0 + 2.0 * (dt / arc_day) ** 2)) * float(np.sqrt(2.0))
        if err_as > max_conf_err_arcsec:
            continue
        ra_p = ra_ref + (vra * dt) / cosd
        dec_p = dec_ref + vdec * dt
        # a static at a member endpoint must not read as the mover "arriving"
        if any(np.hypot((ra_p - e["ra"]) * cosd, dec_p - e["dec"]) * 3600.0 <= guard_as
               for e in eps):
            continue
        hw = max(halfw_px, err_as / PXSCALE)
        m = np_.measure(v, ra_p, dec_p, rate, pa, exptime_s=exptime_s, halfw_px=hw)
        if m is None:
            continue
        out["tested"] += 1
        if m["valid"] and m["snr"] >= conf_sigma and (out["snr"] is None or m["snr"] > out["snr"]):
            out.update(confirmed=True, visit=int(v), snr=m["snr"], errArcsec=round(err_as, 2))
    return out


def vet_alert(np_, alert, *, exptime_s, halfw_px, margin_arcsec, kill_sigma, flag_sigma,
              conf_sigma, max_stat_visits, confident_fpp_max):
    """Annotate one 2-visit NEW alert in place: pixelVet block + top-level `confident`."""
    mo = alert.get("motion") or {}
    rate, pa = mo.get("rate_degday"), mo.get("pa_deg")
    if rate is None or pa is None or not np.isfinite(rate) or rate <= 0:
        return None
    night = int(alert["night"])
    eps = sorted(alert["epochs"], key=lambda e: e["mjd"])
    per_epoch = [_stat_epoch(np_, e, rate, pa, night, exptime_s=exptime_s,
                             halfw_px=halfw_px, margin_arcsec=margin_arcsec,
                             kill_sigma=kill_sigma, flag_sigma=flag_sigma,
                             max_stat_visits=max_stat_visits) for e in eps]
    conf = _conf_test(np_, alert, night, exptime_s=exptime_s, halfw_px=halfw_px,
                      margin_arcsec=margin_arcsec, conf_sigma=conf_sigma)
    statics = [pe["static"] for pe in per_epoch]
    n_valid = sum(pe["nValid"] for pe in per_epoch)
    if any(statics):
        verdict = ("STATIC_BOTH" if all(statics)
                   else ("STATIC_E1" if statics[0] else "STATIC_E2"))
        killed = True
    elif conf["confirmed"]:
        verdict, killed = "CONFIRMED", False
    elif any(pe["flagZone"] for pe in per_epoch):
        verdict, killed = "FLAGGED", False
    elif n_valid > 0:
        verdict, killed = "CLEAN", False
    else:
        verdict, killed = "NO_COVERAGE", False
    alert["pixelVet"] = dict(verdict=verdict, killed=killed, killSigma=kill_sigma,
                             flagSigma=flag_sigma, epochs=per_epoch, confirm=conf)
    share = (alert.get("fpp") or {}).get("perAlertShare")
    alert["confident"] = bool(
        verdict in ("CLEAN", "CONFIRMED")
        and not (alert.get("stationarity") or {}).get("vetoStationary")
        and (share is None or share <= confident_fpp_max))
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True, help="alerts.jsonl from link_2visit")
    ap.add_argument("--dets", required=True,
                    help="dets catalog with fits_path (adcnn_dets_masked.csv); no fits_path = no-op")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write the vetted+re-ranked stream here")
    g.add_argument("--in-place", action="store_true",
                   help="rewrite --alerts (original preserved once as alerts_prevet.jsonl)")
    ap.add_argument("--kill-sigma", type=float, default=5.0,
                    help="combined-or-single mask-clean snr_at0 KILL threshold (audited: 5)")
    ap.add_argument("--flag-sigma", type=float, default=3.0,
                    help="FLAG (demote, never drop) threshold (audited: 3)")
    ap.add_argument("--conf-sigma", type=float, default=5.0,
                    help="third-visit predicted-position confirmation threshold")
    ap.add_argument("--halfw-px", type=float, default=2.0, help="capsule half-width (px)")
    ap.add_argument("--margin-arcsec", type=float, default=3.0,
                    help="displacement-guard margin beyond L/2+halfw (audited: 3 arcsec)")
    ap.add_argument("--exptime-s", type=float, default=30.0, help="visit exposure (trail length)")
    ap.add_argument("--max-stat-visits", type=int, default=8,
                    help="cap on covering visits stacked per endpoint (closest-in-time first)")
    ap.add_argument("--confident-fpp-max", type=float, default=0.01,
                    help="max fpp.perAlertShare for the `confident` bit")
    ap.add_argument("--max-panels", type=int, default=48,
                    help="panel LRU cache size. Each resident panel is ~0.2 GB, so 48 is ~9 GB -- "
                         "trivial on a 500 GB node. PROFILED: one alert costs ~14 panel loads and "
                         "19.4 s, essentially all of it decompressing whole 4072x4000 tile-"
                         "compressed planes to read a ~20x20 px capsule. A bigger cache only helps "
                         "when consecutive alerts reuse panels, so pair it with --panel-order; the "
                         "real fix is reading just the capsule's tiles via HDU.section, which needs "
                         "the panel store to hold open handles rather than materialised arrays")
    a = ap.parse_args()

    out_path = a.alerts if a.in_place else a.out
    alerts = [json.loads(l) for l in open(a.alerts) if l.strip()]
    head = pd.read_csv(a.dets, nrows=0)
    if "fits_path" not in head.columns:
        print(f"[pixel_vet] SKIP: {a.dets} has no fits_path column (catalog-only run); "
              f"alerts pass through unvetted", flush=True)
        if not a.in_place and os.path.abspath(a.out) != os.path.abspath(a.alerts):
            shutil.copyfile(a.alerts, a.out)
        return
    dets = pd.read_csv(a.dets, usecols=["visit", "ra", "dec", "mjd", "fits_path"])
    np_ = NightPixels(dets, max_panels=a.max_panels)

    counts, n_vet = {}, 0
    for al in alerts:
        if al.get("tier") != "2visit" or al.get("status") != "NEW":
            continue
        v = vet_alert(np_, al, exptime_s=a.exptime_s, halfw_px=a.halfw_px,
                      margin_arcsec=a.margin_arcsec, kill_sigma=a.kill_sigma,
                      flag_sigma=a.flag_sigma, conf_sigma=a.conf_sigma,
                      max_stat_visits=a.max_stat_visits, confident_fpp_max=a.confident_fpp_max)
        if v is None:
            continue
        n_vet += 1
        counts[v] = counts.get(v, 0) + 1
        if v.startswith("STATIC"):
            pe = al["pixelVet"]["epochs"]
            sig = max(x for p in pe for x in (p["snrCombined"], p["snrMaxSingle"])
                      if x is not None)
            print(f"  KILL {al['alertId']}: {v} (max {sig:.1f} sigma, "
                  f"nValid {pe[0]['nValid']}/{pe[1]['nValid']})", flush=True)

    if a.in_place:
        bak = str(Path(a.alerts).with_name("alerts_prevet.jsonl"))
        if not os.path.exists(bak):        # keep the FIRST pre-vet copy, never clobber it
            shutil.copyfile(a.alerts, bak)
    n = write_alerts(alerts, out_path)     # demotion-aware re-rank (killed last, flagged next-to-last)
    n_conf = sum(1 for al in alerts if al.get("confident"))
    print(f"[pixel_vet] {n_vet} alerts vetted: "
          + ", ".join(f"{k} {counts[k]}" for k in sorted(counts))
          + f"; {n_conf} confident -> {out_path} ({n} lines)", flush=True)


if __name__ == "__main__":
    main()
