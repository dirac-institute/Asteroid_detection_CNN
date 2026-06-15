"""Emit same-night NEO candidates as a JSONL alert stream for follow-up.

Each 2-visit `NEW` track from `trail_state_link` is an *actionable short-arc alert*: two trailed
detections over a single night over-determine the on-sky motion (Method of Herget), enough to point a
follow-up the same night / next night even though they cannot self-confirm a discovery (no survey
confirms a NEO from 2 same-night points — see README / SAME_NIGHT_2v_3sigma.md). This module turns the
member detections of a track into a self-describing alert record and appends ONE json object per line:

  endpoints (per-epoch ra/dec/mjd/snr/...), the motion vector (rate, position angle, dRA/dDec rates),
  a forward-predicted ephemeris (linear extrapolation with a 2-point lever-arm uncertainty), and the
  confidence (orbit-fit chi2 / a / ecc / tier / priority).

JSONL is append-able same-night and schema-stamped, so a follow-up coordinator or human vetter can
ingest it line-by-line. The producer (the linker) calls `build_alert` while it still holds the member
detection rows, then `write_alerts`. Pure / no I/O deps beyond json — unit-tested without GPU or data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ALERT_SCHEMA_VERSION = "adcnn-samenight-2v/1.1"
SOLARDAY = 86400.0
# forward-prediction look-aheads (days from the last epoch): +30/60/90 min, +4 h, +1 night.
PREDICT_OFFSETS_DAYS = (30 / 1440.0, 60 / 1440.0, 90 / 1440.0, 240 / 1440.0, 1.0)
RMS_FLOOR_ARCSEC = 0.1   # per-epoch astrometric sigma floor for the extrapolation error budget
# Admissible-region curvature bound: a same-night arc fixes the on-sky VELOCITY but not the topocentric
# distance rho, and the apparent track BENDS by ~0.5*(a_obs/rho)*dt^2 (differential observer acceleration
# projected at the unknown range; a_obs ~ Earth's rotational centripetal 0.034 m/s^2 + heliocentric tidal
# margin). The search region must cover this over the plausible rho range -- the linear-extrapolation
# ellipse alone undersizes the box for close NEOs (degree-scale at +1 night for rho ~ 0.01 AU).
A_OBS_MS2 = 0.04                       # m/s^2, conservative differential-acceleration bound
RHO_GRID_AU = (0.01, 0.05, 0.3)        # close / mid / far -- the admissible-range grid reported per epoch
AU_M = 1.495978707e11


def _wrap180(d):
    """Wrap a degree difference into (-180, 180] — RA-wrap safe."""
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def _motion(g):
    """Least-squares on-sky velocity from a track's epochs (RA-wrap safe, cos-dec scaled).

    Returns (dra_degday, ddec_degday, rate_degday, pa_deg, ra_ref, dec_ref, mjd_ref) where the
    reference is the LAST epoch (predictions extrapolate forward from there). dRA is the great-circle
    RA rate (already × cos(dec)); pa is measured from North through East, like a trail PA."""
    s = g.sort_values("mjd")
    mjd = s.mjd.to_numpy(np.float64)
    ra = s.ra.to_numpy(np.float64)
    dec = s.dec.to_numpy(np.float64)
    cosd = np.cos(np.radians(dec.mean()))
    t = mjd - mjd[-1]                       # days relative to the last epoch
    # great-circle offsets relative to the last epoch
    xra = _wrap180(ra - ra[-1]) * cosd      # deg on-sky East
    xdec = dec - dec[-1]                    # deg North
    A = np.vstack([t, np.ones_like(t)]).T
    vra = float(np.linalg.lstsq(A, xra, rcond=None)[0][0])     # deg/day on-sky (East)
    vdec = float(np.linalg.lstsq(A, xdec, rcond=None)[0][0])   # deg/day (North)
    rate = float(np.hypot(vra, vdec))
    pa = float(np.degrees(np.arctan2(vra, vdec)) % 360.0)      # N->E
    return vra, vdec, rate, pa, float(ra[-1]), float(dec[-1]), float(mjd[-1])


def _curv_arcsec(dt_days, rho_au):
    """Curvature bound of the apparent track at lag dt for topocentric distance rho:
    0.5 * (A_OBS / (rho*AU)) * (dt_s)^2  [rad] -> arcsec. The admissible-region term the linear
    extrapolation misses (dominates for close objects / long lags)."""
    dt_s = float(dt_days) * SOLARDAY
    theta = 0.5 * (A_OBS_MS2 / (float(rho_au) * AU_M)) * dt_s * dt_s
    return float(np.degrees(theta) * 3600.0)


def _predict(ra_ref, dec_ref, vra, vdec, rate, arc_days, rms_arcsec, offsets_days,
             pa_deg=None, rho_grid_au=RHO_GRID_AU):
    """Forward ephemeris by linear extrapolation from the reference (last) epoch + a SEARCH REGION.

    Linear term: for a straight fit through the arc with per-epoch sigma s, the extrapolated 1-D error
    at lag dt past the last epoch is s·sqrt(1 + 2·(dt/arc)²) (2-point lever-arm propagation); reported
    2-D radial (×sqrt2). Admissible-region term: the unknown topocentric distance lets the track bend by
    _curv_arcsec(dt, rho) — reported per rho in `rho_grid_au` (close/mid/far), and folded into
    `search_radius_arcsec` at the CLOSEST plausible rho (the conservative follow-up box, elongated
    along-track: pa_deg gives the orientation). Honest: a same-night arc is short — at +1 night the
    close-NEO search region is degree-scale, which is WHY same-night follow-up matters."""
    s = max(float(rms_arcsec) if np.isfinite(rms_arcsec) else RMS_FLOOR_ARCSEC, RMS_FLOOR_ARCSEC)
    arc = max(float(arc_days), 1.0 / SOLARDAY)
    cosd = np.cos(np.radians(dec_ref))
    out = []
    for dt in offsets_days:
        dec_p = dec_ref + vdec * dt
        ra_p = ra_ref + (vra * dt) / max(cosd, 1e-6)          # convert on-sky East-rate back to RA deg
        err = s * float(np.sqrt(1.0 + 2.0 * (dt / arc) ** 2)) * float(np.sqrt(2.0))   # arcsec, 2-D radial
        curv = {f"rho{r}au": round(_curv_arcsec(dt, r), 1) for r in rho_grid_au}
        search = float(np.hypot(err, _curv_arcsec(dt, min(rho_grid_au))))
        out.append(dict(dt_min=round(dt * 1440.0, 2), mjd=None,   # absolute MJD stamped by build_alert
                        ra=round((ra_p % 360.0), 6), dec=round(dec_p, 6), err_arcsec=round(err, 3),
                        curv_arcsec=curv, search_radius_arcsec=round(search, 1),
                        search_pa_deg=_f(pa_deg)))
    return out


def _f(x):
    """JSON-safe float: NaN/inf -> None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def priority_score(status, tier, chi2, score_min, mfsnr_min):
    """CONTINUOUS follow-up priority (higher = point a telescope sooner). RECALIBRATED 2026-06-10 on the
    v2 per-pair table (82 injection fields, exact FP, field-grouped CV; ALERT_SWEEP_DECISION.md addendum):
    within the gated stream the WEAKEST-MEMBER CNN score is the entire useful ranking signal -- it beat the
    old chi2-weighted formula (top-5 faint-fast truth 71 vs 40; med rank 7 vs 11) AND every logistic
    combination of [chi2, mfsnr, trail-PA/rate residuals] (each added term injects more variance than
    information; chance 2-point fits give FPs a fat LOW-chi2 tail, so chi2 is a GATE, not a ranking
    weight). chi2/mfsnr args are kept for API stability but intentionally NOT used in the variable term.
    Tier base (3+visit > 2-visit NEW > known recovery) + 0.95*score_min keeps tiers separated
    (2v NEW max 2.95 < 3.0 = 3+visit base)."""
    base = 0.5 if status == "CONFIRMED" else (3.0 if tier == "3+visit" else 2.0)
    sc = float(score_min) if (score_min is not None and np.isfinite(score_min)) else 0.0
    return float(base + 0.95 * min(max(sc, 0.0), 1.0))


def build_alert(g, *, alert_id, night, obscode, status, tier, chi2, a_au, ecc, rms_arcsec,
                match_obj="", match_frac=0.0, offsets_days=PREDICT_OFFSETS_DAYS, thumbnails=None):
    """Build one alert dict from a track's member detection rows `g` (a DataFrame slice) + its summary.

    `g` must carry per-epoch mjd, ra, dec (and optionally mag, mf_snr, len_db, score, visit, detector,
    art_frac). `thumbnails`: optional list of cutout paths/IDs (references, never embedded blobs).
    Returns a json-serializable dict (the alert packet)."""
    s = g.sort_values("mjd")
    vra, vdec, rate, pa, ra_ref, dec_ref, mjd_ref = _motion(s)
    arc_days = float(s.mjd.max() - s.mjd.min())
    epochs = []
    for _, r in s.iterrows():
        epochs.append(dict(
            visit=int(r["visit"]) if "visit" in s.columns and pd.notna(r.get("visit")) else None,
            detector=int(r["detector"]) if "detector" in s.columns and pd.notna(r.get("detector")) else None,
            mjd=_f(r.get("mjd")), ra=_f(r.get("ra")), dec=_f(r.get("dec")),
            mag=_f(r.get("mag")), snr=_f(r.get("mf_snr")),
            trail_len_px=_f(r.get("len_db")), score=_f(r.get("score"))))
    predict = _predict(ra_ref, dec_ref, vra, vdec, rate, arc_days, rms_arcsec, offsets_days, pa_deg=pa)
    for p in predict:                                   # stamp absolute MJDs onto the look-aheads
        p["mjd"] = round(mjd_ref + p["dt_min"] / 1440.0, 6)
    # priority: 3+visit NEW (3-sigma grade) first, then 2-visit NEW candidates, then known recoveries.
    if status == "CONFIRMED":
        priority = 3
    elif tier == "3+visit":
        priority = 1
    else:
        priority = 2
    # vetting block: everything a human/robot vetter ranks and filters on, pulled from the members.
    score_min = float(s.score.min()) if "score" in s.columns else None
    mfsnr_min = float(s.mf_snr.min()) if "mf_snr" in s.columns else None
    vetting = dict(
        score_min=_f(score_min),
        score_max=_f(float(s.score.max())) if "score" in s.columns else None,
        mfsnr_min=_f(mfsnr_min),
        art_frac_max=_f(float(s.art_frac.max())) if "art_frac" in s.columns else None,
        trail_len_px=[_f(v) for v in s.len_db.tolist()] if "len_db" in s.columns else None,
    )
    return dict(
        schema=ALERT_SCHEMA_VERSION,
        alertId=alert_id,
        night=int(night),
        obscode=str(obscode),
        asOfMjd=round(mjd_ref, 6),                       # earliest this same-night alert could fire
        status=status,                                   # NEW (candidate) | CONFIRMED (known recovery)
        tier=tier,                                       # 2visit | 3+visit
        priority=priority,
        priorityScore=round(priority_score(status, tier, chi2, score_min, mfsnr_min), 4),
        nEpochs=int(len(epochs)),
        arcMin=round(arc_days * 1440.0, 3),
        epochs=epochs,
        motion=dict(rate_degday=_f(rate), pa_deg=_f(pa),
                    dra_degday=_f(vra), ddec_degday=_f(vdec)),
        predict=predict,
        orbit=dict(chi2=_f(chi2), a_au=_f(a_au), ecc=_f(ecc)),
        match=dict(obj=match_obj or None, frac=_f(match_frac)),
        vetting=vetting,
        thumbnails=thumbnails,
    )


def write_alerts(alerts, path, *, append=False, top_n=None):
    """Write/append alert dicts to a JSONL file (one compact json object per line), ranked by
    priorityScore descending (falls back to the integer priority) so the headline is line 1.
    `top_n` caps the emitted count (per-night follow-up budget); the cut is logged, never silent."""
    ranked = sorted(alerts, key=lambda a: (-a.get("priorityScore", 0.0), a.get("priority", 9)))
    if top_n is not None and len(ranked) > top_n:
        print(f"[alerts] top-N cap: emitting {top_n} of {len(ranked)} alerts", flush=True)
        ranked = ranked[:top_n]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as fh:
        for a in ranked:
            fh.write(json.dumps(a, separators=(",", ":")) + "\n")
    return len(ranked)
