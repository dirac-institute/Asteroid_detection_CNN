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

ALERT_SCHEMA_VERSION = "adcnn-samenight-2v/1.0"
SOLARDAY = 86400.0
# forward-prediction look-aheads (days from the last epoch): +30 min, +1 h, +4 h, +1 night.
PREDICT_OFFSETS_DAYS = (30 / 1440.0, 60 / 1440.0, 240 / 1440.0, 1.0)
RMS_FLOOR_ARCSEC = 0.1   # per-epoch astrometric sigma floor for the extrapolation error budget


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


def _predict(ra_ref, dec_ref, vra, vdec, rate, arc_days, rms_arcsec, offsets_days):
    """Forward ephemeris by linear extrapolation from the reference (last) epoch.

    Position error grows with the lever arm: for a straight fit through the arc with per-epoch sigma s,
    the extrapolated 1-D error at lag dt past the last epoch is s·sqrt(1 + 2·(dt/arc)²) (2-point linear
    propagation). We report the 2-D radial error (×sqrt2) as a conservative pointing budget. Honest: a
    same-night arc is short, so a +1-night prediction carries a large (reported) uncertainty."""
    s = max(float(rms_arcsec) if np.isfinite(rms_arcsec) else RMS_FLOOR_ARCSEC, RMS_FLOOR_ARCSEC)
    arc = max(float(arc_days), 1.0 / SOLARDAY)
    cosd = np.cos(np.radians(dec_ref))
    out = []
    for dt in offsets_days:
        dec_p = dec_ref + vdec * dt
        ra_p = ra_ref + (vra * dt) / max(cosd, 1e-6)          # convert on-sky East-rate back to RA deg
        err = s * float(np.sqrt(1.0 + 2.0 * (dt / arc) ** 2)) * float(np.sqrt(2.0))   # arcsec, 2-D radial
        out.append(dict(dt_min=round(dt * 1440.0, 2), mjd=None,   # absolute MJD stamped by build_alert
                        ra=round((ra_p % 360.0), 6), dec=round(dec_p, 6), err_arcsec=round(err, 3)))
    return out


def _f(x):
    """JSON-safe float: NaN/inf -> None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def build_alert(g, *, alert_id, night, obscode, status, tier, chi2, a_au, ecc, rms_arcsec,
                match_obj="", match_frac=0.0, offsets_days=PREDICT_OFFSETS_DAYS):
    """Build one alert dict from a track's member detection rows `g` (a DataFrame slice) + its summary.

    `g` must carry per-epoch mjd, ra, dec (and optionally mag, mf_snr, len_db, score, visit, detector).
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
    predict = _predict(ra_ref, dec_ref, vra, vdec, rate, arc_days, rms_arcsec, offsets_days)
    for p in predict:                                   # stamp absolute MJDs onto the look-aheads
        p["mjd"] = round(mjd_ref + p["dt_min"] / 1440.0, 6)
    # priority: 3+visit NEW (3-sigma grade) first, then 2-visit NEW candidates, then known recoveries.
    if status == "CONFIRMED":
        priority = 3
    elif tier == "3+visit":
        priority = 1
    else:
        priority = 2
    return dict(
        schema=ALERT_SCHEMA_VERSION,
        alertId=alert_id,
        night=int(night),
        obscode=str(obscode),
        asOfMjd=round(mjd_ref, 6),                       # earliest this same-night alert could fire
        status=status,                                   # NEW (candidate) | CONFIRMED (known recovery)
        tier=tier,                                       # 2visit | 3+visit
        priority=priority,
        nEpochs=int(len(epochs)),
        arcMin=round(arc_days * 1440.0, 3),
        epochs=epochs,
        motion=dict(rate_degday=_f(rate), pa_deg=_f(pa),
                    dra_degday=_f(vra), ddec_degday=_f(vdec)),
        predict=predict,
        orbit=dict(chi2=_f(chi2), a_au=_f(a_au), ecc=_f(ecc)),
        match=dict(obj=match_obj or None, frac=_f(match_frac)),
    )


def write_alerts(alerts, path, *, append=False):
    """Write/append a list of alert dicts to a JSONL file (one compact json object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as fh:
        for a in alerts:
            fh.write(json.dumps(a, separators=(",", ":")) + "\n")
    return len(alerts)
