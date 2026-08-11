"""Emit same-night NEO candidates as a JSONL alert stream for follow-up.

Each 2-visit `NEW` track from `trail_state_link` is an *actionable short-arc alert*: two trailed
detections over a single night over-determine the on-sky motion (Method of Herget), enough to point a
follow-up the same night / next night even though they cannot self-confirm a discovery (no survey
confirms a NEO from 2 same-night points — see README / SAME_NIGHT_2v_3sigma.md). This module turns the
member detections of a track into a self-describing alert record and appends ONE json object per line:

  endpoints (per-epoch ra/dec/mjd/snr/...), the motion vector (rate, position angle, dRA/dDec rates),
  a forward-predicted ephemeris (linear extrapolation with a 2-point lever-arm uncertainty), and the
  confidence (geometry-gate chi2 + the admissible-region orbit RANGES / tier / priority). A 2-point
  same-night arc cannot determine an orbit -- the packet publishes the Milani admissible-region
  [lo,hi] element ranges with degenerate=true, never a scalar (a, e) point estimate (see
  orbit_check.fit_orbit for the measured degeneracy).

JSONL is append-able same-night and schema-stamped, so a follow-up coordinator or human vetter can
ingest it line-by-line. The producer (the linker) calls `build_alert` while it still holds the member
detection rows, then `write_alerts`. Pure / no I/O deps beyond json — unit-tested without GPU or data.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ALERT_SCHEMA_VERSION = "adcnn-samenight-2v/1.5"   # 1.5: trainVeto block (shared-great-circle LINE veto:
#     satellite-train glint chains AND static template-artifact lines both put >=10 trail-PA-aligned
#     collinear dets on the members' great circle; measured embargo 0629/0630 pathologies 11-15 vs
#     clean <=8 -- link_2visit.train_veto_check; FLAG-demoted, never dropped).
#     1.4: staticVeto block (template-footprint bright-static
#     seed-exclusion; single-static alerts FLAG-demoted, never dropped -- expt_staticveto/RESULTS.md).
#     1.3: veto-stack annotations (stationarity/fpp/pixelVet), sigma_rate-aware motion block +
#     neoRateGate, dt^2 short-gap priority bonus, demotion-aware ranking.
#     1.2: orbit block = admissible-region ranges (was argmin point).
SOLARDAY = 86400.0
# Per-epoch astrometric scatter floor (arcsec) for the RATE-uncertainty budget: faint trailed dets
# centroid to ~0.4" (the audit's short-arc term: 49-s pair => sigma_rate ~0.3 deg/day; a 39-min pair
# => ~0.006, negligible). Distinct from RMS_FLOOR_ARCSEC (0.1, the ephemeris-extrapolation floor):
# a 2-point track fits exactly (rms==0), so the rate error must carry its own realistic floor.
SIGMA_POS_2V_ARCSEC = 0.4
# Short-gap priority bonus reference (min): the chance-link annulus scales as dt^2 (measured lambda
# law), so a short-gap pair is intrinsically more trustworthy. The bonus saturates below DT_BONUS_REF.
DT_BONUS_REF_MIN = 10.0
DT_BONUS_MAX = 0.04   # keeps 2v NEW max = 2.0 + 0.95 + 0.04 = 2.99 < 3.0 (the 3+visit base): tier order is preserved
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


def _orbit_block(chi2, tier, adm):
    """Honest orbit reporting for a short-arc alert (schema 1.2, 2026-07-02). A same-night 2-point arc
    CANNOT determine an orbit: the fit residual is FLAT in topocentric distance (measured 1.2e-5 deg/day
    variation across the whole bound family vs ~0.5 deg/day trail-velocity noise), so the old scalar
    (a_au, ecc) argmin was a grid artifact -- it sat on the rho-grid floor and published Earth-clone
    a~1/e~0 for 50/74 alerts on 20260630, reading falsely as "NEO-like"; ground truth 1997 UT25
    (main-belt, a~2.3) came out a=1.01. We publish the Milani ADMISSIBLE REGION instead: [lo, hi]
    ranges of the bound, physically plausible family the gate accepted (orbit_check.orbit_ok adm_*).
    degenerate=true marks the 2-point family explicitly. chi2 stays: it is the GEOMETRY gate statistic
    (collinearity + trail-vs-motion PA/speed + bound-orbit residual), not an orbit-fit merit. 3+visit
    tracks carry null ranges (no 2-point fit; their confirmation is the (FP)^N triplet geometry). A
    real orbit determination needs a 3rd epoch -- which is what the alert requests."""
    adm = adm or {}
    n = adm.get("adm_n")
    try:
        has_family = n is not None and np.isfinite(float(n)) and float(n) > 0
    except (TypeError, ValueError):
        has_family = False
    if has_family:
        def rng(k):
            lo, hi = _f(adm.get(k + "_lo")), _f(adm.get(k + "_hi"))
            return [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)]
        return dict(chi2=_f(chi2), degenerate=True, rho_au=rng("adm_rho"),
                    a_au=rng("adm_a"), ecc=rng("adm_e"), q_au=rng("adm_q"))
    return dict(chi2=_f(chi2), degenerate=True if tier == "2visit" else None,
                rho_au=None, a_au=None, ecc=None, q_au=None)


def priority_score(status, tier, chi2, score_min, mfsnr_min, dt_min=None):
    """CONTINUOUS follow-up priority (higher = point a telescope sooner). RECALIBRATED 2026-06-10 on the
    v2 per-pair table (82 injection fields, exact FP, field-grouped CV; ALERT_SWEEP_DECISION.md addendum):
    within the gated stream the WEAKEST-MEMBER CNN score is the entire useful ranking signal -- it beat the
    old chi2-weighted formula (top-5 faint-fast truth 71 vs 40; med rank 7 vs 11) AND every logistic
    combination of [chi2, mfsnr, trail-PA/rate residuals] (each added term injects more variance than
    information; chance 2-point fits give FPs a fat LOW-chi2 tail, so chi2 is a GATE, not a ranking
    weight). chi2/mfsnr args are kept for API stability but intentionally NOT used in the variable term.
    Tier base (3+visit > 2-visit NEW > known recovery) + 0.95*score_min keeps tiers separated
    (2v NEW max 2.95 < 3.0 = 3+visit base).

    dt^2 SHORT-GAP BONUS (2026-07-02, INVESTIGATION_2V_CONFIDENCE.md section 5/6): the chance-link rate
    scales as dt^2, so a ~1-min mosaic-revisit pair has ~10^3x smaller chance area than a 39-min WFD
    pair. 2v NEW alerts get + DT_BONUS_MAX*min(1, (DT_BONUS_REF_MIN/dt)^2) -- bounded at 0.04 so the
    tier ordering is preserved BY CONSTRUCTION (max 2.99 < 3.0) and the WFD-cadence ranking that the
    2026-06-10 recalibration validated is essentially untouched (39-min pairs get +0.003). The real
    chance-term weight lives in the per-alert fpp block; this is the ordinal tiebreak."""
    base = 0.5 if status == "CONFIRMED" else (3.0 if tier == "3+visit" else 2.0)
    sc = float(score_min) if (score_min is not None and np.isfinite(score_min)) else 0.0
    bonus = 0.0
    if tier == "2visit" and status == "NEW" and dt_min is not None and np.isfinite(dt_min) and dt_min > 0:
        bonus = DT_BONUS_MAX * min(1.0, (DT_BONUS_REF_MIN / float(dt_min)) ** 2)
    return float(base + 0.95 * min(max(sc, 0.0), 1.0) + bonus)


def rate_sigma_degday(rms_arcsec, arc_days):
    """1-sigma apparent-rate uncertainty for a short linear arc: sqrt(2)*sigma_pos / dt, with the
    per-epoch positional scatter floored at SIGMA_POS_2V_ARCSEC (a 2-point fit has rms==0 by
    construction, so the measured rms alone would claim zero rate error). 49-s pair -> ~0.3 deg/day;
    39-min pair -> ~0.006."""
    s = max(float(rms_arcsec) if (rms_arcsec is not None and np.isfinite(rms_arcsec)) else 0.0,
            SIGMA_POS_2V_ARCSEC)
    dt = max(float(arc_days), 1.0 / SOLARDAY)
    return float(np.sqrt(2.0) * (s / 3600.0) / dt)


def build_alert(g, *, alert_id, night, obscode, status, tier, chi2, rms_arcsec, orbit_adm=None,
                match_obj="", match_frac=0.0, offsets_days=PREDICT_OFFSETS_DAYS, thumbnails=None,
                hiconf_score=0.80, stationarity=None, fpp=None, static_veto=None, train_veto=None,
                rate_lo=1.0):
    """Build one alert dict from a track's member detection rows `g` (a DataFrame slice) + its summary.

    `g` must carry per-epoch mjd, ra, dec (and optionally mag, mf_snr, len_db, score, visit, detector,
    art_frac). `orbit_adm`: the adm_* admissible-region summary dict from pair_chi2/orbit_ok (2-visit
    tracks; None/empty for 3+visit) -- becomes the packet's orbit block via _orbit_block. `thumbnails`:
    optional list of cutout paths/IDs (references, never embedded blobs).

    Veto-stack annotations (schema 1.3, all FLAG-not-drop): `stationarity` = the motion-aware catalog
    stationarity block from link_2visit.stationarity_check (vetoStationary demotes the alert in
    write_alerts, below every clean alert, but it is still published); `fpp` = the null-calibrated
    chance-link block from link_2visit.fpp_block; `static_veto` (schema 1.4) = the template-footprint
    bright-static annotation from link_2visit (nStaticMembers>=1 demotes like a stationarity flag --
    the member sits in a bright coadd static's residual wings; static-STATIC pairs never reach here,
    they are excluded at seeding); `train_veto` (schema 1.5) = the shared-great-circle LINE block from
    link_2visit.train_veto_check (vetoTrain demotes like the other flags -- the members sit on a line
    of >=minAligned trail-PA-aligned dets: a satellite-train glint chain or a static template-artifact
    line, the two measured line-FP classes); `rate_lo` = the op rate floor used ONLY for the
    neoRateGate annotation (rate - 3*sigma_rate > rate_lo -- is the NEO-rate claim secure against the
    short-arc rate error? the hard gate on the measured rate stays in physical_check, unchanged).
    A later pixel_vet stage adds the `pixelVet` block.
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
    # short-arc rate error + the sigma-aware NEO-rate annotation: is rate>=rate_lo secure against the
    # 2-point lever arm? (49-s pair: sigma~0.3 deg/day -- a 1.2 deg/day "NEO rate" is NOT 3-sigma secure.)
    # Annotation only: the hard gate on the measured rate lives in physical_check, unchanged.
    rate_sig = rate_sigma_degday(rms_arcsec, arc_days)
    neo_rate_gate = (bool(rate - 3.0 * rate_sig > float(rate_lo))
                     if (tier == "2visit" and rate_lo is not None) else None)
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
    # two-tier follow-up confidence: A = both members >= hiconf_score (the formal 0.80 purity-floor op);
    # B = a CANDIDATE alert (weakest member in [candidate_floor, hiconf)) -- lower per-alert purity, kept
    # because the stream is RANKED + CAPPED and a follow-up/3rd epoch confirms. Tier A always outranks B
    # via priorityScore (0.95*weakest-score), so this label just lets follow-up filter by capacity.
    confidence_tier = ("A" if (score_min is not None and score_min >= hiconf_score) else "B")
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
        confidenceTier=confidence_tier,                  # A = both members >=hiconf (0.80); B = candidate (0.60-0.80)
        priority=priority,
        priorityScore=round(priority_score(status, tier, chi2, score_min, mfsnr_min,
                                           dt_min=arc_days * 1440.0), 4),
        nEpochs=int(len(epochs)),
        arcMin=round(arc_days * 1440.0, 3),
        epochs=epochs,
        motion=dict(rate_degday=_f(rate), pa_deg=_f(pa),
                    dra_degday=_f(vra), ddec_degday=_f(vdec),
                    rate_sigma_degday=_f(rate_sig), neoRateGate=neo_rate_gate),
        predict=predict,
        orbit=_orbit_block(chi2, tier, orbit_adm),
        match=dict(obj=match_obj or None, frac=_f(match_frac)),
        vetting=vetting,
        stationarity=stationarity,                       # catalog veto block (link_2visit); None if untested
        fpp=fpp,                                         # null-calibrated chance-link block; None if no calib
        staticVeto=static_veto,                          # bright-static template-footprint block; None if no catalog
        trainVeto=train_veto,                            # shared-great-circle line block; None if untested
        thumbnails=thumbnails,
    )


def _rank_class(a):
    """Demotion class for ranking (0 = clean, 1 = veto-flagged, 2 = pixel-killed). FLAG-not-drop:
    a vetoed alert is still PUBLISHED (the audit measured a 3-5%/alert true-mover cost on the veto
    stack, so silent drops are not economical), it just sorts below every clean alert and cannot
    consume the --cap-alerts follow-up budget ahead of one. Class 1 = catalog vetoStationary OR
    staticVeto nStaticMembers>=1 (a member in a bright coadd static's residual wings; measured
    2026-07-02 -- static members dominate the false 2v alerts, expt_staticveto/RESULTS.md) OR
    trainVeto vetoTrain (the members sit on a shared-great-circle LINE of trail-PA-aligned dets --
    a satellite-train glint chain or a static template-artifact line; measured 2026-07-03 on embargo
    0629/0630: pathologies score 11-15 aligned line dets, everything clean <=8, golden NEO 1) OR
    pixelVet FLAGGED (3-5-sigma static evidence / defect-dominated capsule); class 2 = pixelVet
    killed (>=5-sigma mask-clean static, combined-or-single rule)."""
    pv = a.get("pixelVet") or {}
    if pv.get("killed"):
        return 2
    if ((a.get("stationarity") or {}).get("vetoStationary")
            or (a.get("staticVeto") or {}).get("nStaticMembers", 0) >= 1
            or (a.get("trainVeto") or {}).get("vetoTrain")
            or pv.get("verdict") == "FLAGGED"):
        return 1
    return 0


def write_alerts(alerts, path, *, append=False, top_n=None, rank_by="priority"):
    """Write/append alert dicts to a JSONL file (one compact json object per line), ranked by
    (demotion class, priorityScore desc) so the headline is line 1 and no veto-flagged alert
    outranks a clean one. `top_n` caps the emitted count (per-night follow-up budget) AFTER the
    ordering, so demoted alerts are cut first; the cut is logged, never silent.

    ``rank_by='chi2'`` orders instead by the 2-visit orbit-fit chi2 (best geometry first) within
    the same demotion classes. MEASURED on real night 20260630: in an 11,150-alert low-threshold
    stream the four validated production alerts present sit at ranks 3771/8687/9352/9942 under
    the default priority ordering (which is dominated by the weakest member's CNN score at that
    volume) but at 22/56/126/260 -- top 2.3% -- under chi2. Use it for the QA stream; the frozen
    science product keeps 'priority'."""
    if rank_by == "chi2":
        def _key(a):
            c = (a.get("orbit") or {}).get("chi2")
            # TIER FIRST. 3+visit tracks carry chi2=None (link_2visit sets NaN for n_ep != 2), which
            # maps to +inf and sorted them LAST inside the clean class -- on 0706 the night's only
            # 3+visit alert, the ~100%-purity discovery tier, landed at rank 4795 of 5790 instead of
            # rank 1. priority_score's own comment says the tier order holds "BY CONSTRUCTION";
            # rank_by="chi2" discarded that construction. `priority` is 1 for 3+visit, 2 for 2visit.
            return (_rank_class(a), a.get("priority", 9),
                    float(c) if c is not None else float("inf"),
                    -a.get("priorityScore", 0.0))
    else:
        def _key(a):
            return (_rank_class(a), -a.get("priorityScore", 0.0), a.get("priority", 9))
    ranked = sorted(alerts, key=_key)
    if top_n is not None and len(ranked) > top_n:
        print(f"[alerts] top-N cap: emitting {top_n} of {len(ranked)} alerts", flush=True)
        ranked = ranked[:top_n]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as fh:
        for a in ranked:
            fh.write(json.dumps(a, separators=(",", ":")) + "\n")
    return len(ranked)
