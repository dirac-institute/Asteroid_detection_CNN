"""Same-night trail-state linking: cluster ADCNN trail detections directly in (position, velocity)
phase space — NO heliocentric hypothesis grid, so NO candidate explosion.

Why this works where heliolinc didn't: heliolinc scans a grid of (r, rdot) hypotheses ONLY because
it must INFER an object's velocity from single-epoch angular positions (a point source gives position
but not motion). An ADCNN TRAIL measures the on-sky velocity DIRECTLY (the two trail endpoints over
the exposure time). So we skip the hypothesis grid entirely: propagate each detection to a common
reference time using ITS OWN trail velocity, and cluster in the 4-D space (RA@tref, Dec@tref, vRA,
vDec). Detections of one moving object collapse to the same 4-D point (same position and velocity at
tref); false positives — with random positions and random trail orientations — scatter and do not
cluster. This is O(N log N) (a KD-tree range query), tractable at any FP density, and uses ADCNN's
defining strength. A surviving cluster spanning >= npt distinct visits is a same-night track; matched
to a known SSObject it is a recovery, unmatched it is a NEW candidate (short-arc -> a candidate for
follow-up, not a determined orbit).

TWO confidence tiers (physical_check):
  - 3+visit: >=3 distinct epochs, full linear-motion residual test (high-confidence).
  - 2visit : exactly 2 epochs (the Heinze minimum — two trailed tracklets over-determine the short-arc
    motion). The linear-residual safety is degenerate for 2 points, so the 2-visit tier is held to a
    TIGHTER trail-PA tolerance (pa_tol_2v) AND an INDEPENDENT discriminator: the two trails' own
    intra-exposure velocity vectors must agree with each other (direction + magnitude), not merely
    with the connecting vector. Dropping the floor from 3 to 2 sightings is what lets us recover faint
    fast movers that ADCNN catches only twice in a night. Validate any 2visit NEW with the randomized-
    trail null test (validate_candidate.py) before calling it a candidate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
HL = REPO / "ADCNN/pipelines/heliolinc"
SOLARDAY = 86400.0


def trail_velocity(d, exptime_s):
    """On-sky angular velocity (deg/day) from the trail endpoints, in the local tangent plane.
    vx is the RA*cos(Dec) rate, vy the Dec rate. Trail spans the exposure (endpoints exptime_s apart)."""
    dt = exptime_s / SOLARDAY
    cosd = np.cos(np.radians(d.dec.to_numpy()))
    vx = (d.ra1.to_numpy() - d.ra0.to_numpy()) * cosd / dt
    vy = (d.dec1.to_numpy() - d.dec0.to_numpy()) / dt
    return vx, vy


# real-pair scatter (68th pct of |feature|), calibrated on injected NEO pairs in real off-ecliptic images;
# used to weight the 2-visit combined-orbit-fit chi^2. perp [arcsec], resid [frac of trail speed], dsnr
# [|dSNR|/min], dpa_tm [deg trail-vs-motion PA], dspeed [frac trail-vs-motion speed].
CHI2_SIG_2V = dict(perp=0.127, resid=0.133, dsnr=0.558, dpa_tm=4.869, dspeed=0.237)


def pair_chi2(g, exptime_s=30.0, sig=None):
    """Combined orbit-fit chi^2 for a 2-visit pair (g = 2-row member df). Returns (chi2, info-dict).
    Features (collinearity, bound-orbit rate-residual, brightness, trail-vs-motion PA & speed) are each
    divided by their real-pair scatter and summed in quadrature (Mahalanobis goodness-of-fit). Real pairs
    have chi2 ~ Ndof; a chance pair is large on >=1 axis. No training -- scatters are fixed from data."""
    sig = sig or CHI2_SIG_2V
    g = g.sort_values("mjd"); a, b = g.iloc[0], g.iloc[-1]
    dt = exptime_s / SOLARDAY

    def tv(r):
        cd = np.cos(np.radians(r.dec)); return (r.ra1 - r.ra0) * cd / dt, (r.dec1 - r.dec0) / dt
    cd = np.cos(np.radians(a.dec)); mdt = b.mjd - a.mjd
    mx = (b.ra - a.ra) * cd / mdt; my = (b.dec - a.dec) / mdt
    mpa = np.degrees(np.arctan2(my, mx)) % 180.0; mspeed = np.hypot(mx, my)
    tvs = [tv(a), tv(b)]
    pas = [np.degrees(np.arctan2(ty, tx)) % 180.0 for tx, ty in tvs]
    dpa_tm = max(abs(((pa - mpa + 90) % 180) - 90) for pa in pas)
    dspeed = max(abs(np.hypot(tx, ty) - mspeed) / max(mspeed, 0.3) for tx, ty in tvs)
    dpa_tt = abs(((pas[0] - pas[1] + 90) % 180) - 90)
    rej = dict(bound=False, a=np.nan, e=np.nan, perp=np.nan, resid=np.nan, dsnr=np.nan, dpa_tm=dpa_tm, dspeed=dspeed)
    # CHEAP pre-gate (trail-vs-motion PA & speed, trail-vs-trail PA): reject chance pairs on O(1) geometry
    # BEFORE the expensive bound-orbit solve (astropy ephemeris + Lambert). Cuts ~all chance chords on the
    # dense fields, so the chi2 path is fast (without this the orbit solve runs on every candidate -> O(hour)).
    if dpa_tm > 20.0 or dpa_tt > 15.0 or dspeed > 0.6:
        return np.inf, rej
    c0 = np.cos(np.radians(g.dec.mean()))
    P = np.array([[(ra - g.ra.mean()) * c0 * 3600.0, (dec - g.dec.mean()) * 3600.0]
                  for _, r in g.iterrows() for ra, dec in ((r.ra0, r.dec0), (r.ra1, r.dec1))])
    P = P - P.mean(0); perp = float(np.sqrt(np.mean((P @ np.linalg.svd(P)[2][1])**2)))
    from ADCNN.pipelines.heliolinc.orbit_check import orbit_ok
    _, of = orbit_ok(g, exptime_s=exptime_s, rate_frac_tol=1.0)
    sp = max(np.hypot(*tvs[0]), np.hypot(*tvs[1]), 0.1)
    resid = of["rate_resid"] / sp if of.get("bound") else 99.0
    s = g.mf_snr.to_numpy() if "mf_snr" in g.columns else np.array([1.0, 1.0])
    dsnr = abs(s[0] - s[-1]) / max(min(s), 1e-3)
    f = dict(perp=perp, resid=resid, dsnr=dsnr, dpa_tm=dpa_tm, dspeed=dspeed)
    chi2 = float(sum((f[k] / sig[k])**2 for k in sig))
    return chi2, dict(bound=bool(of.get("bound", False)), a=float(of.get("a", np.nan)),
                      e=float(of.get("e", np.nan)), **f)


def link(dets, *, exptime_s=30.0, tref=None, pos_tol_deg=0.017, vel_frac=0.30, vel_floor=0.3,
         npt=2, min_visits=2):
    """Cluster detections in (RA@tref, Dec@tref, vRA, vDec). Returns a label per detection (-1=noise)
    and the list of track member-index lists.

    pos_tol_deg: clustering radius in propagated position (deg). The dominant error is the trail-
    velocity ANGLE error (~8 deg for NEO-length trails) times the propagation arm |v|*(t-tref);
    0.017 deg (~60") accommodates a ~2.9 deg/day mover propagated ~0.5 h at ~8 deg angle error.
    vel: two velocities match within vel_frac of the larger magnitude (+ vel_floor deg/day floor).
    """
    n = len(dets)
    if n == 0:
        return np.full(0, -1), []
    mjd = dets.mjd.to_numpy()
    if tref is None:
        tref = 0.5 * (mjd.min() + mjd.max())
    cosd = np.cos(np.radians(dets.dec.to_numpy()))
    vx, vy = trail_velocity(dets, exptime_s)
    # propagate each detection back to tref using its own trail velocity
    x0 = dets.ra.to_numpy() * cosd - vx * (mjd - tref)
    y0 = dets.dec.to_numpy() - vy * (mjd - tref)
    visit = dets.visit.to_numpy()

    # scale velocity into position-like units so a single KD radius is meaningful: multiply velocity
    # by a characteristic arm (the night half-span) and weight by vel_frac so the radius is shared.
    arm = max(mjd.max() - mjd.min(), 1e-3) * 0.5
    vscale = arm / vel_frac
    pts = np.column_stack([x0, y0, vx * vscale, vy * vscale])
    rad = pos_tol_deg
    # also enforce velocity-fraction match separately (the vscale makes the KD radius ~ pos_tol when
    # the velocity difference is within vel_frac*|v|), plus the absolute vel_floor.
    tree = cKDTree(pts[:, :2])  # cluster on propagated position; refine with velocity below
    labels = np.full(n, -1, dtype=int)
    tracks = []
    visited = np.zeros(n, bool)
    order = np.argsort(-(vx**2 + vy**2))  # seed from fastest movers (cleanest trails)
    vmag = np.hypot(vx, vy)
    for i in order:
        if visited[i]:
            continue
        # neighbours within pos radius
        cand = np.array(tree.query_ball_point(pts[i, :2], rad))
        if len(cand) < npt:
            continue
        # velocity consistency with seed i
        dv = np.hypot(vx[cand] - vx[i], vy[cand] - vy[i])
        vtol = vel_frac * np.maximum(vmag[cand], vmag[i]) + vel_floor
        cand = cand[dv <= vtol]
        # keep at most one detection per visit (the closest in propagated position)
        if len(cand) < npt:
            continue
        dd = (x0[cand] - x0[i])**2 + (y0[cand] - y0[i])**2
        byv = {}
        for c, dist in sorted(zip(cand, dd), key=lambda t: t[1]):
            v = visit[c]
            if v not in byv:
                byv[v] = c
        members = list(byv.values())
        if len(members) < max(npt, min_visits):
            continue
        for m in members:
            visited[m] = True
            labels[m] = len(tracks)
        tracks.append(members)
    return labels, tracks


def chord_seed_pairs(dets, *, max_arc_min=40.0, rate_min=0.3, rate_max=10.0):
    """Seed 2-visit candidate pairs by the POSITION CHORD, not the noisy trail-velocity. For each pair of
    adjacent same-night visits (gap <= max_arc_min) enumerate detection pairs whose sky separation is
    consistent with a rate_min..rate_max deg/day mover, via a k-d tree. Returns [i,j] member-index lists
    (iloc into `dets`). physical_check (trail-vs-chord PA/speed + collinearity + bound orbit) then verifies.
    WHY: the trail-velocity tref-clustering in link() scatters ~80% of real pairs beyond the cluster radius
    (8 deg / 18% trail-velocity error x propagation arm) AND manufactures FP via that scatter; seeding on the
    PRECISE position chord recovers ~4x more real pairs at ~10x LOWER false rate (measured on real DP2 FP).
    Seed on what's precise (positions), verify with what's distinctive (trails)."""
    from scipy.spatial import cKDTree
    d = dets.reset_index(drop=True)
    mjd = d.mjd.to_numpy(); ra = d.ra.to_numpy(); dec = d.dec.to_numpy(); vis = d.visit.to_numpy()
    uv = sorted(set(vis.tolist()))
    vmjd = {v: float(np.median(mjd[vis == v])) for v in uv}
    idx_by = {v: np.where(vis == v)[0] for v in uv}
    pairs = []
    # ALL same-night visit pairs within max_arc_min (NOT just adjacent visits). The old adjacent-only
    # zip(uv[:-1],uv[1:]) missed real movers detected in NON-adjacent visits -- e.g. a deep-drilling night
    # with many same-night revisits where a faint mover is detected in visits 3 & 7 (4,5,6 below threshold).
    # No-op for the WFD 2-visit-per-night case (one pair); recovers the multi-visit movers otherwise.
    for ai in range(len(uv)):
        for bi in range(ai + 1, len(uv)):
            a_, b_ = uv[ai], uv[bi]
            dt = vmjd[b_] - vmjd[a_]
            if dt <= 0 or dt * 1440.0 > max_arc_min:
                continue
            ia, ib = idx_by[a_], idx_by[b_]
            if not len(ia) or not len(ib):
                continue
            cd = float(np.cos(np.radians(dec[ib].mean())))
            tree = cKDTree(np.column_stack([ra[ib] * cd, dec[ib]]))
            dmin, dmax = rate_min * dt, rate_max * dt             # deg
            for i in ia:
                for jp in tree.query_ball_point([ra[i] * cd, dec[i]], dmax):
                    j = int(ib[jp])
                    if np.hypot((ra[j] - ra[i]) * cd, dec[j] - dec[i]) >= dmin:
                        pairs.append([int(i), j])
    return pairs


def physical_check(dets, members, exptime_s=30.0, pa_tol_deg=20.0, speed_frac=0.5,
                   lin_rms_arcsec=1.0, min_epochs=2, epoch_gap_s=120.0, pa_tol_2v_deg=10.0,
                   orbit_check_2v=True, orbit_rate_tol=0.5, score_2v_min=0.0, max_arc_2v_min=None,
                   perp_collinear_2v_arcsec=None, snr_frac_2v=None, chi2_2v_max=None, chi2_sig=None,
                   mfsnr_min_2v=None, rate_lo_2v=None, rate_hi_2v=10.0):
    """Defensible physical consistency of a candidate track (rejects chance/trail-angle-coincidence
    false links that pass position clustering). Requires:
      1. >= min_epochs DISTINCT time epochs (merge sub-epoch_gap snaps — back-to-back snaps are one).
      2. LINEAR motion: residual of a straight (constant-velocity) fit over ALL member detections
         < lin_rms_arcsec. (A quadratic overfits 3-4 points and is NOT used here — linear is the
         real-mover test over a <1h arc.) SKIPPED ONLY when there are <=2 detection points (a line
         fits 2 points exactly, so the residual is identically zero). A 2-EPOCH track with >=3
         detections (e.g. a snap pair + one more visit) STILL gets the residual test over its points
         — this rejects spatially-scattered chance links that the trail checks alone would pass.
      3. The TRAIL of each detection points along the inter-epoch MOTION: |beta - motion_PA| < pa_tol
         (the trail IS the object's motion smeared over the exposure; mod 180).
      4. The trail SPEED (len_db/exptime) matches the inter-epoch angular speed within speed_frac.
      5. 2-VISIT ONLY (n_epochs==2): the linear-residual safety (2) is gone, so we (a) hold the trail-
         PA check to the TIGHTER pa_tol_2v, and (b) add an INDEPENDENT discriminator — the member
         trails' OWN velocity vectors (each measured within a single exposure) must agree with EACH
         OTHER in direction (< pa_tol_2v) and magnitude (< speed_frac). This does not derive from the
         inter-epoch connecting vector, so a chance pair of misaligned trails cannot fake it (the
         randomized-trail null test confirms ~0 survivors).
    Returns (ok, info)."""
    g = dets.iloc[members].sort_values("mjd").reset_index(drop=True)
    t = g.mjd.to_numpy()
    # 1. distinct epochs. PRIMARY definition = distinct VISIT id (each visit is ONE observation epoch,
    # cadence-independent). This fixes the failure on rapid same-night cadences (deep-drilling, ~1-min
    # revisits) where the old time-gap merge (epoch_gap_s) wrongly fused genuine separate visits into one
    # epoch and rejected real 2-visit links as "1 epoch". Fall back to the time-gap merge only when there
    # is no visit column (merge sub-epoch_gap_s back-to-back snaps).
    if "visit" in g.columns:
        n_ep = int(g.visit.nunique())
    else:
        ep = [0]
        for i in range(1, len(t)):
            if (t[i] - t[ep[-1]]) * SOLARDAY > epoch_gap_s:
                ep.append(i)
        n_ep = len(ep)
    if n_ep < min_epochs:
        return False, f"only {n_ep} distinct epochs", n_ep
    two_visit = (n_ep == 2)
    pa_tol = pa_tol_2v_deg if two_visit else pa_tol_deg
    # 2-visit Δt WINDOW: real same-night pairs are the SHORT scheduler pair gap (~20-40 min); chance FP
    # pairs predominantly span much longer arcs (linking non-adjacent visits). Capping the arc to the
    # pair gap is the single strongest 2v FP filter (purity 0.28->0.71) at ~no recall cost, and matches
    # the cadence (the WFD pair IS the tracklet). No-op for the 3+visit tier.
    if two_visit and max_arc_2v_min is not None:
        arc_min = (t.max() - t.min()) * SOLARDAY / 60.0
        if arc_min > max_arc_2v_min:
            return False, f"2v arc {arc_min:.0f}min>{max_arc_2v_min}", n_ep
    # 2-visit FP-density control: a 2-epoch link is only as clean as its members. The ADCNN stage-2
    # score is a trained real/bogus for TRAILED sources (a faint fast NEO still scores high — its trail
    # is distinctive); requiring BOTH members above score_2v_min thins the chance-pair pool ~density^2
    # WITHOUT touching the full-recall 3+visit tier. (Stack RBTransiNet reliability is NOT used: it
    # labels real fast NEOs bogus and the FP are mostly stack-missed — see memory two-visit-not-defensible.)
    if two_visit and score_2v_min > 0 and "score" in g.columns:
        if float(g.score.min()) < score_2v_min:
            return False, f"2v member score {g.score.min():.2f}<{score_2v_min}", n_ep
    # 2-visit PHOTOMETRIC purity cut (the strongest non-ML 2v lever, measured on real DP2 FP): a recovered
    # mover is bright in BOTH visits while a surviving chance-FP pair has >=1 faint marginal member. Require
    # the fainter member's matched-filter TRAIL SNR >= mfsnr_min_2v. This is the MATCHED-FILTER trail SNR
    # (integrates along the streak), NOT the per-PSF point SNR -> it does NOT revert to the 5sigma-stack
    # regime: the stack-missed fast/long-trail movers have high mf_snr and are KEPT. With mfsnr carrying the
    # purity, chi2_2v_max can be LOOSENED (~10) to recover noisy real movers (orbit chi2 doesn't separate
    # true/false among survivors). Lifts the 3sigma op-point from S0.95 (comp .044) to S0.80 (comp ~.09).
    if two_visit and mfsnr_min_2v is not None and "mf_snr" in g.columns:
        if float(g.mf_snr.min()) < mfsnr_min_2v:
            return False, f"2v mf_snr {g.mf_snr.min():.1f}<{mfsnr_min_2v}", n_ep
    if two_visit and rate_lo_2v is not None:
        cd = float(np.cos(np.radians(g.dec.mean()))); dt = t.max() - t.min()
        rate = np.hypot((g.ra.iloc[-1] - g.ra.iloc[0]) * cd, g.dec.iloc[-1] - g.dec.iloc[0]) / dt if dt > 0 else 0.0
        if rate < rate_lo_2v or rate > rate_hi_2v:
            return False, f"2v rate {rate:.1f} out of [{rate_lo_2v},{rate_hi_2v}]deg/day", n_ep
    # 2-visit COMBINED-χ² GATE (preferred over the independent AND-thresholds below): weight the orbit-fit
    # evidence (collinearity, rate-residual, brightness, trail-vs-motion PA & speed) by its real-pair scatter
    # and sum in quadrature — a Mahalanobis goodness-of-fit. Keeps a real pair that is excellent on most axes
    # but borderline on ONE (which AND-cuts reject) while rejecting an FP mediocre on ALL axes (which loose
    # AND-cuts admit): ~2.5x more completeness at the SAME false rate (measured on real DP2 FP), no ML.
    if two_visit and chi2_2v_max is not None:
        c2, ci = pair_chi2(g, exptime_s, chi2_sig)
        if not ci["bound"]:
            return False, f"2v no bound orbit", n_ep
        if c2 > chi2_2v_max:
            return False, f"2v chi2 {c2:.1f}>{chi2_2v_max}", n_ep
        return True, f"OK 2v chi2 {c2:.1f} a{ci['a']:.2f} e{ci['e']:.2f}", n_ep
    # 2. LINEAR fit residual over ALL member detections (degenerate only for <=2 points)
    tt = (g.mjd.to_numpy() - g.mjd.mean())
    cosd = np.cos(np.radians(g.dec.to_numpy()))
    x = g.ra.to_numpy() * cosd; y = g.dec.to_numpy()
    px = np.polyfit(tt, x, 1); py = np.polyfit(tt, y, 1)
    if len(g) <= 2:
        rms = 0.0
    else:
        rms = np.sqrt(np.mean((x - np.polyval(px, tt))**2 + (y - np.polyval(py, tt))**2)) * 3600
        if rms > lin_rms_arcsec:
            return False, f"non-linear (lin rms {rms:.2f}\")", n_ep
    # motion PA + speed from the linear fit
    mvx, mvy = px[0], py[0]                       # deg/day
    motion_pa = np.degrees(np.arctan2(mvy, mvx)) % 180.0
    motion_speed = np.hypot(mvx, mvy)
    # 3+4. each detection's trail PA + speed vs the motion
    dt = exptime_s / SOLARDAY
    tvecs = []
    for _, r in g.iterrows():
        tvx = (r.ra1 - r.ra0) * np.cos(np.radians(r.dec)) / dt
        tvy = (r.dec1 - r.dec0) / dt
        tvecs.append((tvx, tvy))
        tpa = np.degrees(np.arctan2(tvy, tvx)) % 180.0
        dpa = abs(((tpa - motion_pa + 90) % 180) - 90)
        if dpa > pa_tol:
            return False, f"trail PA {tpa:.0f} vs motion {motion_pa:.0f} (d={dpa:.0f})", n_ep
        tspeed = np.hypot(tvx, tvy)
        if abs(tspeed - motion_speed) > speed_frac * max(motion_speed, 0.3):
            return False, f"trail speed {tspeed:.2f} vs motion {motion_speed:.2f}", n_ep
    # 5. 2-visit independent discriminator: the two trails' own velocities must agree with each other
    if two_visit:
        (ax, ay), (bx, by) = tvecs[0], tvecs[-1]
        apa = np.degrees(np.arctan2(ay, ax)) % 180.0
        bpa = np.degrees(np.arctan2(by, bx)) % 180.0
        dpa = abs(((apa - bpa + 90) % 180) - 90)
        if dpa > pa_tol_2v_deg:
            return False, f"trails disagree PA (d={dpa:.0f})", n_ep
        sa, sb = np.hypot(ax, ay), np.hypot(bx, by)
        if abs(sa - sb) > speed_frac * max(sa, sb, 0.3):
            return False, f"trails disagree speed ({sa:.2f} vs {sb:.2f})", n_ep
        # 5b. 4-ENDPOINT COLLINEARITY (position-coincidence, the rho-scaling discriminator the angle/
        #     speed checks miss): a real mover's two trail segments lie on ONE trajectory line, so all
        #     four trail endpoints are collinear to sub-arcsec; two chance FP trails are merely parallel
        #     (random perpendicular offset). RMS perpendicular distance of the 4 endpoints from their
        #     best-fit line. (Real p90 ~0.19"; FP p10 ~0.19" -> ~10x FP cut at ~90% recall.)
        if perp_collinear_2v_arcsec is not None:
            c0 = np.cos(np.radians(g.dec.mean()))
            P = np.array([[(ra - g.ra.mean()) * c0 * 3600.0, (dec - g.dec.mean()) * 3600.0]
                          for _, r in g.iterrows() for ra, dec in ((r.ra0, r.dec0), (r.ra1, r.dec1))])
            P = P - P.mean(0)
            perp = float(np.sqrt(np.mean((P @ np.linalg.svd(P)[2][1])**2)))
            if perp > perp_collinear_2v_arcsec:
                return False, f"non-collinear trails (perp {perp:.2f}\")", n_ep
        # 5c. BRIGHTNESS consistency (independent of geometry): a real object has ~constant flux over the
        #     ~20-min pair; FP members have uncorrelated SNR. |dSNR|/min(SNR).
        if snr_frac_2v is not None and "mf_snr" in g.columns:
            s1, s2 = float(g.mf_snr.iloc[0]), float(g.mf_snr.iloc[-1])
            if abs(s1 - s2) > snr_frac_2v * max(min(s1, s2), 1e-3):
                return False, f"SNR inconsistent ({s1:.1f} vs {s2:.1f})", n_ep
    # 6. 2-visit BOUND-ORBIT test (the non-circular discriminator): the two trailed tracklets must be
    #    reproduced by ONE bound, physically-plausible heliocentric orbit (Method of Herget + Lambert,
    #    using the trail velocities as the constraint). A chance FP pair admits no such orbit.
    if two_visit and orbit_check_2v:
        from ADCNN.pipelines.heliolinc.orbit_check import orbit_ok
        ok_orb, of = orbit_ok(g, exptime_s=exptime_s, rate_frac_tol=orbit_rate_tol)
        if not ok_orb:
            return False, f"no bound orbit (a={of['a']:.2f} e={of['e']:.2f} resid={of['rate_resid']:.2f})", n_ep
    tag = (f"2v a{of['a']:.2f} e{of['e']:.2f}" if (two_visit and orbit_check_2v)
           else "2v" if two_visit else f"linrms {rms:.2f}\"")
    return True, f"OK {tag} PA {motion_pa:.0f} {motion_speed:.2f}deg/day {n_ep}ep", n_ep


def fit_residual(dets, members, exptime_s=30.0):
    """Fit a constant-velocity (+ small accel) track to the member detections; return astrometric
    RMS (arcsec) and the fitted deg/day speed. Short-arc: linear+quadratic in each coord vs time."""
    g = dets.iloc[members]
    t = g.mjd.to_numpy(); t = t - t.mean()
    cosd = np.cos(np.radians(g.dec.to_numpy()))
    x = g.ra.to_numpy() * cosd; y = g.dec.to_numpy()
    deg = 2 if len(g) >= 4 else 1
    rx = x - np.polyval(np.polyfit(t, x, deg), t)
    ry = y - np.polyval(np.polyfit(t, y, deg), t)
    rms = np.sqrt(np.mean(rx**2 + ry**2)) * 3600.0
    # speed from linear term
    sx = np.polyfit(t, x, 1)[0]; sy = np.polyfit(t, y, 1)[0]
    return rms, float(np.hypot(sx, sy))


def crossmatch(dets, members, known, tol_arcsec, tol_day):
    """Best-matching known ObjID for a track's member detections (or '' if none)."""
    g = dets.iloc[members]
    kmjd = known.mjd.to_numpy(); kra = known.ra.to_numpy(); kdec = known.dec.to_numpy()
    kobj = known.ObjID.astype(str).to_numpy()
    hits = []
    for _, r in g.iterrows():
        sel = np.abs(kmjd - r.mjd) <= tol_day
        if not sel.any():
            continue
        sep = np.hypot((kra[sel] - r.ra) * np.cos(np.radians(r.dec)), kdec[sel] - r.dec) * 3600
        j = np.argmin(sep)
        if sep[j] <= tol_arcsec:
            hits.append(kobj[sel][j])
    if not hits:
        return "", 0.0
    vc = pd.Series(hits).value_counts()
    return vc.index[0], vc.iloc[0] / len(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="ADCNN catalog: mjd,ra,dec,ra0,dec0,ra1,dec1,visit[,len_db,score,art_frac,snr]")
    ap.add_argument("--known", default=str(HL / "run_night8731/known.csv"))
    ap.add_argument("--out", default=str(HL / "run_night8731/tracks.csv"))
    ap.add_argument("--exptime", type=float, default=30.0)
    ap.add_argument("--len-db-min", type=float, default=6.0, help="hard trail-length floor (px); cut ALL shorter dets regardless of source")
    ap.add_argument("--art-frac-max", type=float, default=0.3, help="LSST mask cut")
    ap.add_argument("--score-min", type=float, default=0.0, help="ADCNN stage-2 CNN (trained real/bogus) score floor; raise to thin FP density for 2-visit linking")
    ap.add_argument("--npt", type=int, default=2, help="min detections (distinct visits) per track")
    ap.add_argument("--pos-tol", type=float, default=0.017, help="deg; propagated-position cluster radius")
    ap.add_argument("--vel-frac", type=float, default=0.30)
    ap.add_argument("--max-rms", type=float, default=1.0, help="arcsec; LINEAR motion fit RMS (physical_check, >=3 epochs)")
    ap.add_argument("--pa-tol", type=float, default=20.0, help="deg; trail PA vs motion PA agreement (>=3 epochs)")
    ap.add_argument("--pa-tol-2v", type=float, default=10.0, help="deg; TIGHTER trail-PA tol for 2-visit tier + trail-vs-trail agreement")
    ap.add_argument("--score-2v-min", type=float, default=0.0, help="min ADCNN score for BOTH members of a 2-visit link (purity/recall dial; set ~0.90 for a clean candidate stream; 3+visit tier unaffected)")
    ap.add_argument("--max-arc-2v-min", type=float, default=40.0, help="2-visit Δt window (min): only pair within the scheduler pair gap; the single strongest 2v FP cut (purity 0.28->0.71). None to disable")
    ap.add_argument("--orbit-rate-tol", type=float, default=0.25, help="2-visit bound-orbit velocity-residual tol (frac of trail speed); 0.25 is the purity/recall knee (0.5 was too loose). Tighter=purer")
    ap.add_argument("--perp-collinear-2v", type=float, default=0.30, help="2-visit 4-trail-endpoint COLLINEARITY RMS tol (arcsec): real movers' two trail segments lie on ONE line (~0.08), FP merely parallel; recall-safe ~3x FP cut. None to disable")
    ap.add_argument("--snr-frac-2v", type=float, default=None, help="2-visit brightness consistency |dSNR|/min; OFF by default (costs recall). ~0.6 for extra purity")
    ap.add_argument("--epoch-gap-s", type=float, default=40.0, help="seconds; detections closer than this in time are merged into ONE epoch (intra-visit snaps). MUST be < the same-night revisit gap or genuine separate visits merge and 2-visit links are rejected as '1 epoch' -- the shipped 120s WRONGLY merged the ~1-min deep-drilling cadence. 40s separates >=40s-apart visits, merges true back-to-back snaps; no-op for WFD's ~34-min pairs.")
    ap.add_argument("--chi2-2v-max", type=float, default=5.0, help="2-visit COMBINED orbit-fit chi^2 gate -- THE primary purity lever. The GEOMETRIC chi2 (collinearity + trail-vs-motion PA & speed) is the STRONG true/false discriminator (true median ~4 vs false ~39, FAST movers); TIGHTENING it 10->5 lifts the 3sigma completeness at the realistic 34-min WFD cadence from 0.070 to ~0.10 (+~45%%, validated on 80 off-ecliptic lambda fields) at NO purity cost. 0 to disable")
    ap.add_argument("--mfsnr-min-2v", type=float, default=10.0, help="2-visit photometric purity floor: fainter member's matched-filter TRAIL SNR >= this. CADENCE-DEPENDENT DIAL: at the realistic ~34-min WFD pair gap the residual chance-FP need this floor to reach 3sigma -> keep ~10. At RAPID cadence (deep-drilling/short dt, sparse FP) the geometric chi2 alone carries purity -> lower to ~5 to recover the fast FAINT movers (mf_snr ~ point_SNR*sqrt(PSF/trail_area) is low for long trails, so a high floor rejects exactly them). 0 to disable")
    ap.add_argument("--rate-lo-2v", type=float, default=1.0, help="2-visit NEO apparent-rate band low (deg/day)")
    ap.add_argument("--rate-hi-2v", type=float, default=8.0, help="2-visit NEO apparent-rate band high (deg/day)")
    ap.add_argument("--seed-2v", choices=["chord", "cluster"], default="chord", help="2-visit seeding: 'chord' (position-chord pairs + trail verify; ~4x recall, ~10x lower FP than 'cluster' trail-velocity clustering)")
    ap.add_argument("--pos-tol-3v", type=float, default=0.05, help="3+visit cluster radius (deg); 0.05 ~doubles 3v recall vs 0.017 at zero purity cost (physical_check is the gate)")
    ap.add_argument("--rate-min", type=float, default=0.3, help="chord seeder min apparent rate (deg/day)")
    ap.add_argument("--rate-max", type=float, default=10.0, help="chord seeder max apparent rate (deg/day)")
    ap.add_argument("--recur-max", type=int, default=None, help="TP-safe stationarity veto: drop detections recurring at the same sky position in >= this many other same-night visits (residuals recur; a >=1deg/day mover never does). 2 is strong. Needs many visits/night (dense fields); ~no-op for a WFD pair. None=off")
    ap.add_argument("--min-epochs", type=int, default=2, help="distinct time epochs (snaps merged); 2 enables 2-visit linking")
    ap.add_argument("--tol-arcsec", type=float, default=5.0)
    ap.add_argument("--tol-day", type=float, default=0.02)
    a = ap.parse_args()

    d = pd.read_csv(a.dets)
    n0 = len(d)
    if "art_frac" in d and a.art_frac_max > 0:
        d = d[d.art_frac.fillna(0.0) < a.art_frac_max]   # NaN (unmeasured mask) -> keep, never silently drop
    if "len_db" in d and a.len_db_min > 0:
        d = d[d.len_db >= a.len_db_min]
    if "score" in d and a.score_min > 0:
        d = d[d.score >= a.score_min]
    d = d.reset_index(drop=True)
    need = ["mjd", "ra", "dec", "ra0", "dec0", "ra1", "dec1", "visit"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"--dets missing {miss}")
    known = pd.read_csv(a.known)
    d["night"] = np.floor(d.mjd - 0.5).astype(int)
    print(f"[trail-link] {n0} dets -> {len(d)} after cuts | nights {sorted(d.night.unique())}", flush=True)

    rows = []
    for night, dn in d.groupby("night"):
        dn = dn.reset_index(drop=True)
        if a.recur_max is not None:
            from ADCNN.pipelines.heliolinc.recurrence import add_recurrence
            dn = add_recurrence(dn)
            dn = dn[dn.recur < a.recur_max].reset_index(drop=True)   # TP-safe (real movers have recur==0)
        if a.seed_2v == "chord":
            # 3+visit via (looser) trail-velocity clustering; 2-visit via precise position-chord seeding
            _, clus = link(dn, exptime_s=a.exptime, npt=3, pos_tol_deg=a.pos_tol_3v,
                           vel_frac=a.vel_frac, min_visits=3)
            cand = list(clus)
            if a.min_epochs <= 2:
                cand += chord_seed_pairs(dn, max_arc_min=(a.max_arc_2v_min or 1e9),
                                         rate_min=a.rate_min, rate_max=a.rate_max)
        else:
            _, cand = link(dn, exptime_s=a.exptime, npt=a.npt, pos_tol_deg=a.pos_tol,
                           vel_frac=a.vel_frac, min_visits=a.npt)
        cand.sort(key=len, reverse=True)   # 3+visit (longer) first; a triplet's dets aren't re-reported as pairs
        npass = 0; used = set()
        for members in cand:
            ok, info, n_ep = physical_check(dn, members, a.exptime, pa_tol_deg=a.pa_tol,
                                            lin_rms_arcsec=a.max_rms, min_epochs=a.min_epochs,
                                            epoch_gap_s=a.epoch_gap_s,
                                            pa_tol_2v_deg=a.pa_tol_2v, score_2v_min=a.score_2v_min,
                                            max_arc_2v_min=a.max_arc_2v_min, orbit_rate_tol=a.orbit_rate_tol,
                                            perp_collinear_2v_arcsec=a.perp_collinear_2v, snr_frac_2v=a.snr_frac_2v,
                                            chi2_2v_max=(a.chi2_2v_max if a.chi2_2v_max and a.chi2_2v_max > 0 else None),
                                            mfsnr_min_2v=(a.mfsnr_min_2v if a.mfsnr_min_2v and a.mfsnr_min_2v > 0 else None),
                                            rate_lo_2v=a.rate_lo_2v, rate_hi_2v=a.rate_hi_2v)
            if not ok:
                continue
            if n_ep == 2 and any(m in used for m in members):
                continue
            used.update(members)
            npass += 1
            rms, speed = fit_residual(dn, members, a.exptime)
            obj, frac = crossmatch(dn, members, known, a.tol_arcsec, a.tol_day)
            g = dn.iloc[members]
            # numeric orbit-fit columns for ranking the (candidate-grade) 2-visit stream
            if n_ep == 2:
                c2, ci = pair_chi2(g, a.exptime); chi2v, av, ev = c2, ci["a"], ci["e"]
            else:
                chi2v, av, ev = np.nan, np.nan, np.nan
            rows.append(dict(night=int(night), ndet=len(members), nvisit=g.visit.nunique(),
                             n_epochs=n_ep, tier=("2visit" if n_ep == 2 else "3+visit"),
                             arc_hr=(g.mjd.max() - g.mjd.min()) * 24, rms_arcsec=rms, speed_degday=speed,
                             chi2=chi2v, a_au=av, ecc=ev, ra=g.ra.mean(), dec=g.dec.mean(), check=info,
                             match_obj=obj, match_frac=frac, status="CONFIRMED" if obj else "NEW"))
        print(f"  night {night}: {len(dn)} dets -> {len(cand)} candidates, {npass} passed", flush=True)
    T = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(a.out, index=False)
    if len(T):
        conf = sorted(T[T.status == "CONFIRMED"].match_obj.unique())
        new = T[T.status == "NEW"]
        n2, n3 = int((T.tier == "2visit").sum()), int((T.tier == "3+visit").sum())
        print(f"\n[trail-link] {len(T)} tracks | tiers: {n3} 3+visit, {n2} 2visit | "
              f"{int((T.status=='CONFIRMED').sum())} CONFIRMED ({len(conf)} known objs) | {len(new)} NEW", flush=True)
        print(f"[trail-link] confirmed: {', '.join(conf[:30])}", flush=True)
        if len(new):
            # rank: 3+visit (confirmed-grade) first, then 2visit candidates by ascending orbit-fit chi2
            new = new.sort_values(['n_epochs', 'chi2'], ascending=[False, True])
            print(f"[trail-link] NEW candidates (best first; 3+visit are 3-sigma, 2visit are follow-up candidates):", flush=True)
            print(new[['ra','dec','tier','speed_degday','arc_hr','chi2','a_au','ecc','nvisit','match_frac']].head(20).to_string(index=False), flush=True)
    else:
        print("[trail-link] no tracks", flush=True)


if __name__ == "__main__":
    main()
