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
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
HL = REPO / "ADCNN/pipelines/heliolinc"          # frozen calib/op-point INPUTS only
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")  # all runtime OUTPUT goes here
SOLARDAY = 86400.0


def radec_to_unit(ra_deg, dec_deg):
    """(ra,dec) deg -> unit 3-vector(s) on the sphere. Used for cKDTrees so seeding/crossmatch are correct
    ACROSS RA=0/360 and near the poles (a 2-D ra*cos(dec) tree breaks at the meridian). Scalar -> (3,);
    array -> (N,3). An angular radius theta(deg) maps to a chord-length query radius 2*sin(theta/2)."""
    ra = np.radians(np.asarray(ra_deg, dtype=float)); dec = np.radians(np.asarray(dec_deg, dtype=float))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def _chord_radius(theta_deg):
    """Euclidean (chord) radius on the unit sphere for an angular separation theta (deg)."""
    return 2.0 * np.sin(np.radians(theta_deg) / 2.0)


def trail_velocity(d, exptime_s):
    """On-sky angular velocity (deg/day) from the trail endpoints, in the local tangent plane.
    vx is the RA*cos(Dec) rate, vy the Dec rate. Trail spans the exposure (endpoints exptime_s apart)."""
    dt = exptime_s / SOLARDAY
    cosd = np.cos(np.radians(d.dec.to_numpy()))
    dra = (d.ra1.to_numpy() - d.ra0.to_numpy() + 180.0) % 360.0 - 180.0   # wrap so an RA=0 straddle can't flip sign
    vx = dra * cosd / dt
    vy = (d.dec1.to_numpy() - d.dec0.to_numpy()) / dt
    return vx, vy


# real-pair scatter (68th pct of |feature|), calibrated on injected NEO pairs in real off-ecliptic images;
# used to weight the 2-visit combined-orbit-fit chi^2. perp [arcsec], resid [frac of trail speed], dsnr
# [|dSNR|/min], dpa_tm [deg trail-vs-motion PA], dspeed [frac trail-vs-motion speed].
CHI2_SIG_2V = dict(perp=0.127, resid=0.133, dsnr=0.558, dpa_tm=4.869, dspeed=0.237)

# admissible-region summary keys (orbit_check.orbit_ok): [lo,hi] ranges of the bound, plausible 2-point
# orbit FAMILY the gate accepted. Reported in tracks.csv and the alert orbit block INSTEAD of the old
# argmin (a, e) point estimate, which is degenerate for a same-night 2-point arc (resid flat in rho;
# the argmin sat on the grid floor and reported Earth-clone a~1/e~0 -- see orbit_check.fit_orbit).
ADM_KEYS = ("adm_n", "adm_rho_lo", "adm_rho_hi", "adm_a_lo", "adm_a_hi",
            "adm_e_lo", "adm_e_hi", "adm_q_lo", "adm_q_hi")


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
                      e=float(of.get("e", np.nan)),
                      **{k: of.get(k, np.nan) for k in ADM_KEYS}, **f)


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


def auto_2v_window_min(dets, *, pointing_tol_deg=1.0, margin=1.15, lo=40.0, hi=75.0):
    """PER-NIGHT 2-visit Δt window, sized from the data so every visit can pair with its nearest
    SAME-POINTING revisit. A pure time heuristic fails on a WFD night -- consecutive visits in TIME
    are different fields (~30s apart) and the same-field revisit is ~35-45 min later -- so we group by
    POINTING: a visit pair counts only if their detection centroids lie within `pointing_tol_deg`.

    window = clamp( max_visit(nearest same-pointing gap) * margin, lo, hi ). The floor `lo` keeps the
    calibrated behaviour on dense/deep-drilling cadence (and preserves skip-visit linking); the ceiling
    `hi` stops a stray wide pair from exploding the seed count. Cross-night pairs (gap > 120 min) are
    excluded, so this is safe to compute over a multi-night `dets`. Falls back to `lo` if no
    same-pointing pair exists. Returns (window_min, info_str)."""
    if "visit" not in getattr(dets, "columns", []) or not len(dets):
        return lo, f"no visit column -> {lo:.0f}min"
    cen = {v: (float(np.median(g.ra)), float(np.median(g.dec)), float(np.median(g.mjd)))
           for v, g in dets.groupby("visit")}
    vs = list(cen)
    if len(vs) < 2:
        return lo, f"<2 visits -> {lo:.0f}min"
    ra = np.array([cen[v][0] for v in vs]); dec = np.array([cen[v][1] for v in vs])
    mjd = np.array([cen[v][2] for v in vs])
    U = radec_to_unit(ra, dec)                       # unit vectors; dot -> angular sep (RA-wrap/pole safe)
    nn = []                                           # per visit: gap to nearest SAME-POINTING revisit
    for i in range(len(vs)):
        sep = np.degrees(np.arccos(np.clip(U @ U[i], -1.0, 1.0)))
        gap = np.abs(mjd - mjd[i]) * 1440.0
        same = (sep < pointing_tol_deg) & (gap > 5.0) & (gap < 120.0)
        if same.any():
            nn.append(float(gap[same].min()))
    if not nn:
        return lo, f"no same-pointing pairs (tol {pointing_tol_deg:g}deg) -> {lo:.0f}min"
    raw = max(nn) * margin
    win = min(max(raw, lo), hi)
    return win, (f"{len(nn)} paired visit(s), widest nearest-revisit {max(nn):.1f}min "
                 f"x{margin:g} -> {win:.0f}min (clamp[{lo:.0f},{hi:.0f}])")


def chord_seed_pairs(dets, *, max_arc_min=40.0, rate_min=0.3, rate_max=10.0, max_visit_pairs=None):
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
    # SCALE GUARD: V visits -> O(V^2) pairs; a deep-drilling night (~100 visits) -> ~5000 pairs x KD queries.
    # Cap to the nearest-in-time pairs (the real same-night tracklets) and warn, so a pathological cadence
    # can't blow up the per-pair orbit-solve cost downstream.
    vpairs = []
    for ai in range(len(uv)):
        for bi in range(ai + 1, len(uv)):
            dt = vmjd[uv[bi]] - vmjd[uv[ai]]
            if 0 < dt <= max_arc_min / 1440.0:
                vpairs.append((dt, uv[ai], uv[bi]))
    if max_visit_pairs is not None and len(vpairs) > max_visit_pairs:
        vpairs.sort(key=lambda t: t[0])   # smallest time-gap first = the genuine same-night tracklets
        print(f"[chord-seed] dense cadence: capping {len(vpairs)} visit-pairs -> {max_visit_pairs} "
              f"(nearest-in-time)", flush=True)
        vpairs = vpairs[:max_visit_pairs]
    for dt, a_, b_ in vpairs:
        ia, ib = idx_by[a_], idx_by[b_]
        if not len(ia) or not len(ib):
            continue
        tree = cKDTree(radec_to_unit(ra[ib], dec[ib]))       # 3-D unit-sphere: correct across RA=0 + poles
        dmin, dmax = rate_min * dt, rate_max * dt             # deg (angular)
        qmax = _chord_radius(dmax)
        for i in ia:
            cd = float(np.cos(np.radians(dec[i])))
            for jp in tree.query_ball_point(radec_to_unit(ra[i], dec[i]), qmax):
                j = int(ib[jp])
                dra = (ra[j] - ra[i] + 180.0) % 360.0 - 180.0
                if np.hypot(dra * cd, dec[j] - dec[i]) >= dmin:   # rate floor (RA-wrap-safe)
                    pairs.append([int(i), j])
    return pairs


def prefilter_2v_pairs(dets, pairs, chi2_max, exptime_s=30.0):
    """EXACT vectorized pre-filter for 2-visit chord pairs: drop every pair whose PARTIAL chi2 (the four
    cheap pair_chi2 terms: perp-collinearity, dsnr, trail-vs-motion PA, trail-vs-chord speed) already
    exceeds chi2_max. The orbit-residual term is non-negative, so partial > chi2_max => chi2 > chi2_max
    => physical_check would reject -- removing the pair changes NOTHING (validated bit-identical to the
    python chain on 4/4 fields at S0.80 and 74/74 fields vs an independent run at S0.70; see
    exact_lowS_pairs). This turns the per-pair 135ms rho-scan orbit fit into a numpy pass for the ~99% of
    chance pairs -- the LSST-scale fast path for low score floors. Formulas/sigmas identical to pair_chi2.
    Apply ONLY to the 2v candidate list, never to the promotion/triplet-seeding input."""
    if chi2_max is None or not pairs:
        return pairs
    d = dets.reset_index(drop=True)
    P2 = np.asarray([p for p in pairs if len(p) == 2], int)
    if not len(P2):
        return pairs
    I, J = P2[:, 0], P2[:, 1]
    mjd = d.mjd.to_numpy(); ra = d.ra.to_numpy(); dec = d.dec.to_numpy()
    dt_exp = exptime_s / SOLARDAY
    cosd = np.cos(np.radians(dec))
    tvx = (d.ra1.to_numpy() - d.ra0.to_numpy()) * cosd / dt_exp
    tvy = (d.dec1.to_numpy() - d.dec0.to_numpy()) / dt_exp
    tpa = np.degrees(np.arctan2(tvy, tvx)) % 180.0
    tsp = np.hypot(tvx, tvy)
    mdt = mjd[J] - mjd[I]
    cdI = np.cos(np.radians(dec[I]))
    mx = (ra[J] - ra[I]) * cdI / mdt
    my = (dec[J] - dec[I]) / mdt
    mpa = np.degrees(np.arctan2(my, mx)) % 180.0
    msp = np.hypot(mx, my)
    dpa_tm = np.maximum(np.abs(((tpa[I] - mpa + 90) % 180) - 90),
                        np.abs(((tpa[J] - mpa + 90) % 180) - 90))
    dspeed = np.maximum(np.abs(tsp[I] - msp), np.abs(tsp[J] - msp)) / np.maximum(msp, 0.3)
    ra0 = d.ra0.to_numpy(); de0 = d.dec0.to_numpy(); ra1 = d.ra1.to_numpy(); de1 = d.dec1.to_numpy()
    mfs = d.mf_snr.to_numpy() if "mf_snr" in d.columns else np.ones(len(d))
    dm = (dec[I] + dec[J]) / 2.0; c0 = np.cos(np.radians(dm)); ram = (ra[I] + ra[J]) / 2.0
    Px = np.stack([(ra0[I] - ram) * c0, (ra1[I] - ram) * c0,
                   (ra0[J] - ram) * c0, (ra1[J] - ram) * c0], 1) * 3600.0
    Py = np.stack([de0[I] - dm, de1[I] - dm, de0[J] - dm, de1[J] - dm], 1) * 3600.0
    Px -= Px.mean(1, keepdims=True); Py -= Py.mean(1, keepdims=True)
    sxx = (Px * Px).mean(1); syy = (Py * Py).mean(1); sxy = (Px * Py).mean(1)
    perp = np.sqrt(np.maximum(0.5 * (sxx + syy) - np.sqrt(np.maximum(0.25 * (sxx - syy) ** 2 + sxy ** 2,
                                                                     0.0)), 0.0))
    smn = np.minimum(mfs[I], mfs[J])
    dsnr = (np.maximum(mfs[I], mfs[J]) - smn) / np.maximum(smn, 1e-3)
    partial = ((perp / CHI2_SIG_2V["perp"]) ** 2 + (dsnr / CHI2_SIG_2V["dsnr"]) ** 2 +
               (dpa_tm / CHI2_SIG_2V["dpa_tm"]) ** 2 + (dspeed / CHI2_SIG_2V["dspeed"]) ** 2)
    keep = partial <= float(chi2_max)
    kept = [[int(a_), int(b_)] for a_, b_ in P2[keep]]
    longer = [p for p in pairs if len(p) != 2]
    return kept + longer


def drop_static_static_pairs(pairs, static_flag):
    """SEED-EXCLUSION static veto (measured 2026-07-02, run_embargo_0630/expt_staticveto/RESULTS.md):
    drop every 2-visit seed pair whose BOTH members are template-footprint statics (the per-det
    `static_veto` flag: within the veto radius of a bright coadd object). A static-static pair is a
    repeating subtraction artifact linking to itself -- 95/107 of the floor-0.5 alerts on the dense
    0630 field -- so excluding it from SEEDING removes the structured FP background at ~zero true cost
    (a real mover pair is static-static only by double chance overlap, ~(1.3%)^2). Pairs with <=1
    static member are KEPT: the single-static alert is annotated + demoted downstream, never dropped.
    Only raw [i,j] seed pairs are ever passed here (before prefilter/promotion)."""
    if not len(pairs):
        return pairs
    P = np.asarray(pairs, dtype=int)
    keep = ~(static_flag[P[:, 0]] & static_flag[P[:, 1]])
    return P[keep].tolist()


def _jf(x, nd=2):
    """JSON-safe rounded float for alert annotation blocks: NaN/inf/None -> None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def extend_to_triplets(dets, pairs, *, pos_tol_arcsec=5.0):
    """Promote 2-visit chord pairs to 3+visit tracks WHEN a consistent 3rd same-night detection exists.

    For each pair, extrapolate the PRECISE 2-centroid linear track (not the noisy trail velocity) to every
    other same-night visit and attach the nearest detection within pos_tol of the predicted position. A
    3-point track is far purer than a pair: requiring a real 3rd detection on the 2-point line at the
    predicted time is the (FP)^N collapse, so this is ~free purity for the subset that has a recoverable 3rd.
    The merged member list is handed to physical_check (3v linear-RMS + PA gate), which rejects bad attaches;
    if the triplet fails, the original pair is still evaluated downstream (no recall lost). No-op for a
    2-visit (WFD) night. Distinct from forced photometry: the 3rd is a REAL CNN detection, not a pixel measure."""
    d = dets.reset_index(drop=True)
    mjd = d.mjd.to_numpy(); ra = d.ra.to_numpy(); dec = d.dec.to_numpy(); vis = d.visit.to_numpy()
    uv = sorted(set(vis.tolist()))
    if len(uv) < 3:
        return []
    idx_by = {v: np.where(vis == v)[0] for v in uv}
    vmjd = {v: float(np.median(mjd[vis == v])) for v in uv}
    trees = {v: cKDTree(radec_to_unit(ra[idx_by[v]], dec[idx_by[v]])) for v in uv}
    tol_chord = _chord_radius(pos_tol_arcsec / 3600.0)
    out = []
    for pr in pairs:
        i, j = pr
        if mjd[i] > mjd[j]:
            i, j = j, i
        ti, tj = mjd[i], mjd[j]; dt = tj - ti
        if dt <= 0:
            continue
        cdi = np.cos(np.radians(dec[i]))
        vx = (((ra[j] - ra[i] + 180.0) % 360.0 - 180.0)) * cdi / dt    # RA*cosDec rate (deg/day), wrap-safe
        vy = (dec[j] - dec[i]) / dt
        extra = []
        for v in uv:
            if v == vis[i] or v == vis[j]:
                continue
            tk = vmjd[v]
            pra = ra[i] + vx * (tk - ti) / cdi
            pdec = dec[i] + vy * (tk - ti)
            nb = trees[v].query_ball_point(radec_to_unit(pra, pdec), tol_chord)
            if not nb:
                continue
            # nearest detection in this visit to the predicted position
            uvi = radec_to_unit(pra, pdec)
            k = int(idx_by[v][min(nb, key=lambda p: float(np.sum((radec_to_unit(ra[int(idx_by[v][p])],
                                                                                dec[int(idx_by[v][p])]) - uvi) ** 2)))])
            extra.append(k)
        if extra:
            out.append([int(i), int(j)] + extra)
    return out


def physical_check(dets, members, exptime_s=30.0, pa_tol_deg=20.0, speed_frac=0.5,
                   lin_rms_arcsec=1.0, min_epochs=2, epoch_gap_s=120.0, pa_tol_2v_deg=10.0,
                   orbit_check_2v=True, orbit_rate_tol=0.5, score_2v_min=0.0, max_arc_2v_min=None,
                   perp_collinear_2v_arcsec=None, snr_frac_2v=None, chi2_2v_max=None, chi2_sig=None,
                   mfsnr_min_2v=None, rate_lo_2v=None, rate_hi_2v=10.0, out=None):
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
        if out is not None:      # expose the numeric gate statistic (the return is a display string)
            out["chi2"] = float(c2)
        if not ci["bound"]:
            return False, f"2v no bound orbit", n_ep
        if not np.isfinite(c2) or c2 > chi2_2v_max:   # NaN must REJECT (a>max comparison is False for NaN)
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


def stationarity_check(g, stat_trees, stat_mjd, *, tol_arcsec=3.0, min_disp_arcsec=10.0):
    """Motion-aware CATALOG stationarity test for a 2-visit track (AUDITED 2026-07-02,
    INVESTIGATION_2V_CONFIDENCE.md section 3b): a >=1 deg/day mover is >=27" from itself after a
    39-min pair gap, so a counterpart detection within `tol_arcsec` of a member's position IN THE
    OTHER MEMBER'S VISIT means that member is a repeating static artifact, not the mover. The
    counterpart catalog is ADCNN's OWN full detection list (pre-score-floor -- ADCNN-vs-ADCNN), so
    the test inherits the detector's sub-5sigma depth and keeps ~83%% kill power in the faintest
    (mf_snr<7) bin where 5sigma forced photometry collapses; LSST source detection appears nowhere.

    MOTION-AWARE VALIDITY GUARD: the test only applies when the expected displacement rate*dt
    exceeds `min_disp_arcsec` (>> tol) -- at a 46-s companion cadence a real mover moves only
    ~2-9" and would otherwise SELF-veto (its own other-epoch detection is the "counterpart").
    Measured false-kill cost on real movers (30"-offset KD test): ~1.0%%/member = ~2%%/alert.
    This is a FLAG, never a silent drop: vetoed alerts stay published, ranked after clean ones.

    g: the 2 member rows (sorted by mjd). stat_trees: {visit: (cKDTree over unit vectors, n_dets)}
    built from the FULL (pre-floor) night catalog. stat_mjd: {visit: median mjd}.
    Returns dict(vetoStationary, testable, e1/e2 sub-dicts with sep_arcsec / n_counterparts)."""
    g = g.sort_values("mjd")
    a, b = g.iloc[0], g.iloc[-1]
    cd = float(np.cos(np.radians(g.dec.mean())))
    dt = float(b.mjd - a.mjd)
    rate = (np.hypot((b.ra - a.ra) * cd, b.dec - a.dec) / dt) if dt > 0 else 0.0   # deg/day
    disp_as = rate * dt * 3600.0                                                    # = member separation
    out = {"testable": bool(disp_as >= min_disp_arcsec), "minDispArcsec": float(min_disp_arcsec),
           "tolArcsec": float(tol_arcsec), "expectedDispArcsec": round(disp_as, 2)}
    veto = False
    qr = _chord_radius(tol_arcsec / 3600.0)
    for tag, mem, other_visit in (("e1", a, int(b.visit)), ("e2", b, int(a.visit))):
        sub = {"counterpart": None, "sep_arcsec": None, "n_counterparts": 0}
        if out["testable"] and other_visit in stat_trees:
            tree, _n = stat_trees[other_visit]
            u = radec_to_unit(mem.ra, mem.dec)
            idx = tree.query_ball_point(u, qr)
            sub["n_counterparts"] = int(len(idx))
            sub["counterpart"] = bool(idx)
            if idx:
                veto = True
                chord = float(tree.query(u, k=1)[0])
                sub["sep_arcsec"] = round(np.degrees(2 * np.arcsin(min(chord / 2, 1.0))) * 3600, 2)
        out[tag] = sub
    out["vetoStationary"] = bool(veto and out["testable"])
    return out


def _sky_pa_mod180(u, v):
    """Position angle (deg, mod 180) of tangent direction(s) v at sky position(s) u (both (N,3);
    v need not be normalized). PA convention: 0 = north, 90 = east; mod 180 because trails and
    great-circle directions are unsigned."""
    east = np.cross(np.array([0.0, 0.0, 1.0]), u)
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(u, east)
    return np.degrees(np.arctan2(np.sum(v * east, -1), np.sum(v * north, -1))) % 180.0


def train_veto_check(g, train_arrays, *, perp_tol_arcsec=2.5, window_arcsec=1800.0,
                     align_tol_deg=20.0, align_len_min_px=5.0, min_aligned=10):
    """Shared-great-circle LINE veto for a 2-visit track (MEASURED 2026-07-03, embargo 0629+0630).

    Two structured FP classes put the pair's members on a LINE that is densely populated by OTHER
    detections whose own trail PAs match the line direction:
      1. SATELLITE TRAINS: train member B trails A by ~1 visit gap on the same orbital plane (= the
         same great circle on sky), so B's glints land near where A's glints were -- the linker pairs
         A-knots with B-knots into fake slow "movers" whose motion PA equals the trail-line PA
         (0629: alert pa 306.1 vs line PA 306.5; STREAK masking caught NONE of it -- the glint chain
         is sub-5sigma per pixel).
      2. STATIC template-artifact lines (e.g. coadd column defects): residual knots repeat at the
         SAME along-positions every visit (sub-arcsec); a pair of non-repeating knots on the line
         passes stationarity/staticVeto/pixelVet because those test only the member positions.
    One statistic catches both: the great circle through the two members; count OTHER dets (of the
    pair's two visits, from the FULL pre-floor catalog) within `perp_tol_arcsec` of the circle and
    within `window_arcsec` along-track of the member midpoint; 'aligned' additionally requires the
    det's own trail PA (ra0/dec0 -> ra1/dec1) within `align_tol_deg` of the local circle direction
    (only dets with length > `align_len_min_px` vote). Veto when the two visits' aligned counts sum
    to >= `min_aligned`. Threshold basis (floor-0.5 real alerts): pathologies scored {11,12,14,15,15}
    aligned; every believed-clean alert <=8; the golden NEO scored 1. An isolated real mover adds ~0
    -- background dets are isotropic, so P(on-circle AND PA-aligned) per det is ~1e-4.

    nRepeats (annotation only): cross-visit pairs of on-line dets at the SAME along-position
    (<1.5") -- static lines repeat, train glints drift, so it tells a vetter WHICH class fired.

    g: the 2 member rows. train_arrays: {visit: (u, u0, u1, length)} unit-vector arrays over the
    FULL pre-floor night catalog (u = det centers, u0/u1 = trail endpoints, length in px).
    Returns a json-able dict (tested, vetoTrain, nCollinear, nAligned, nRepeats, perVisit, config).
    FLAG, never drop: vetoTrain demotes the alert in ranking (class 1), it is still published."""
    r2as = np.degrees(1.0) * 3600.0
    g = g.sort_values("mjd")
    a, b = g.iloc[0], g.iloc[-1]
    out = {"tested": False, "vetoTrain": False, "nCollinear": 0, "nAligned": 0, "nRepeats": 0,
           "perVisit": [], "perpTolArcsec": float(perp_tol_arcsec),
           "windowArcsec": float(window_arcsec), "alignTolDeg": float(align_tol_deg),
           "alignLenMinPx": float(align_len_min_px), "minAligned": int(min_aligned)}
    p1, p2 = radec_to_unit(a.ra, a.dec), radec_to_unit(b.ra, b.dec)
    n = np.cross(p1, p2)
    nn = float(np.linalg.norm(n))
    if nn <= 0.0:
        return out                       # coincident members -- the circle is undefined; leave untested
    n = n / nn
    mid = p1 + p2; mid /= np.linalg.norm(mid)
    e2 = np.cross(n, mid)                # along-track basis: along = atan2(u.e2, u.mid), 0 at the midpoint
    on_line = {}                         # visit -> along positions of on-line dets (for nRepeats)
    tested = False
    for mem, pm in ((a, p1), (b, p2)):
        v = int(mem.visit)
        if v not in train_arrays:
            out["perVisit"].append({"visit": v, "nCollinear": None, "nAligned": None})
            continue
        tested = True
        u, u0, u1, length = train_arrays[v]
        perp = np.abs(np.arcsin(np.clip(u @ n, -1.0, 1.0))) * r2as
        along = np.arctan2(u @ e2, u @ mid) * r2as
        selfsep = np.degrees(2.0 * np.arcsin(np.minimum(np.linalg.norm(u - pm, axis=1) / 2.0, 1.0))) * 3600.0
        m = (perp < perp_tol_arcsec) & (np.abs(along) < window_arcsec) & (selfsep > 2.0)
        ncol, nal = int(m.sum()), 0
        idx = np.flatnonzero(m & (length > align_len_min_px))
        if len(idx):
            t = np.cross(np.broadcast_to(n, (len(idx), 3)), u[idx])   # local circle direction (tangent)
            t /= np.linalg.norm(t, axis=1, keepdims=True)
            pa_c = _sky_pa_mod180(u[idx], t)
            d01 = u1[idx] - u0[idx]                                   # det's own trail direction,
            d01 -= np.sum(d01 * u[idx], axis=1, keepdims=True) * u[idx]   # projected to the tangent plane
            pa_d = _sky_pa_mod180(u[idx], d01)
            dpa = np.abs(pa_d - pa_c); dpa = np.minimum(dpa, 180.0 - dpa)
            nal = int((dpa < align_tol_deg).sum())
        on_line[v] = along[m]
        out["nCollinear"] += ncol; out["nAligned"] += nal
        out["perVisit"].append({"visit": v, "nCollinear": ncol, "nAligned": nal})
    if len(on_line) == 2:
        a1, a2 = on_line.values()
        if len(a1) and len(a2):
            out["nRepeats"] = int((np.abs(a1[:, None] - a2[None, :]) < 1.5).any(axis=1).sum())
    out["tested"] = bool(tested)
    out["vetoTrain"] = bool(tested and out["nAligned"] >= min_aligned)
    return out


def fpp_block(calib, n1, n2, dt_min, visits=None):
    """Per-pair chance-link expectation lambda = k*n1*n2*(dt/dt_ref)^p from the null-donor
    calibration (fpp_2v_chance.json; k factor-~2 with 4 donors). n1/n2 = the two visits' post-floor
    linkable pools, dt_min = the pair gap; the dt^2 term is the measured chance-annulus law.
    perAlertShare (= lambda / #alerts this pair produced) is filled AFTER the night's loop by
    _finalize_fpp, once the pair's alert count is known: it is the pre-pixel-evidence prior that
    any given alert from this pair is a chance link."""
    if not calib:
        return None
    lam = (float(calib["k_per_det2"]) * float(n1) * float(n2)
           * (float(dt_min) / float(calib["dt_ref_min"])) ** float(calib.get("dt_power", 2.0)))
    return {"lambdaPair": round(lam, 4), "n1": int(n1), "n2": int(n2),
            "dtMin": round(float(dt_min), 2), "perAlertShare": None,
            "visits": ([int(v) for v in visits] if visits is not None else None),
            "calib": calib.get("calibrated_on"), "calibQuality": calib.get("calib_quality")}


def _finalize_fpp(alerts):
    """Second pass over the night's alerts: fpp.perAlertShare = lambdaPair / (# 2v alerts from the
    same visit pair). min(1, ...) keeps it a probability; pairs with a single alert get the full
    lambda. Non-2v alerts / missing fpp are left untouched."""
    from collections import Counter
    npair = Counter()
    key = lambda f: tuple(f["visits"]) if f.get("visits") else (f["n1"], f["n2"], f["dtMin"])
    for al in alerts:
        f = al.get("fpp")
        if f and al.get("tier") == "2visit":
            npair[key(f)] += 1
    for al in alerts:
        f = al.get("fpp")
        if f and al.get("tier") == "2visit":
            m = max(npair[key(f)], 1)
            f["perAlertShare"] = round(min(1.0, f["lambdaPair"] / m), 4)
            f["nAlertsPair"] = int(m)


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


def build_known_index(known):
    """Pre-build a spatial cKDTree over the known-object sightings ONCE so crossmatch() is O(log M) per
    detection instead of O(M). At LSST scale known.csv is ~1M sightings/night and there are thousands of
    tracks, so the old per-track nested scan (O(tracks x members x M)) was the linker's worst hotspot."""
    if known is None or not len(known):
        return None
    kra = known.ra.to_numpy(); kdec = known.dec.to_numpy()
    return dict(tree=cKDTree(radec_to_unit(kra, kdec)), kmjd=known.mjd.to_numpy(),   # 3-D: RA=0/pole-safe
                kra=kra, kdec=kdec, kobj=known.ObjID.astype(str).to_numpy())


def crossmatch(dets, members, known, tol_arcsec, tol_day, index=None):
    """Best-matching known ObjID for a track's member detections (or '' if none). Pass a prebuilt `index`
    (build_known_index) to avoid rebuilding the tree per track. Results are IDENTICAL to a brute-force
    nearest-neighbour-within-tolerance scan: the tree query uses an inflated radius (superset) and the exact
    angular separation + time window are re-checked per candidate."""
    g = dets.iloc[members]
    ix = index if index is not None else build_known_index(known)
    if ix is None:
        return "", 0.0
    tree, kmjd, kra, kdec, kobj = ix["tree"], ix["kmjd"], ix["kra"], ix["kdec"], ix["kobj"]
    qr = _chord_radius(tol_arcsec / 3600.0 * 1.5)                          # inflated chord radius -> superset
    hits = []
    for _, r in g.iterrows():
        cand = tree.query_ball_point(radec_to_unit(r.ra, r.dec), qr)
        if not cand:
            continue
        cand = np.asarray(cand)
        cand = cand[np.abs(kmjd[cand] - r.mjd) <= tol_day]                  # time window
        if not cand.size:
            continue
        dra = (kra[cand] - r.ra + 180.0) % 360.0 - 180.0                    # RA-wrap-safe small-angle sep
        sep = np.hypot(dra * np.cos(np.radians(r.dec)), kdec[cand] - r.dec) * 3600
        jm = int(np.argmin(sep))
        if sep[jm] <= tol_arcsec:                                          # exact angular check
            hits.append(kobj[cand[jm]])
    if not hits:
        return "", 0.0
    vc = pd.Series(hits).value_counts()
    return vc.index[0], vc.iloc[0] / len(g)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="ADCNN catalog: mjd,ra,dec,ra0,dec0,ra1,dec1,visit[,len_db,score,art_frac,snr]")
    ap.add_argument("--known", default=str(OUTPUTS / "runs/run_night8731/known.csv"))
    ap.add_argument("--out", default=str(OUTPUTS / "runs/run_night8731/tracks.csv"))
    ap.add_argument("--exptime", type=float, default=30.0)
    ap.add_argument("--len-db-min", type=float, default=6.0, help="hard trail-length floor (px); cut ALL shorter dets regardless of source")
    ap.add_argument("--art-frac-max", type=float, default=0.3, help="LSST mask cut")
    ap.add_argument("--score-min", type=float, default=0.0, help="ADCNN stage-2 CNN (trained real/bogus) score floor; raise to thin FP density for 2-visit linking")
    ap.add_argument("--score-candidate-min", type=float, default=0.0, help="TWO-TIER follow-up floor: link down to this score (tier-B CANDIDATE alerts), labelling alerts confidenceTier A (both members>=--score-hiconf) vs B. 0=off (use --score-min as the single floor). The op sets 0.60 for the WFD 2-visit-pair product; do NOT use on dense fields (0.60 explodes the link).")
    ap.add_argument("--score-hiconf", type=float, default=0.80, help="weakest-member score boundary for confidenceTier A (high-confidence) vs B (candidate)")
    ap.add_argument("--npt", type=int, default=2, help="min detections (distinct visits) per track")
    ap.add_argument("--vel-frac", type=float, default=0.30)
    ap.add_argument("--max-rms", type=float, default=1.0, help="arcsec; LINEAR motion fit RMS (physical_check, >=3 epochs)")
    ap.add_argument("--pa-tol", type=float, default=20.0, help="deg; trail PA vs motion PA agreement (>=3 epochs)")
    ap.add_argument("--pa-tol-2v", type=float, default=10.0, help="deg; TIGHTER trail-PA tol for 2-visit tier + trail-vs-trail agreement")
    ap.add_argument("--max-arc-2v-min", type=lambda s: s if str(s).lower() == "auto" else float(s),
                    default=40.0, help="2-visit Δt window (min): only pair within the scheduler pair gap; the "
                    "single strongest 2v FP cut (purity 0.28->0.71). 'auto' = derive per-night from the actual "
                    "same-pointing visit-pair gaps (auto_2v_window_min) so longer WFD pairs (e.g. 42min) are not "
                    "silently dropped; the op-point sets 'auto'. None to disable")
    ap.add_argument("--orbit-rate-tol", type=float, default=0.25, help="2-visit bound-orbit velocity-residual tol (frac of trail speed); 0.25 is the purity/recall knee (0.5 was too loose). Tighter=purer")
    ap.add_argument("--epoch-gap-s", type=float, default=40.0, help="seconds; detections closer than this in time are merged into ONE epoch (intra-visit snaps). MUST be < the same-night revisit gap or genuine separate visits merge and 2-visit links are rejected as '1 epoch' -- the shipped 120s WRONGLY merged the ~1-min deep-drilling cadence. 40s separates >=40s-apart visits, merges true back-to-back snaps; no-op for WFD's ~34-min pairs.")
    ap.add_argument("--chi2-2v-max", type=float, default=5.0, help="2-visit COMBINED orbit-fit chi^2 gate -- THE primary purity lever. The GEOMETRIC chi2 (collinearity + trail-vs-motion PA & speed) is the STRONG true/false discriminator (true median ~4 vs false ~39, FAST movers); TIGHTENING it 10->5 lifts the 3sigma completeness at the realistic 34-min WFD cadence from 0.070 to ~0.10 (+~45%%, validated on 80 off-ecliptic lambda fields) at NO purity cost. 0 to disable")
    ap.add_argument("--mfsnr-min-2v", type=float, default=10.0, help="2-visit photometric purity floor: fainter member's matched-filter TRAIL SNR >= this. CADENCE-DEPENDENT DIAL: at the realistic ~34-min WFD pair gap the residual chance-FP need this floor to reach 3sigma -> keep ~10. At RAPID cadence (deep-drilling/short dt, sparse FP) the geometric chi2 alone carries purity -> lower to ~5 to recover the fast FAINT movers (mf_snr ~ point_SNR*sqrt(PSF/trail_area) is low for long trails, so a high floor rejects exactly them). 0 to disable")
    ap.add_argument("--rate-lo-2v", type=float, default=1.0, help="2-visit NEO apparent-rate band low (deg/day)")
    ap.add_argument("--rate-hi-2v", type=float, default=8.0, help="2-visit NEO apparent-rate band high (deg/day)")
    ap.add_argument("--claim-order", choices=["seed", "quality", "preal"], default="seed",
                    help="which candidate claims a shared detection when tracks overlap. 'seed' "
                         "(default, frozen behaviour): longest-first, ties broken by SEEDING order. "
                         "'quality': longest-first, ties broken by best orbit-fit chi2 -- matters in a "
                         "low-threshold stream, where a spurious pair can otherwise claim a member "
                         "before the good pair is reached (measured: 8 of 12 production alerts lost "
                         "from an 11k-alert stream on night 20260630). 'preal' breaks ties by the "
                         "CALIBRATED P(real) instead of chi2 -- chi2 alone let a veto-flagged pair "
                         "steal a member from a validated alert")
    ap.add_argument("--preal-model", default=None,
                    help="calibrated reality model for --claim-order preal "
                         "(default ADCNN/calibration/alert_ranking_model.json)")
    ap.add_argument("--rank-by", choices=["priority", "chi2"], default="priority",
                    help="alert stream ordering. 'priority' (default, frozen): tier + weakest-member "
                         "CNN score. 'chi2': best orbit-fit geometry first -- at stream volume the "
                         "CNN-score ordering buries validated alerts (ranks 3771-9942 of 11150 vs "
                         "22-260 by chi2 on night 20260630)")
    ap.add_argument("--seed-2v", choices=["chord"], default="chord", help="2-visit seeding (position-chord pairs + trail verify). Only 'chord' is supported; the inferior trail-velocity 'cluster' path was removed")
    ap.add_argument("--pos-tol-3v", type=float, default=0.05, help="3+visit cluster radius (deg); 0.05 ~doubles 3v recall vs 0.017 at zero purity cost (physical_check is the gate)")
    ap.add_argument("--rate-min", type=float, default=0.3, help="chord seeder min apparent rate (deg/day)")
    ap.add_argument("--rate-max", type=float, default=10.0, help="chord seeder max apparent rate (deg/day)")
    ap.add_argument("--recur-max", type=int, default=None, help="TP-safe stationarity veto: drop detections recurring at the same sky position in >= this many other same-night visits (residuals recur; a >=1deg/day mover never does). 2 is strong. Needs many visits/night (dense fields); ~no-op for a WFD pair. None=off")
    ap.add_argument("--min-epochs", type=int, default=2, help="distinct time epochs (snaps merged); 2 enables 2-visit linking")
    ap.add_argument("--tol-arcsec", type=float, default=5.0)
    ap.add_argument("--tol-day", type=float, default=0.02)
    ap.add_argument("--op-point", default=os.environ.get("LINK_OP_POINT"),
                    help="JSON op-point config (link_op_point.json): sets the calibrated 2v/3v params; any CLI flag overrides its value")
    ap.add_argument("--max-visit-pairs", type=int, default=200,
                    help="cap on same-night visit PAIRS the 2-visit chord seeder enumerates (O(V^2)); guards against a deep-drilling night (~100 visits -> ~5000 pairs). Keeps the nearest-in-time pairs and warns when it triggers")
    ap.add_argument("--promote-3v", action=argparse.BooleanOptionalAction, default=True,
                    help="promote a passing 2-visit chord pair to the PURE 3+visit tier when a real same-night detection lies on its precise 2-centroid track (the (FP)^N collapse). Free purity for the multi-visit subset; no-op for a WFD pair. --no-promote-3v to disable")
    ap.add_argument("--promote-tol-arcsec", type=float, default=5.0,
                    help="position tolerance (arcsec) for attaching the 3rd detection to the chord-extrapolated track in --promote-3v")
    ap.add_argument("--promote-from", choices=["survivors", "raw"], default="survivors",
                    help="which 2v pairs get extended to triplets. 'survivors' (default): only the "
                    "prefilter-passing pairs -- ~1700x faster on dense fields (0.3s vs ~500s), same real "
                    "triplets, fewer chance triplets. 'raw': every seed pair (exhaustive; a triplet never "
                    "needs a passing 2v parent) -- intractable on dense/ecliptic fields, kept for audit.")
    ap.add_argument("--alerts-out", default=None,
                    help="JSONL same-night alert stream (one actionable candidate per line: endpoints, motion vector, forward-predicted ephemeris, confidence). Default: alerts.jsonl beside --out. --no-alerts to disable")
    ap.add_argument("--no-alerts", action="store_true", help="do not emit the JSONL alert stream")
    ap.add_argument("--alerts-top-n", type=int, default=None,
                    help="the recommended per-night follow-up budget (op: 50). Only ENFORCED as a hard cap when --cap-alerts is given; otherwise it is advisory and ALL alerts are published (ranked).")
    ap.add_argument("--cap-alerts", action="store_true",
                    help="OPT-IN: truncate the emitted alert stream to the top --alerts-top-n by priorityScore "
                    "(the follow-up budget). DEFAULT: publish ALL ranked alerts (no cap) -- the alert file is "
                    "priorityScore-ordered so a follow-up consumer can read top-down and stop at its own capacity.")
    ap.add_argument("--report", action="store_true",
                    help="OPT-IN: after writing the alert stream, render the human-inspection QA package "
                    "beside it -- report/rankNN_<alertId>_<class>.png cutout stamps + report/overlay_* full "
                    "trail overlays (ADCNN.qa.trail_overlays), the ranked ALERT_REPORT.md, and the "
                    "machine-readable alert_report.csv (ADCNN.qa.alert_report). Needs the --dets CSV to "
                    "carry fits_path (the masked adcnn dets do) and the diffim pixels to be reachable; "
                    "report failures WARN, never fail the link run. Default off = exact no-op")
    ap.add_argument("--seed-3v-arc-min", type=float, default=120.0,
                    help="3v-FIRST seeding: also seed chord pairs up to this arc (min) and use them ONLY to extend to triplets (3-epoch-gated), so a mover with no pair inside the 2v window (e.g. visits 0/50/100 min) is still found. 0 to disable")
    ap.add_argument("--stat-tol-arcsec", type=float, default=3.0,
                    help="catalog stationarity veto: counterpart match radius in the OTHER member visit "
                    "(audited 2026-07-02: kills 88%% of false 2v alerts at ~2%%/alert true cost; FLAG, never drop)")
    ap.add_argument("--stat-min-disp-arcsec", type=float, default=10.0,
                    help="stationarity validity guard: test only when the expected displacement rate*dt "
                    "exceeds this (motion-aware -- a short-gap real mover must not self-veto). Below it the "
                    "alert is marked stationarity.testable=false")
    ap.add_argument("--no-stationarity", action="store_true", help="skip the catalog stationarity annotation")
    ap.add_argument("--fpp-calib", default=str(HL / "fpp_2v_chance.json"),
                    help="null-donor chance-link calibration JSON (fpp_2v_chance.json) for the per-alert "
                    "fpp block: lambda = k*n1*n2*(dt/dt_ref)^2. 'none' or a missing file disables the block")
    ap.add_argument("--static-catalog", default=None,
                    help="bright-static template-footprint catalog (parquet/CSV with ra,dec,mag columns; "
                    "build with ADCNN/linking/build_static_catalog.py from the coadd `object` tables). "
                    "Enables the SEED-EXCLUSION static veto: 2v seed pairs whose BOTH members lie within "
                    "--static-radius-arcsec of a mag<--static-mag-max static are never seeded (measured "
                    "2026-07-02 on embargo 0630: ~90%% of the dense-field 2v FP background is template-"
                    "static subtraction residuals, not chance); SINGLE-static alerts get a staticVeto "
                    "annotation + ranking demotion -- FLAG, never drop. Omit = veto off (default)")
    ap.add_argument("--static-mag-max", type=float, default=20.0,
                    help="static-veto catalog magnitude cut (brightest griz cModel). 20.0 = the measured "
                    "knee: removes the bright-star wing artifact population at ~1.3%%/det true-mover cost; "
                    "deeper cuts eat real movers fast (mag<23 -> 3.4%%/det, full depth -> 14-43%%)")
    ap.add_argument("--static-radius-arcsec", type=float, default=3.0,
                    help="static-veto match radius: 3\" reaches the 2-3\" bright-star WINGS where the "
                    "subtraction-residual dipoles live (per-visit kill fraction 0.25->0.60 going 2\"->3\")")
    ap.add_argument("--train-veto", action="store_true",
                    help="shared-great-circle LINE veto for 2v alerts (measured 2026-07-03, embargo "
                    "0629+0630): flag a pair whose members sit on a line of >=--train-min-aligned "
                    "trail-PA-aligned dets -- satellite-train glint chains (train member B glints where "
                    "A glinted a visit earlier; STREAK masking sees none of it) AND static template-"
                    "artifact lines both fire; isolated real movers score ~0-1 (golden NEO = 1 vs "
                    "pathologies 11-15). FLAG + ranking demotion, never drop. Default off = exact no-op")
    ap.add_argument("--train-perp-arcsec", type=float, default=2.5,
                    help="train veto: max perpendicular distance (arcsec) from the members' great "
                    "circle for a det to count as on-line (measured train knots sit <1\" off-plane)")
    ap.add_argument("--train-window-arcsec", type=float, default=1800.0,
                    help="train veto: along-track window (arcsec, each side of the member midpoint) -- "
                    "bounds the line locally so a whole-focal-plane circle cannot accumulate chance dets")
    ap.add_argument("--train-align-deg", type=float, default=20.0,
                    help="train veto: max |trail PA - local circle PA| (deg, mod 180) for an on-line "
                    "det to vote 'aligned' (the pathologies' knot trails lie ALONG the line)")
    ap.add_argument("--train-align-len-min", type=float, default=5.0,
                    help="train veto: only dets with trail length > this (px) vote on alignment (a "
                    "shorter trail's PA is noise)")
    ap.add_argument("--train-min-aligned", type=int, default=10,
                    help="train veto: flag when the two visits' aligned counts sum to >= this. 10 = "
                    "the measured separation (pathologies {11,12,14,15,15}, clean <=8, golden NEO 1)")
    a = ap.parse_args()
    # Overlay the calibrated op-point JSON: it sets each param UNLESS that flag was passed explicitly on the CLI.
    if a.op_point and os.path.exists(a.op_point):
        _op = json.load(open(a.op_point))
        _flag = {"score_min": "--score-min", "score_candidate_min": "--score-candidate-min",
                 "score_hiconf": "--score-hiconf", "chi2_2v_max": "--chi2-2v-max", "mfsnr_min_2v": "--mfsnr-min-2v",
                 "rate_lo_2v": "--rate-lo-2v", "rate_hi_2v": "--rate-hi-2v", "pa_tol": "--pa-tol",
                 "pa_tol_2v": "--pa-tol-2v", "max_rms": "--max-rms", "pos_tol_3v": "--pos-tol-3v",
                 "max_arc_2v_min": "--max-arc-2v-min", "promote_3v": "--promote-3v",
                 "promote_tol_arcsec": "--promote-tol-arcsec", "alerts_top_n": "--alerts-top-n",
                 "seed_3v_arc_min": "--seed-3v-arc-min", "promote_from": "--promote-from",
                 # len_db_min used to live ONLY as a CLI default, so the trail-length floor was the
                 # one op parameter an op-point file could not state. Additive: the frozen alert op
                 # carries no such key, so its behaviour is byte-unchanged (floor stays 6.0).
                 "len_db_min": "--len-db-min"}
        _applied = [f"{k}={_op[k]}" for k, fl in _flag.items()
                    if k in _op and fl not in sys.argv and (setattr(a, k, _op[k]) or True)]
        if _applied:
            print(f"[trail-link] op-point {a.op_point}: {', '.join(_applied)}", flush=True)

    d = pd.read_csv(a.dets)
    n0 = len(d)
    # FULL pre-floor catalog snapshot for the stationarity counterpart trees: the veto is
    # ADCNN-vs-ADCNN, so its counterpart catalog must be the DEEPEST list available (every det the
    # detector emitted, before the art/len/score cuts below) -- that is what preserves the faint-bin
    # (sub-5sigma) kill power. Only the 4 columns the KD-trees need are kept.
    d_all = (d[["ra", "dec", "mjd", "visit"]].copy()
             if (not a.no_stationarity and {"ra", "dec", "mjd", "visit"} <= set(d.columns)) else None)
    if d_all is not None:
        d_all["night"] = np.floor(d_all.mjd - 0.5).astype(int)
    # FULL pre-floor snapshot for the train veto: like the stationarity trees, the line population
    # must be the DEEPEST det list available -- the glint/artifact knots that populate the line are
    # exactly the faint dets the score/length floors remove. Thresholds were measured on this
    # pre-floor catalog, so cutting first would silently shift the statistic.
    d_train = None
    if a.train_veto:
        _tcols = ["ra", "dec", "ra0", "dec0", "ra1", "dec1", "length", "mjd", "visit"]
        _tmiss = [c for c in _tcols if c not in d.columns]
        if _tmiss:
            raise SystemExit(f"--train-veto needs columns {_tmiss} in --dets")
        d_train = d[_tcols].copy()
        d_train["night"] = np.floor(d_train.mjd - 0.5).astype(int)
        print(f"[trail-link] train veto ON: perp<{a.train_perp_arcsec:g}\" window +-{a.train_window_arcsec:g}\" "
              f"align<{a.train_align_deg:g}deg len>{a.train_align_len_min:g}px -> flag at "
              f">={a.train_min_aligned} aligned line dets", flush=True)
    fpp_calib = None
    if a.fpp_calib and str(a.fpp_calib).lower() != "none" and os.path.exists(a.fpp_calib):
        fpp_calib = json.load(open(a.fpp_calib))
        print(f"[trail-link] fpp calib: {a.fpp_calib} (k={fpp_calib['k_per_det2']:.3g}/det^2 "
              f"@ dt_ref {fpp_calib['dt_ref_min']}min)", flush=True)
    if "art_frac" in d and a.art_frac_max > 0:
        d = d[d.art_frac.fillna(0.0) < a.art_frac_max]   # NaN (unmeasured mask) -> keep, never silently drop
    if "len_db" in d and a.len_db_min > 0:
        d = d[d.len_db >= a.len_db_min]
    # TWO-TIER follow-up: link down to the CANDIDATE floor when set (op: 0.60), else the single score_min.
    # The hi-confidence boundary (score_hiconf, 0.80) only LABELS each alert A/B in build_alert -- it does
    # not gate here, so tier-B (0.60-0.80) pairs still pass the mfsnr/chi2/rate purity gates.
    _score_floor = a.score_candidate_min if (a.score_candidate_min and a.score_candidate_min > 0) else a.score_min
    if "score" in d and _score_floor > 0:
        d = d[d.score >= _score_floor]
        if a.score_candidate_min and a.score_candidate_min > 0:
            print(f"[trail-link] TWO-TIER follow-up: candidate floor {_score_floor}, hi-conf (tier A) >= {a.score_hiconf}", flush=True)
    d = d.reset_index(drop=True)
    need = ["mjd", "ra", "dec", "ra0", "dec0", "ra1", "dec1", "visit"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"--dets missing {miss}")
    # TEMPLATE-FOOTPRINT STATIC VETO (measured 2026-07-02 on embargo 0630, expt_staticveto/RESULTS.md):
    # ~90% of the dense-field 2v FP background is subtraction residuals living in the 2-3" WINGS of
    # bright (mag<20) coadd statics -- structured, NOT chance (95/107 floor-0.5 alerts were static-
    # static pairs; exactly the clean ones survived removal). Flag every linkable det within the veto
    # radius of a bright static; the night loop then EXCLUDES static-static pairs from 2v seeding
    # (drop_static_static_pairs) and ANNOTATES + DEMOTES single-static alerts (staticVeto block --
    # FLAG, never drop; the det-level true-mover flag cost is ~1.3%). No catalog = exact no-op.
    static_cfg = None
    if a.static_catalog and str(a.static_catalog).lower() != "none":
        if not os.path.exists(a.static_catalog):
            raise SystemExit(f"--static-catalog not found: {a.static_catalog}")
        _sc = (pd.read_parquet(a.static_catalog) if str(a.static_catalog).endswith((".parquet", ".pq"))
               else pd.read_csv(a.static_catalog))
        _sc = _sc[np.isfinite(_sc.mag) & (_sc.mag < a.static_mag_max)]
        if not len(_sc):
            raise SystemExit(f"--static-catalog {a.static_catalog}: no statics brighter than "
                             f"mag {a.static_mag_max} -- wrong catalog or cut")
        _sd, _si = cKDTree(radec_to_unit(_sc.ra.to_numpy(), _sc.dec.to_numpy())).query(
            radec_to_unit(d.ra.to_numpy(), d.dec.to_numpy()), k=1)
        _sep = np.degrees(2.0 * np.arcsin(np.minimum(np.atleast_1d(_sd) / 2.0, 1.0))) * 3600.0
        d["static_veto"] = _sep <= a.static_radius_arcsec
        d["static_sep_arcsec"] = _sep
        d["static_mag"] = _sc.mag.to_numpy()[np.atleast_1d(_si)]
        static_cfg = dict(magMax=float(a.static_mag_max), radiusArcsec=float(a.static_radius_arcsec))
        print(f"[trail-link] static veto: {len(_sc)} statics (mag<{a.static_mag_max:g}) from "
              f"{a.static_catalog} -> {int(d.static_veto.sum())}/{len(d)} dets flagged "
              f"(r={a.static_radius_arcsec:g}\")", flush=True)
    known = pd.read_csv(a.known)
    kindex = build_known_index(known)   # one cKDTree over all known sightings (crossmatch O(log M)/det at scale)
    d["night"] = np.floor(d.mjd - 0.5).astype(int)
    print(f"[trail-link] {n0} dets -> {len(d)} after cuts | nights {sorted(d.night.unique())}", flush=True)
    # PER-NIGHT auto 2v Δt window: size the gap filter to the data's actual same-pointing pair gaps so a
    # longer WFD pair (e.g. this night's 42.5min vs a hardcoded 40) is not silently dropped. Resolved here,
    # once d's visits are known; --max-arc-2v-min auto (op default) triggers it, an explicit number overrides.
    if isinstance(a.max_arc_2v_min, str):
        a.max_arc_2v_min, _winfo = auto_2v_window_min(d)
        print(f"[trail-link] auto 2v window: {_winfo}", flush=True)

    rows = []
    alerts = []
    emit_alerts = not a.no_alerts
    if emit_alerts:
        from ADCNN.linking.rank_alerts import build_alert, write_alerts
    for night, dn in d.groupby("night"):
        dn = dn.reset_index(drop=True)
        # stationarity counterpart trees over the FULL pre-floor catalog (ADCNN-vs-ADCNN, sub-5sigma
        # deep) + the post-floor per-visit pools/mjds the fpp chance-link block needs.
        stat_trees, stat_mjd = {}, {}
        if d_all is not None:
            for v, gg in d_all[d_all.night == night].groupby("visit"):
                stat_trees[int(v)] = (cKDTree(radec_to_unit(gg.ra.to_numpy(), gg.dec.to_numpy())), len(gg))
                stat_mjd[int(v)] = float(gg.mjd.median())
        # train-veto per-visit line population (full pre-floor): det centers + trail endpoints as
        # unit vectors, so per-alert great-circle geometry is pure vectorized algebra.
        train_arrays = {}
        if d_train is not None:
            for v, gg in d_train[d_train.night == night].groupby("visit"):
                train_arrays[int(v)] = (radec_to_unit(gg.ra.to_numpy(), gg.dec.to_numpy()),
                                        radec_to_unit(gg.ra0.to_numpy(), gg.dec0.to_numpy()),
                                        radec_to_unit(gg.ra1.to_numpy(), gg.dec1.to_numpy()),
                                        gg.length.to_numpy())
        pool = dn.groupby("visit").size().to_dict()
        vmjd_n = dn.groupby("visit").mjd.median().to_dict()
        if a.recur_max is not None:
            from ADCNN.pipelines.heliolinc.recurrence import add_recurrence
            dn = add_recurrence(dn)
            dn = dn[dn.recur < a.recur_max].reset_index(drop=True)   # TP-safe (real movers have recur==0)
        # 3+visit via (looser) trail-velocity clustering; 2-visit via precise position-chord seeding.
        # (The old `--seed-2v cluster` 2-visit path was removed: it linked ~4x fewer real pairs at ~10x higher
        # FP than chord seeding and was never used in production.)
        _, clus = link(dn, exptime_s=a.exptime, npt=3, pos_tol_deg=a.pos_tol_3v,
                       vel_frac=a.vel_frac, min_visits=3)
        cand = list(clus)
        if a.min_epochs <= 2:
            cpairs = chord_seed_pairs(dn, max_arc_min=(a.max_arc_2v_min or 1e9),
                                      rate_min=a.rate_min, rate_max=a.rate_max,
                                      max_visit_pairs=a.max_visit_pairs)
            # SEED-EXCLUSION static veto: a static-static pair is a repeating-artifact self-link --
            # never seed it. Single-static pairs stay (annotated + demoted at the alert, not dropped).
            if static_cfg is not None:
                _sflag = dn.static_veto.to_numpy()
                _nsp = len(cpairs)
                cpairs = drop_static_static_pairs(cpairs, _sflag)
                print(f"  [static-veto] night {night}: 2v seed pairs {_nsp} -> {len(cpairs)} "
                      f"(static-static excluded)", flush=True)
            # EXACT vectorized fast path (prefilter_2v_pairs): drop pairs whose partial chi2 already
            # exceeds the gate -- behavior-identical (the orbit-residual term is >=0), removes the
            # per-pair orbit fit for ~99% of chance pairs. Applied to the 2v CANDIDATE list only; the
            # raw cpairs still feed promotion/3v-first seeding below (3v candidates must never require
            # a passing 2v parent).
            surv2v = prefilter_2v_pairs(dn, cpairs, a.chi2_2v_max, exptime_s=a.exptime)
            cand += surv2v
            # PROMOTE 2v->3v: attach a consistent 3rd same-night detection on the precise chord track.
            # A real 3rd on the 2-point line -> pure 3v tier (free purity for the multi-visit subset);
            # no-op for a 2-visit WFD night. The pair stays in `cand` as a fallback if the triplet fails.
            #
            # SCALABILITY (--promote-from): on a DENSE field cpairs is ~1e6 (mostly chance pairs) and
            # extending EVERY raw pair to triplets is the dominant cost -- measured ~500s (the ~47-min
            # 20260630-ecliptic hang) vs 0.3s when extending only the ~1e2 prefilter survivors, which also
            # manufactures ~1e4 chance triplets that physical_check must then reject. Default 'survivors'
            # extends only the physically-plausible (2v-gate-passing) pairs: a real 3-visit mover's
            # constituent pairs pass the 2v trail-PA/speed gate (its 3 dets are collinear at constant
            # velocity), so real triplets are preserved while chance-triplet manufacture is cut. 'raw'
            # restores the exhaustive behaviour (extend every seed pair -- a triplet then never requires a
            # passing 2v parent, at the dense-field cost). Empirically validated equal on the NY2 anchor +
            # real-night recoveries; use 'raw' only to audit that equivalence.
            if a.promote_3v:
                _promote_src = surv2v if a.promote_from == "survivors" else cpairs
                cand += extend_to_triplets(dn, _promote_src, pos_tol_arcsec=a.promote_tol_arcsec)
                # 3v-FIRST seeding: the 40-min 2v arc cap is an FP lever for the PAIR tier, not a physical
                # constraint on triplets -- a real mover seen in visits at e.g. 0/50/100 min has NO pair
                # inside the 2v window and was previously unfindable. Seed pairs in a WIDER window
                # (seed_3v_arc_min) and use them ONLY to extend to triplets (the wide pairs themselves are
                # NOT kept as 2v candidates); the triplets face the same 3-epoch physical_check (linear-RMS
                # + trail-PA + speed) as every other 3+visit track -- geometry carries the purity, and NO
                # constituent 2v pair is required to pass the 2v alert gates.
                if a.seed_3v_arc_min and a.seed_3v_arc_min > (a.max_arc_2v_min or 0):
                    wide = chord_seed_pairs(dn, max_arc_min=a.seed_3v_arc_min,
                                            rate_min=a.rate_min, rate_max=a.rate_max,
                                            max_visit_pairs=a.max_visit_pairs)
                    # same seed-exclusion as the main 2v path: a triplet built ON a static-static
                    # pair is bogus by construction (2/3 members are repeating artifacts); a real
                    # 3-visit mover keeps 2 other constituent pairs to seed the same triplet.
                    if static_cfg is not None:
                        wide = drop_static_static_pairs(wide, dn.static_veto.to_numpy())
                    # SAME survivor-gating as the main path (dominant on dense fields): the wide window
                    # (seed_3v_arc_min=120min default) seeds ~1e7 pairs on an ecliptic field -- 13.6M on
                    # 20260630, extending all of them raw = the ~95-min hang. Prefilter to the physically-
                    # plausible pairs first (13.6M->3.3k in 11s); a real >window-arc mover passes (its long-
                    # arc chord is well-determined so trail-PA/speed agree). 'raw' keeps the exhaustive path.
                    if a.promote_from == "survivors":
                        wide = prefilter_2v_pairs(dn, wide, a.chi2_2v_max, exptime_s=a.exptime)
                    cand += extend_to_triplets(dn, wide, pos_tol_arcsec=a.promote_tol_arcsec)
        cand.sort(key=len, reverse=True)   # 3+visit (longer) first; a triplet's dets aren't re-reported as pairs
        _PR_COEF = _PR_DOM = None
        _p_real = None
        if a.claim_order == "preal":
            from ADCNN.qa.rerank_alerts import p_real as _p_real, DEFAULT_MODEL as _PR_MODEL
            _m = json.load(open(a.preal_model or _PR_MODEL))
            _PR_COEF, _PR_DOM = _m["coef"], _m.get("domain")
            print(f"[trail-link] claim priority = calibrated P(real) "
                  f"(model fit on night {_m.get('night')})", flush=True)
        if a.claim_order in ("quality", "preal"):
            # Two-pass claim: evaluate EVERY candidate, then let the best-fitting one claim its
            # detections. The single-pass path below sorts by length only, so among 2-visit pairs
            # (all length 2, stable sort) the SEEDING order decides who claims a detection. That is
            # harmless at the frozen op (few survivors) and wrong in a low-threshold stream, where a
            # spurious pair can claim a member before the good pair is reached: MEASURED on night
            # 20260630, only 4 of the 12 production alerts survived into an 11,150-alert stream, the
            # other 8 having had a member claimed by another pairing. Same compute (physical_check
            # runs once per candidate either way), more memory (results held before claiming).
            evald = []
            for members in cand:
                st = {}
                ok, info, n_ep = physical_check(
                    dn, members, a.exptime, pa_tol_deg=a.pa_tol, lin_rms_arcsec=a.max_rms,
                    min_epochs=a.min_epochs, epoch_gap_s=a.epoch_gap_s, pa_tol_2v_deg=a.pa_tol_2v,
                    score_2v_min=0.0, max_arc_2v_min=a.max_arc_2v_min,
                    orbit_rate_tol=a.orbit_rate_tol, perp_collinear_2v_arcsec=None, snr_frac_2v=None,
                    chi2_2v_max=(a.chi2_2v_max if a.chi2_2v_max and a.chi2_2v_max > 0 else None),
                    mfsnr_min_2v=(a.mfsnr_min_2v if a.mfsnr_min_2v and a.mfsnr_min_2v > 0 else None),
                    rate_lo_2v=a.rate_lo_2v, rate_hi_2v=a.rate_hi_2v, out=st)
                if ok:
                    # 3+visit candidates carry no pair chi2; they sort first on length anyway.
                    c2 = float(st.get("chi2", np.inf))
                    if a.claim_order == "preal":
                        # Claim by the CALIBRATED reality probability, not chi2 alone. Measured on
                        # 20260630: ordering by chi2 let a VETO-FLAGGED pair (final rank 8627) claim
                        # a member of validated science alert 2v_61221_000007 purely because its
                        # geometry scored better, orphaning the real pair -- its other member then
                        # appeared in no alert at all. chi2 is one term of the evidence, not the
                        # evidence; P(real) weighs it against the CNN score and mf_snr as fitted.
                        g_ = dn.iloc[members]
                        _al = {"vetting": {"score_min": float(g_.score.min()) if "score" in g_ else None,
                                           "mfsnr_min": float(g_.mf_snr.min()) if "mf_snr" in g_ else None},
                               "orbit": {"chi2": c2}}
                        pr = _p_real(_al, _PR_COEF, _PR_DOM) if _PR_COEF else None
                        evald.append((-len(members), -(pr if pr is not None else -1.0), members))
                    else:
                        evald.append((-len(members), c2, members))
            evald.sort(key=lambda t: (t[0], t[1]))     # longest first, then best quality
            cand = [m for _l, _c, m in evald]
            print(f"[trail-link] claim-order={a.claim_order}: {len(cand)} candidates passed, claiming "
                  f"{'highest-P(real)' if a.claim_order == 'preal' else 'best-chi2'} first", flush=True)
        npass = 0; used = set()
        for members in cand:
            # NB: purity is carried by the chi2 gate; the independent AND-threshold discriminators
            # (score_2v_min, perp_collinear, snr_frac) are off in the shipped op-point (analysis tools that
            # bypass chi2 set them directly when calling physical_check).
            ok, info, n_ep = physical_check(dn, members, a.exptime, pa_tol_deg=a.pa_tol,
                                            lin_rms_arcsec=a.max_rms, min_epochs=a.min_epochs,
                                            epoch_gap_s=a.epoch_gap_s,
                                            pa_tol_2v_deg=a.pa_tol_2v, score_2v_min=0.0,
                                            max_arc_2v_min=a.max_arc_2v_min, orbit_rate_tol=a.orbit_rate_tol,
                                            perp_collinear_2v_arcsec=None, snr_frac_2v=None,
                                            chi2_2v_max=(a.chi2_2v_max if a.chi2_2v_max and a.chi2_2v_max > 0 else None),
                                            mfsnr_min_2v=(a.mfsnr_min_2v if a.mfsnr_min_2v and a.mfsnr_min_2v > 0 else None),
                                            rate_lo_2v=a.rate_lo_2v, rate_hi_2v=a.rate_hi_2v)
            if not ok:
                continue
            # dedup across ALL tiers: cand is sorted longest-first, so the best (longest) track claims its
            # detections; shorter overlapping candidates -- a 2v pair under a 3v track, OR two promoted
            # triplets sharing an object's detections -- are skipped (prevents reporting one object N times).
            if any(m in used for m in members):
                continue
            used.update(members)
            npass += 1
            rms, speed = fit_residual(dn, members, a.exptime)
            obj, frac = crossmatch(dn, members, known, a.tol_arcsec, a.tol_day, index=kindex)
            g = dn.iloc[members]
            # numeric orbit-fit columns for the (candidate-grade) 2-visit stream: chi2 is the geometry
            # gate statistic; a_au/ecc are the DEGENERATE argmin diagnostics (kept in tracks.csv for
            # audit only); the adm_* admissible-region ranges are what the alert packet publishes.
            if n_ep == 2:
                c2, ci = pair_chi2(g, a.exptime); chi2v, av, ev = c2, ci["a"], ci["e"]
                adm = {k: ci.get(k, np.nan) for k in ADM_KEYS}
            else:
                chi2v, av, ev = np.nan, np.nan, np.nan
                adm = dict.fromkeys(ADM_KEYS, np.nan)
            tier = "2visit" if n_ep == 2 else "3+visit"
            status = "CONFIRMED" if obj else "NEW"
            # 2v veto-stack annotations (FLAG, never drop): catalog stationarity + chance-link fpp
            # + template-footprint staticVeto (static-static pairs never got here -- seed-excluded;
            # what remains is 0- or 1-static, and the 1-static alerts are demoted in ranking).
            stat = fpp = sveto = tveto = None
            if n_ep == 2:
                if stat_trees:
                    stat = stationarity_check(g, stat_trees, stat_mjd, tol_arcsec=a.stat_tol_arcsec,
                                              min_disp_arcsec=a.stat_min_disp_arcsec)
                if train_arrays:
                    tveto = train_veto_check(g, train_arrays, perp_tol_arcsec=a.train_perp_arcsec,
                                             window_arcsec=a.train_window_arcsec,
                                             align_tol_deg=a.train_align_deg,
                                             align_len_min_px=a.train_align_len_min,
                                             min_aligned=a.train_min_aligned)
                vv = sorted(int(v) for v in g.visit.unique())
                if fpp_calib and len(vv) == 2:
                    dtm = abs(vmjd_n[vv[1]] - vmjd_n[vv[0]]) * 1440.0
                    fpp = fpp_block(fpp_calib, pool.get(vv[0], 0), pool.get(vv[1], 0), dtm, visits=vv)
                if static_cfg is not None:
                    _gs = g.sort_values("mjd")
                    sveto = dict(nStaticMembers=int(_gs.static_veto.sum()), **static_cfg,
                                 members=[dict(visit=int(r.visit), isStatic=bool(r.static_veto),
                                               sepArcsec=_jf(r.static_sep_arcsec),
                                               staticMag=_jf(r.static_mag))
                                          for _, r in _gs.iterrows()])
            rows.append(dict(night=int(night), ndet=len(members), nvisit=g.visit.nunique(),
                             n_epochs=n_ep, tier=tier,
                             arc_hr=(g.mjd.max() - g.mjd.min()) * 24, rms_arcsec=rms, speed_degday=speed,
                             chi2=chi2v, a_au=av, ecc=ev, ra=g.ra.mean(), dec=g.dec.mean(), check=info,
                             match_obj=obj, match_frac=frac, status=status, **adm,
                             veto_stationary=(stat or {}).get("vetoStationary"),
                             stat_testable=(stat or {}).get("testable"),
                             fpp_lambda_pair=(fpp or {}).get("lambdaPair"),
                             n_static_members=(sveto or {}).get("nStaticMembers"),
                             n_train_aligned=(tveto or {}).get("nAligned"),
                             veto_train=(tveto or {}).get("vetoTrain")))
            if emit_alerts:
                _oc = str(g["obscode"].iloc[0]) if "obscode" in g.columns else os.environ.get("OBSCODE", "I11")
                alerts.append(build_alert(g, alert_id=f"{tier[:2]}_{int(night)}_{len(rows)-1:06d}",
                                          night=night, obscode=_oc, status=status, tier=tier,
                                          chi2=chi2v, orbit_adm=adm, rms_arcsec=rms,
                                          match_obj=obj, match_frac=frac, hiconf_score=a.score_hiconf,
                                          stationarity=stat, fpp=fpp, static_veto=sveto,
                                          train_veto=tveto, rate_lo=a.rate_lo_2v))
        print(f"  night {night}: {len(dn)} dets -> {len(cand)} candidates, {npass} passed", flush=True)
    TRACK_COLS = ["night", "ndet", "nvisit", "n_epochs", "tier", "arc_hr", "rms_arcsec", "speed_degday",
                  "chi2", "a_au", "ecc", "ra", "dec", "check", "match_obj", "match_frac", "status",
                  *ADM_KEYS, "veto_stationary", "stat_testable", "fpp_lambda_pair", "n_static_members",
                  "n_train_aligned", "veto_train"]
    T = pd.DataFrame(rows, columns=TRACK_COLS)   # always carry the header, so an empty result is a valid CSV
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(a.out, index=False)
    if emit_alerts:
        apath = a.alerts_out or str(Path(a.out).with_name("alerts.jsonl"))
        # fpp second pass: per-alert chance share = lambdaPair / (# alerts the same visit pair produced)
        _finalize_fpp(alerts)
        # ranked by continuous priorityScore inside write_alerts (3+visit NEW > 2visit NEW > recovery,
        # then geometric/detector/photometric quality); stationarity-vetoed alerts are DEMOTED after the
        # clean ones (published, never dropped). Publish ALL by default; --cap-alerts opts in to
        # truncating at the --alerts-top-n follow-up budget.
        write_alerts(alerts, apath, top_n=(a.alerts_top_n if a.cap_alerts else None),
                     rank_by=a.rank_by)
        n2new = sum(1 for al in alerts if al["tier"] == "2visit" and al["status"] == "NEW")
        nveto = sum(1 for al in alerts if (al.get("stationarity") or {}).get("vetoStationary"))
        nsv = sum(1 for al in alerts if (al.get("staticVeto") or {}).get("nStaticMembers", 0) >= 1)
        ntv = sum(1 for al in alerts if (al.get("trainVeto") or {}).get("vetoTrain"))
        print(f"[trail-link] alert stream: {len(alerts)} alerts ({n2new} same-night 2-visit NEW, "
              f"{nveto} stationarity-flagged, {nsv} static-flagged, {ntv} train/line-flagged) "
              f"-> {apath}", flush=True)
        if a.report:
            # QA report package (opt-in): imports live in-branch so matplotlib/pixel IO stay out
            # of the default link path; a report failure must never invalidate the science outputs.
            rep_dir = str(Path(apath).with_name("report"))
            rep_args = ["--alerts", apath, "--dets", a.dets, "--out-dir", rep_dir]
            if a.static_catalog:
                rep_args += ["--static-catalog", a.static_catalog,
                             "--static-mag-max", str(a.static_mag_max)]
            try:
                from ADCNN.qa import trail_overlays, alert_report
                trail_overlays.main(rep_args)
                alert_report.main(rep_args)
            except Exception as e:  # noqa: BLE001 -- report is best-effort by contract
                print(f"[trail-link] WARNING: --report failed ({type(e).__name__}: {e}); "
                      f"alert stream itself is intact at {apath}", flush=True)
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
        print(f"[trail-link] WARNING: 0 tracks passed physical_check over {len(d)} dets / "
              f"{d.night.nunique() if len(d) else 0} nights -> empty (header-only) {a.out}. "
              f"Check the input is non-empty and the op-point/cadence are sane.", flush=True)


if __name__ == "__main__":
    main()
