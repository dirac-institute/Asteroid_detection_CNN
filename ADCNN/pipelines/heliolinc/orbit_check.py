"""Per-candidate bound-orbit consistency test for same-night 2-visit trail links.

WHY: a real solar-system object's two trailed tracklets (each = sky position + on-sky angular velocity
over the exposure) must be reproduced by ONE bound heliocentric two-body orbit; a chance pair of false
detections is not. With only two angular POSITIONS an orbit is under-determined (Herget's (rho1,rho2)
family fits any pair), so the discriminating information is the TRAIL-measured angular VELOCITIES.

METHOD (range / range-rate attributable fit, cf. Milani admissible region): the epoch-1 tracklet fixes
the line-of-sight direction and the transverse-velocity direction; the two unknowns are the topocentric
range rho1 and range-rate rhodot1 at epoch 1. For trial (rho1, rhodot1) we build the full heliocentric
state at t1, propagate two-body to t2, and predict the epoch-2 angular position + velocity. We least-
squares fit (rho1, rhodot1) (multi-start over rho1) to the epoch-2 observables; the fit is over-
determined (2 unknowns vs 4 observables). A candidate PASSES only if the best orbit is BOUND, at a
physically plausible heliocentric distance, and reproduces epoch 2 within tolerance. Chance pairs fail.

Geocentric (not topocentric) observer is used — the <~arcsec topocentric parallax over a ~1 h arc is
far below this filter's resolution. Light-time / aberration likewise neglected (plausibility filter).
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache

MU = 0.00029591220828411951  # GM_sun, AU^3/day^2 (Gaussian k^2)
DEG = np.pi / 180.0


# ---------- ephemeris (Earth heliocentric, equatorial AU / AU per day) ----------
@lru_cache(maxsize=4096)
def _earth_helio(mjd):
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    from astropy.time import Time
    solar_system_ephemeris.set("builtin")
    t = Time(mjd, format="mjd", scale="tdb")
    ep, ev = get_body_barycentric_posvel("earth", t)
    sp, sv = get_body_barycentric_posvel("sun", t)
    R = np.array([(ep.x - sp.x).to("AU").value, (ep.y - sp.y).to("AU").value, (ep.z - sp.z).to("AU").value])
    V = np.array([(ev.x - sv.x).to("AU/day").value, (ev.y - sv.y).to("AU/day").value, (ev.z - sv.z).to("AU/day").value])
    return R, V


def _los_basis(ra_deg, dec_deg):
    """Return (e, e_a, e_d): LOS unit vector and the on-sky tangent unit vectors for the RA*cos(dec)
    and Dec directions (equatorial)."""
    a, d = ra_deg * DEG, dec_deg * DEG
    ca, sa, cd, sd = np.cos(a), np.sin(a), np.cos(d), np.sin(d)
    e = np.array([cd * ca, cd * sa, sd])
    e_a = np.array([-sa, ca, 0.0])               # +RA*cos(dec) direction (unit)
    e_d = np.array([-sd * ca, -sd * sa, cd])     # +Dec direction (unit)
    return e, e_a, e_d


def _state_from_attr(ra, dec, vx, vy, mjd, rho, rhodot):
    """Full heliocentric (r,v) [AU, AU/day] from epoch-1 angular state + (rho, rhodot).
    vx, vy = on-sky angular rates (deg/day): vx along RA*cos(dec), vy along Dec."""
    e, e_a, e_d = _los_basis(ra, dec)
    edot = (vx * DEG) * e_a + (vy * DEG) * e_d    # d(e)/dt, rad/day -> AU-frame unit/day
    R, V = _earth_helio(mjd)
    r = R + rho * e
    v = V + rhodot * e + rho * edot
    return r, v


def _angular_state(r, v, mjd):
    """Inverse: geocentric (ra,dec,vx,vy) [deg, deg/day] from heliocentric (r,v) at mjd."""
    R, V = _earth_helio(mjd)
    rg = r - R; vg = v - V
    rho = np.linalg.norm(rg)
    e = rg / rho
    dec = np.degrees(np.arcsin(np.clip(e[2], -1, 1)))
    ra = np.degrees(np.arctan2(e[1], e[0])) % 360.0
    _, e_a, e_d = _los_basis(ra, dec)
    edot = (vg - np.dot(vg, e) * e) / rho        # transverse angular rate, rad/day
    vx = np.dot(edot, e_a) / DEG
    vy = np.dot(edot, e_d) / DEG
    return ra, dec, vx, vy, rho


def _kepler_uv(r0, v0, dt, mu=MU, tol=1e-11, itmax=80):
    """Universal-variable two-body propagation (handles ell/par/hyp). Returns (r,v) at t0+dt."""
    r0n = np.linalg.norm(r0); v0n = np.linalg.norm(v0)
    rv = np.dot(r0, v0)
    alpha = 2.0 / r0n - v0n * v0n / mu            # = 1/a
    sqmu = np.sqrt(mu)
    # initial guess for universal anomaly chi
    if abs(alpha) > 1e-12:
        chi = sqmu * abs(alpha) * dt
    else:
        chi = sqmu * dt / r0n
    for _ in range(itmax):
        psi = chi * chi * alpha
        c2, c3 = _stumpff(psi)
        r = chi * chi * c2 + (rv / sqmu) * chi * (1 - psi * c3) + r0n * (1 - psi * c2)
        chi_new = chi + (sqmu * dt - chi**3 * c3 - (rv / sqmu) * chi**2 * c2 - r0n * chi * (1 - psi * c3)) / r
        if abs(chi_new - chi) < tol:
            chi = chi_new; break
        chi = chi_new
    psi = chi * chi * alpha
    c2, c3 = _stumpff(psi)
    f = 1 - chi * chi / r0n * c2
    g = dt - chi**3 / sqmu * c3
    rvec = f * r0 + g * v0
    rn = np.linalg.norm(rvec)
    fdot = sqmu / (rn * r0n) * chi * (psi * c3 - 1)
    gdot = 1 - chi * chi / rn * c2
    vvec = fdot * r0 + gdot * v0
    return rvec, vvec


def _stumpff(psi):
    psi = float(np.clip(psi, -400.0, 400.0))   # avoid cosh/sinh overflow on absurd trial orbits
    if psi > 1e-6:
        s = np.sqrt(psi); c2 = (1 - np.cos(s)) / psi; c3 = (s - np.sin(s)) / (psi * s)
    elif psi < -1e-6:
        s = np.sqrt(-psi); c2 = (np.cosh(s) - 1) / (-psi); c3 = (np.sinh(s) - s) / (-psi * s)
    else:
        c2 = 0.5; c3 = 1.0 / 6.0
    return c2, c3


def lambert_uv(r1, r2, dt, mu=MU, prograde=True, itmax=80, tol=1e-9):
    """Universal-variable Lambert solver (Bate-Mueller-White / Vallado). Given two heliocentric
    position vectors and the time of flight, return the velocities (v1, v2) of the connecting
    two-body orbit. Raises ValueError on non-convergence / degenerate geometry."""
    r1n = np.linalg.norm(r1); r2n = np.linalg.norm(r2)
    cosdnu = np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0)
    cross = np.cross(r1, r2)
    sgn = 1.0 if ((cross[2] >= 0) == prograde) else -1.0
    A = sgn * np.sqrt(r1n * r2n * (1.0 + cosdnu))
    if abs(A) < 1e-12:
        raise ValueError("degenerate transfer angle")
    psi = 0.0; c2 = 0.5; c3 = 1.0 / 6.0
    psi_up = 4.0 * np.pi**2; psi_lo = -4.0 * np.pi
    sqmu = np.sqrt(mu); y = r1n + r2n
    for _ in range(itmax):
        y = r1n + r2n + A * (psi * c3 - 1.0) / np.sqrt(c2)
        if A > 0 and y < 0:                      # raise psi_lo until y>=0
            psi_lo += 0.1
            psi = 0.5 * (psi_up + psi_lo)
            c2, c3 = _stumpff(psi)
            continue
        chi = np.sqrt(max(y / c2, 0.0))
        dt_calc = (chi**3 * c3 + A * np.sqrt(y)) / sqmu
        if abs(dt_calc - dt) < tol * max(dt, 1.0):
            break
        if dt_calc <= dt:
            psi_lo = psi
        else:
            psi_up = psi
        psi = 0.5 * (psi_up + psi_lo)
        c2, c3 = _stumpff(psi)
    else:
        raise ValueError("Lambert did not converge")
    f = 1.0 - y / r1n
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2n
    if abs(g) < 1e-15:
        raise ValueError("singular g")
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


def _elements(r, v, mu=MU):
    rn = np.linalg.norm(r); vn = np.linalg.norm(v)
    energy = vn * vn / 2 - mu / rn
    a = -mu / (2 * energy) if abs(energy) > 1e-30 else np.inf
    evec = ((vn * vn - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(evec)
    return a, e, energy


def fit_orbit(t1, ra1, dec1, vx1, vy1, t2, ra2, dec2, vx2, vy2, rate_frac=0.35):
    """Method of Herget: anchor on the two PRECISE angular positions, fit the topocentric ranges
    (rho1, rho2) so the two-body (Lambert) orbit threading both heliocentric positions reproduces the
    TRAIL-measured angular velocities at both epochs. Positions are matched exactly (anchored); the
    discriminator is the trail-velocity residual + the orbit being bound and physically plausible.
    Returns dict: rate_resid (deg/day, RMS over the 4 velocity components), a (AU), e, bound, rho1,
    rho2, cost."""
    e1, _, _ = _los_basis(ra1, dec1)
    e2, _, _ = _los_basis(ra2, dec2)
    R1, V1 = _earth_helio(t1); R2, V2 = _earth_helio(t2)
    dt = t2 - t1
    speed = max(np.hypot(vx2, vy2), np.hypot(vx1, vy1), 0.1)
    rate_sig = max(rate_frac * speed, 0.05)             # deg/day

    def rate_resid_at(rho):
        # short-arc (Vaisala) assumption: topocentric distance ~constant over a same-night arc, so
        # rho1 == rho2 == rho. Anchor on the two positions, Lambert for the orbit, residual = mismatch
        # of the orbit's predicted apparent angular velocities vs the two TRAIL velocities.
        r1 = R1 + rho * e1; r2 = R2 + rho * e2
        try:
            v1, v2 = lambert_uv(r1, r2, dt)
            _, _, vx1p, vy1p, _ = _angular_state(r1, v1, t1)
            _, _, vx2p, vy2p, _ = _angular_state(r2, v2, t2)
        except Exception:
            return np.inf, None
        out = np.array([vx1p - vx1, vy1p - vy1, vx2p - vx2, vy2p - vy2])
        if not np.all(np.isfinite(out)):
            return np.inf, None
        return np.sqrt(np.mean(out**2)), (r1, v1)

    # robust 1-D scan over rho (log-spaced), then refine around the best node
    grid = np.geomspace(0.02, 60.0, 80)
    res = [rate_resid_at(r)[0] for r in grid]
    j = int(np.argmin(res))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    fine = np.geomspace(lo, hi, 40)
    best_rho, best_r, best_state = grid[j], res[j], rate_resid_at(grid[j])[1]
    for r in fine:
        rr, st = rate_resid_at(r)
        if rr < best_r:
            best_r, best_rho, best_state = rr, r, st
    if best_state is None:
        return dict(cost=np.inf, rate_resid=np.inf, a=np.nan, e=np.nan, bound=False, rho1=np.nan, rho2=np.nan)
    a, e, energy = _elements(*best_state)
    if not (np.isfinite(a) and np.isfinite(e) and np.isfinite(energy)):   # numerical failure -> not a valid orbit
        return dict(cost=np.inf, rate_resid=np.inf, a=np.nan, e=np.nan, bound=False, rho1=np.nan, rho2=np.nan)
    return dict(cost=float(best_r / rate_sig), rate_resid=float(best_r), a=float(a), e=float(e),
                bound=bool(energy < 0), rho1=float(best_rho), rho2=float(best_rho))


def orbit_ok(track, *, exptime_s=30.0, pos_tol_arcsec=2.0, rate_frac_tol=0.5,
             a_min=0.3, a_max=200.0, q_min=0.05, rho_min=0.02, rho_max=120.0):
    """Apply the bound-orbit test to a 2-epoch track (DataFrame slice with ra,dec,ra0..dec1,mjd).
    Uses the two extreme-time detections. Returns (ok, info_dict)."""
    SOLARDAY = 86400.0
    g = track.sort_values("mjd")
    a_row, b_row = g.iloc[0], g.iloc[-1]
    dt = exptime_s / SOLARDAY

    def trail_v(r):
        cosd = np.cos(np.radians(r.dec))
        return (r.ra1 - r.ra0) * cosd / dt, (r.dec1 - r.dec0) / dt

    vx1, vy1 = trail_v(a_row); vx2, vy2 = trail_v(b_row)
    # the trail endpoints have a 180-deg ambiguity (don't know the sense of motion); orient each trail
    # velocity to point along the inter-epoch motion (a->b), which fixes the sign.
    mdt = (b_row.mjd - a_row.mjd)
    mx = (b_row.ra - a_row.ra) * np.cos(np.radians(b_row.dec)) / mdt
    my = (b_row.dec - a_row.dec) / mdt
    if vx1 * mx + vy1 * my < 0:
        vx1, vy1 = -vx1, -vy1
    if vx2 * mx + vy2 * my < 0:
        vx2, vy2 = -vx2, -vy2
    f = fit_orbit(a_row.mjd, a_row.ra, a_row.dec, vx1, vy1,
                  b_row.mjd, b_row.ra, b_row.dec, vx2, vy2)
    speed = max(np.hypot(vx2, vy2), np.hypot(vx1, vy1))
    q = f["a"] * (1 - f["e"]) if f.get("bound") else np.nan
    ok = (f.get("bound", False)
          and f["rate_resid"] < rate_frac_tol * max(speed, 0.1)
          and a_min < f["a"] < a_max and (np.isnan(q) or q > q_min)
          and rho_min < f["rho1"] < rho_max and rho_min < f["rho2"] < rho_max)
    f["ok"] = bool(ok)
    return bool(ok), f
