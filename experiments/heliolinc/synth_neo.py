"""Generate synthetic >=1 deg/day NEO sky-tracks (real 2-body orbits, so HelioLinC's grid can link them)
on an LSST-style cadence, for the completeness/purity FP-budget experiment. NEOs are placed along the
field line of sight at a sampled geocentric distance, given a bound heliocentric velocity, RK4-propagated
in the heliocentric ECLIPTIC frame (matching Earth1day2020s ephemeris), and projected to RA/Dec. Returns
an ephemeris DataFrame [ObjID, mjd, ra, dec] for objects whose median apparent rate >= rate_min.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trail_tracklets import parse_earth_vectors

GM = 2.9591220828e-4          # AU^3/day^2 (heliocentric)
AU_KM = 1.495978707e8
EPS = np.radians(23.439291)   # obliquity (ecliptic <-> equatorial)
Rx = np.array([[1,0,0],[0,np.cos(EPS),-np.sin(EPS)],[0,np.sin(EPS),np.cos(EPS)]])  # ecl->eq

def _earth(earth_file):
    mjd, xyz = parse_earth_vectors(earth_file)
    return mjd, xyz / AU_KM   # AU, ecliptic heliocentric

def _eq_to_ecl_dir(ra_deg, dec_deg):
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    v_eq = np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])
    return Rx.T @ v_eq        # equatorial -> ecliptic unit vector


def _deriv(s):
    r = s[:3]; return np.concatenate([s[3:], -GM*r/np.dot(r, r)**1.5])

def _prop(state, dt, hmax=0.1):
    """RK4-propagate state by dt with step <= hmax (2-body is smooth -> coarse step is accurate)."""
    if dt == 0: return state.copy()
    n = max(1, int(np.ceil(abs(dt)/hmax))); h = dt/n
    for _ in range(n):
        k1 = _deriv(state); k2 = _deriv(state+0.5*h*k1)
        k3 = _deriv(state+0.5*h*k2); k4 = _deriv(state+h*k3)
        state = state + h/6*(k1+2*k2+2*k3+k4)
    return state

def _radec(topo_eq):
    x, y, z = topo_eq; return np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arcsin(z/np.linalg.norm(topo_eq)))

def generate(epochs, n_target=200, ra0=305.0, dec0=-20.0, field_rad=3.0,
             d_min=0.08, d_max=0.5, rate_min=1.0, rate_max=2.0, exptime_s=30.0, seed=0, earth_file=None):
    """TRAIL dets [ObjID,mjd,ra,dec,ra0,dec0,ra1,dec1,mag,band] for ~n_target NEOs, apparent rate in
    [rate_min,rate_max] deg/d. State placed at epochs[0] along field LOS, propagated cumulatively;
    endpoints from instantaneous sky velocity (small finite-diff)."""
    rng = np.random.default_rng(seed)
    emjd, exyz = _earth(earth_file)
    epochs = np.sort(np.asarray(epochs)); half = (exptime_s/86400.0)/2.0
    Ei = lambda t: np.array([np.interp(t, emjd, exyz[:, k]) for k in range(3)])
    E0 = Ei(epochs[0])
    rows = []; oid = 0; tries = 0
    while oid < n_target and tries < n_target*120:
        tries += 1
        ra = ra0 + rng.uniform(-field_rad, field_rad); dec = dec0 + rng.uniform(-field_rad, field_rad)
        dirn = _eq_to_ecl_dir(ra, dec); Delta = rng.uniform(d_min, d_max)
        pos = E0 + Delta*dirn; r = np.linalg.norm(pos); vcirc = np.sqrt(GM/r)
        rand = rng.normal(size=3); perp = rand - rand.dot(pos)*pos/r**2; perp /= np.linalg.norm(perp)
        vel = vcirc * rng.uniform(0.7, 1.2) * (np.cos(aa:=rng.uniform(0, 0.4))*perp + np.sin(aa)*pos/r)
        state = np.concatenate([pos, vel])
        out = []; t_prev = epochs[0]; ok = True
        for t in epochs:
            state = _prop(state, t - t_prev); t_prev = t
            topo = Rx @ (state[:3] - Ei(t)); cra, cdec = _radec(topo)
            topo2 = Rx @ (state[:3] + state[3:]*half - Ei(t+half))   # +half-exposure position (linear in 15s)
            r1ra, r1dec = _radec(topo2)
            d_ra = ((r1ra - cra + 180.0) % 360.0) - 180.0   # signed half-trail in RA (0/360-safe)
            out.append((t, cra, cdec, (cra - d_ra) % 360.0, 2 * cdec - r1dec,
                        (cra + d_ra) % 360.0, r1dec))        # -half endpoint, center, +half endpoint
        out = np.array(out)
        dr = np.diff(out[:,1]); dr = (dr+180)%360-180; dd = np.diff(out[:,2]); dt_ = np.diff(epochs); m = dt_ > 0.01
        if not m.any(): continue
        rate = np.median(np.hypot(dr[m]*np.cos(np.radians(out[:-1,2][m])), dd[m])/dt_[m])
        if not (rate_min <= rate <= rate_max): continue
        for t, cra, cdec, ra0e, dec0e, ra1e, dec1e in out:
            rows.append(dict(ObjID=f"SYN{oid:04d}", mjd=t, ra=cra, dec=cdec,
                             ra0=ra0e, dec0=dec0e, ra1=ra1e, dec1=dec1e, mag=21.0, band="r"))
        oid += 1
    df = pd.DataFrame(rows)
    print(f"[synth_neo] generated {oid} NEOs (rate {rate_min}-{rate_max} deg/d) in {tries} tries -> {len(df)} trail-dets", flush=True)
    return df

if __name__ == "__main__":
    # quick self-test: LSST cadence epochs (4 nights, 2 visits/night, 15-day window)
    base = 60858.0
    nights = [base, base+3.5, base+8.0, base+14.0]
    epochs = np.array([n + dv for n in nights for dv in (0.0, 0.02)])  # pairs ~30 min
    import time; t0=time.time()
    df = generate(epochs, n_target=200, earth_file="NEO_large/Earth1day2020s_02a.txt")
    print(f"gen time {time.time()-t0:.1f}s | {df.ObjID.nunique()} NEOs, {len(df)} dets")
    # check trail length ~ rate*exptime (>=6px = >=1.25 arcsec at 1 deg/d)
    g=df.groupby("ObjID").first(); tl=np.hypot((g.ra1-g.ra0)*np.cos(np.radians(g.dec)),g.dec1-g.dec0)*3600
    print(f"trail length arcsec: med {tl.median():.1f} (>=1.25 = >=6px expected for >=1 deg/d)")
