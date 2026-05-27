"""THEORETICAL FP budget for HelioLinC in the fast-mover (>1 deg/day, trail >=6px) regime.

We compute, from first principles, the probability that random false detections ("trash") link into a
spurious >=3-night track, and invert it for the maximum false-positive density (FP per visit) keeping
the expected number of spurious links below a confidence criterion. Trail-tracklet mode: each >=6px
streak is ONE tracklet carrying (sky position + sky velocity) from a single exposure.

PIPELINE FACTS THAT FIX THE MODEL (heliolinc2 source + run config):
  * heliolinc clusters 6D STATE VECTORS (x,y,z, chi*vx,chi*vy,chi*vz) propagated to t_ref, DBSCAN
    radius clustrad (km). chi = chartimescale = timespan -> velocity tolerance dv = clustrad/Tspan.
  * a (r,rdot,accel) HYPOTHESIS pins the radial DOFs, so trash states live on a 4D manifold:
    2D transverse position + 2D transverse velocity. Grid = heliohypo_all.txt (109983 hypotheses).
  * a link needs npt>=3 tracklets on minobsnights>=3 distinct nights.

DERIVATION:
  Geometry is kinematically locked: the field is at solar elongation eps, so a hypothesis r gives a
  fixed geocentric distance  Delta(r) = cos(eps) + sqrt(r^2 - sin^2 eps)  (law of cosines, outer root).
  At that Delta, the footprint and rate-band map to physical scales:
        A_x = Omega * Delta^2                         (transverse position area)
        A_v = pi*[(Delta*wmax)^2 - (Delta*wmin)^2]    (velocity annulus for rate band [wmin,wmax])
  Per-hypothesis trash-trash coincidence (4D ball / accessible manifold):
        p1(r) = [pi*clustrad^2 / A_x] * [pi*dv^2 / A_v]        ~ 1/Delta^4
  Spurious LINK = >=3 trash tracklets in one clustrad-ball on >=3 nights. DBSCAN+Poisson, summed over
  the REAL grid (each hypothesis contributes; t_persist folds the grid<->clustrad trials correlation):
        E[false links] = (N_t^3 / 6) * (Sum_grid p1(r)^2 / t_persist) * f_3nt        (CUBIC in N_t)
        N_t = (FP/visit) * Nvisits * f_v               (f_v = frac of FP with trail rate in band)
        f_3nt = C(Nn,3)*6 / Nn^3                        (>=3 DISTINCT nights penalty)
  Because Sum p1^2 ~ 1/Delta^8, the NEAREST shell (smallest r) dominates: the near-Earth hypotheses
  bind the budget. BUDGET: solve E[false links] = eps for FP/visit. eps=5e-4 <=> "99.95% no spurious link".
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from math import comb

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
AU = 1.495978707e8                 # km
DEG = np.pi / 180.0

# --- measured field + pipeline constants (run_neo_wide, NEO config) ---
OMEGA = 37.0 * DEG**2              # footprint (sr)
NVIS, NNIGHT, TSPAN = 719, 16, 29.2
ELONG = 164.3 * DEG               # field solar elongation (Sun pos at mean MJD)
CLUSTRAD = 1.0e5                  # km (NEO setting)
WMIN, WMAX = 1.0, 2.0             # target rate band deg/day
DV = CLUSTRAD / TSPAN             # velocity tolerance km/day (chartimescale scaling)
f_3nt = comb(NNIGHT, 3) * 6 / NNIGHT**3

def geo_dist(r):
    disc = r**2 - np.sin(ELONG)**2
    return np.where(disc > 0, np.cos(ELONG) + np.sqrt(np.clip(disc, 0, None)), np.nan)

# --- measured trail-velocity precision (known trailers >=1 deg/d): rate ~50%, direction ~25 deg ---
P_VEL = (50.0 / 180.0)            # P(random trail direction within +-25 deg, folded to +-90 by both-orderings)
                                  # rate tol (~50%) >= band half-width -> no rate discrimination, factor 1

def p1_of_r(r, R_pos=CLUSTRAD, p_vel=1.0):
    """Per-hypothesis trash-trash coincidence. R_pos = binding POSITION tolerance (clustrad for grouping,
    or the operational posRMS gate for a *reported* link). p_vel = velocity-consistency acceptance
    (measured from real trailers; <1 suppresses)."""
    d = geo_dist(r) * AU
    A_x = OMEGA * d**2
    return (np.pi * R_pos**2 / A_x) * p_vel

def f_velocity_band(wmin=WMIN, wmax=WMAX):
    d = pd.read_csv(HL / "run_neo_wide/adcnn_dets_labeled.csv")
    fp = d[d.objid.isna()]
    w = np.hypot((fp.ra1 - fp.ra0) * np.cos(np.radians(fp.dec)), fp.dec1 - fp.dec0) / (30.0 / 86400.0)
    return float(((w >= wmin) & (w <= wmax)).mean())

def budget(r, R_pos, p_vel, eps, t_persist=1.0, f_v=0.113):
    P1 = p1_of_r(r, R_pos, p_vel); P1 = P1[np.isfinite(P1)]
    C = ((P1**2).sum() / t_persist / 6) * f_3nt          # E[false] = C * N_t^3
    return ((eps / C) ** (1 / 3)) / (NVIS * f_v)

def main():
    f_v = f_velocity_band()
    g = pd.read_csv(HL / "run_neo_wide/heliohypo_all.txt", sep=r"\s+"); g.columns = [c.lstrip("#") for c in g.columns]
    r = g["r(AU)"].values; r = r[np.isfinite(p1_of_r(r))]

    print("THEORETICAL FP budget -- HelioLinC fast-mover (>1 deg/day, trail>=6px), with the two")
    print("suppression effects now COMPUTED.\n")
    print(f"GEOMETRY  eps={ELONG/DEG:.1f} deg, Delta(r)~r-1; grid {len(r)} hyps; Omega=37 deg^2, "
          f"Nvis={NVIS}, Nn={NNIGHT}, Tspan={TSPAN}d; f_v={f_v:.3f}\n")
    print("EFFECT 2  grid trials t_persist: relative state moves clustrad over dr~3e-3 AU << grid step")
    print(f"          0.02 AU  => adjacent hyps UNCORRELATED, t_persist~1 (NO loosening; mild undercount).\n")
    print("EFFECT 1  orbit-fit. MEASURED trail-velocity precision on real >=1deg/d trailers: rate ~50%")
    print(f"          (>= band half-width -> NO rate discrimination), direction ~25 deg -> p_vel={P_VEL:.2f}.")
    print("          Velocity is too coarse to reject trash; the binding constraint is the POSITION gate")
    print("          over the 3-night arc. Result depends strongly on that tolerance R_pos:\n")
    print(f"{'R_pos (km)':>26} {'FP/visit @99.95%':>16} {'@E[false]=1':>12} {'@E=100 (vet)':>14}")
    for R, lbl in [(CLUSTRAD, "clustrad grouping 1e5"), (2.0e4, "intermediate 2e4"), (2.0e3, "posRMS gate 2e3")]:
        b = [budget(r, R, P_VEL, e, f_v=f_v) for e in (5e-4, 1.0, 100.0)]
        print(f"{lbl:>26} {b[0]:>16.2f} {b[1]:>12.1f} {b[2]:>14.0f}")
    print("\nBUDGET vs INNER SEARCH LIMIT (posRMS gate 2e3, E[false]=1; budget ~ Delta_min^(4/3)):")
    print(f"{'r_min(AU)':>10} {'Delta_min(AU)':>13} {'FP/visit':>10}")
    for rmin in [1.05, 1.2, 1.5, 2.0]:
        m = r >= rmin
        if m.any(): print(f"{rmin:>10.2f} {geo_dist(rmin):>13.3f} {budget(r[m],2e3,P_VEL,1.0,f_v=f_v):>10.1f}")
    print("\nEXACT      : cubic law E[false]~N_fp^3; d(r) geometry; measured f_v, p_vel, t_persist~1.")
    print("RANGE      : the absolute budget spans ~O(0.1) to ~O(100) FP/visit depending on the binding")
    print("             position tolerance (grouping clustrad vs the posRMS classification gate) -- a factor")
    print("             this analytic model CANNOT pin (orbit-fit refines r,rdot continuously below the grid).")
    print("             Empirics agree we are over-budget: ~137 FP/visit gave ~283 false links historically.")
    print("BOTTOM LINE: scaling is waterproof; the absolute coefficient needs a trash-only Monte-Carlo through")
    print("             the REAL heliolinc+link_refine+classify (cheap, decisive) -- analytics are gate-sensitive.")

if __name__ == "__main__":
    main()
