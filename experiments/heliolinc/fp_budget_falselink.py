"""First-principles (statistical-physics) false-link rate for HelioLinC trail-tracklet linking, to be
CHECKED against the trash-MC (not fitted). Computes the FP-per-panel that gives a given probability
(e.g. 0.3% = '3 sigma') of linking rubbish->rubbish->rubbish.

PHYSICS
A raw cluster = 3 trash trail-tracklets on 3 distinct nights mutually consistent with ONE orbit within
the clustering radius. Three sky positions ALWAYS fit some orbit (Gauss), so positions don't penalize;
the binding constraint is that each trail's VELOCITY VECTOR (rate + direction) matches that orbit. The
geocentric distance is tied to the seed's sky rate by opposition kinematics  Delta ~ 0.99/omega [AU],
which removes the spurious small-Delta blow-up (a trail of rate omega can only belong to an orbit at the
Delta that produces that rate).

Per night-pair, a trash 'b' links to a seed 'a' (whose trail+orbit predicts b's position & velocity) iff
  - b's position lies on the predicted path within the angular clustering tol  theta_c = R/(Delta*AU)
  - b's trail RATE matches within      d_omega = (R/chi)/(Delta*AU)      [chi = chartimescale = Tspan]
  - b's trail DIRECTION matches within d_phi   = (R/chi)/(omega*Delta*AU) [x2 for head/tail orderings]
    P_link(omega) = [pi*theta_c^2/Omega] * [2*d_omega * f_omega(omega)] * [2*(2*d_phi)/(2*pi)]
E[raw clusters] = C(Nn,3) * (Ndet/Nn)^3 * <f_omega(omega) * P_link(omega)^2>_omega        (cubic in FP)
GATE: a *reported* false link must pass posRMS<2000 km. Tightening R: 1e5 -> 2000 scales every tol ∝ R,
so P_link ∝ R^4 and E ∝ R^8 -> E[gated]/E[raw] = (gate/clustrad)^(4..8) (position-only .. pos+vel).
BUDGET: solve E = p_target (e.g. 0.003) for FP/panel (cubic).
"""
from __future__ import annotations
import numpy as np
from math import comb
from pathlib import Path

HL = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc")
AU = 1.495978707e8; DEG = np.pi/180
OMEGA_DEG2 = 37.0                     # linking footprint (deg^2); false links ~ linear in this
OMEGA = OMEGA_DEG2 * DEG**2           # sr
PANEL_AREA = 0.05                     # deg^2 per detector diffim (LSST CCD)
CLUSTRAD, GATE = 1.0e5, 2.0e3         # km: DBSCAN radius, posRMS gate

# --- CADENCE SETS ---  (false-link combinatorics depend on nights & window)
# run_neo_wide = the targeted field the MC ran on (for validating the theory mechanism)
# LSST = baseline survey cadence: ~2 visits/night (pairs), ~3-6 nights per 15-day SSP discovery window
import sys
CAD = sys.argv[1] if len(sys.argv) > 1 else "LSST"
if CAD == "neo_wide":
    NN, TSPAN, PANELS_PER_NIGHT = 16, 29.2, 1301/16      # MC field (sparse targeted)
else:  # LSST baseline
    NN, TSPAN = 4, 15.0                                  # nights in window, window length (d)
    VIS_PER_NIGHT = 2                                    # LSST pairs
    PANELS_PER_NIGHT = VIS_PER_NIGHT * OMEGA_DEG2 / PANEL_AREA   # full focal-plane tiling of footprint
CHI = TSPAN                           # chartimescale (d) -> velocity tol dv = R/chi (km/d)

w = np.load("/tmp/trash_rate.npy")    # real trash sky-rate sample (deg/day)
w = w[(w > 0.2) & (w < 30)]           # physical asteroid-search band (drop spurious >30 deg/d)
edges = np.linspace(0.2, 30, 300); hist, _ = np.histogram(w, edges, density=True)
wc = 0.5*(edges[:-1]+edges[1:])
def f_omega(om):                      # trash rate PDF (per deg/day)
    return np.interp(om, wc, hist, left=0, right=0)

def Delta_of(om):                     # opposition kinematics: sky rate omega(deg/d) -> geocentric Delta(AU)
    return np.clip(0.99/om, 0.05, 5.5)

def P_link(om, R):
    d = Delta_of(om) * AU             # km
    dv = R / CHI                      # velocity tol (km/day)
    theta_c = R / d                   # rad
    d_omega = (dv / d) / DEG          # deg/day
    om_rad = om * DEG
    d_phi = np.minimum(dv / (om_rad * d), np.pi)   # rad
    P_pos = np.pi * theta_c**2 / OMEGA
    P_rate = 2*d_omega * f_omega(om)
    P_dir = 2 * (2*d_phi) / (2*np.pi)              # x2 head/tail orderings
    return P_pos * P_rate * np.minimum(P_dir, 1.0)

def vel_integral(R):
    """<f_omega * P_link^2> integrated over seed rate (per night-triple, per (det/night)^3)."""
    grid = np.linspace(0.3, 20, 2000)
    return np.trapezoid(f_omega(grid) * P_link(grid, R)**2, grid)

def E(fp_per_panel, R):
    det_per_night = fp_per_panel * PANELS_PER_NIGHT
    return comb(NN, 3) * det_per_night**3 * vel_integral(R)

def budget(p_target, R):
    K = comb(NN, 3) * PANELS_PER_NIGHT**3 * vel_integral(R)   # E = K * fp^3
    return (p_target / K)**(1/3)

print(__doc__.split("PHYSICS")[0])
print(f"CADENCE = {CAD}: Nn={NN} nights, Tspan={TSPAN}d, panels/night={PANELS_PER_NIGHT:.0f}, "
      f"footprint={OMEGA_DEG2} deg^2")
print(f"clustrad={CLUSTRAD:.0e} km, gate={GATE:.0e} km; velocity tol dv=R/chi: raw={CLUSTRAD/CHI:.0f}, gate={GATE/CHI:.0f} km/d\n")
if CAD == "neo_wide":
    print("CHECK vs MC (raw clusters; MC got 0,0,0,1,2 at 1,6,32,75,150):")
    print(f"{'FP/panel':>9} {'E[raw](theory)':>15} {'MC raw':>7}")
    mc = {1:0,6:0,32:0,75:1,150:2}
    for fp in [1,6,32,75,150]:
        print(f"{fp:>9} {E(fp,CLUSTRAD):>15.3g} {mc[fp]:>7}")
print(f"\ngate suppression E[gated]/E[raw] = (gate/clustrad)^k:  k=4 {(GATE/CLUSTRAD)**4:.1e} | k=8 {(GATE/CLUSTRAD)**8:.1e}")
print(f"\n=== CONCRETE FP/PANEL BUDGET ({CAD} cadence; cubic E ~ FP^3) ===")
print(f"{'p_target':>10} {'RAW cluster':>13} {'GATED k=4':>12} {'GATED k=8':>12}")
for p in [0.003, 0.01, 1.0]:
    braw = budget(p, CLUSTRAD)
    bg4 = (p/(comb(NN,3)*PANELS_PER_NIGHT**3*vel_integral(CLUSTRAD)*(GATE/CLUSTRAD)**4))**(1/3)
    bg8 = (p/(comb(NN,3)*PANELS_PER_NIGHT**3*vel_integral(CLUSTRAD)*(GATE/CLUSTRAD)**8))**(1/3)
    print(f"{p:>10.3f} {braw:>13.1f} {bg4:>12.0f} {bg8:>12.3g}")
print("\nRAW = 3 rubbish forming any cluster (MC-verifiable: MC counts n_clusters_raw);")
print("GATED = cluster passing posRMS<2000 = a REPORTED false link (the actual 'linking rubbish').")
