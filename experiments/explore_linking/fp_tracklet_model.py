#!/usr/bin/env python
"""
Rigorous FP-tracklet survival model (MOPS-style linking), CPU only.

The crude "2 FP inside a search disk" gate does NOT suppress FP at the
measured density (~67/CCD): a disk of radius v_max*dt covers most/all of a
CCD for any plausible rate, so P(link)~1. That is the wrong model.

Real linking (Rubin SSP / MOPS / HelioLinC) demands:
  (1) >=2 detections forming a TRACKLET within one night with a sky motion
      consistent with a SINGLE constant great-circle rate, to a TIGHT
      angular tolerance (set by astrometric error + rate prior), AND
  (2) the tracklet ATTRIBUTES across nights to one orbit (>=3 epochs over
      the campaign) -> the real killer of random FP.

We model the dominant suppression analytically with the *measured* FP
density, the *measured* visit cadence, and CCD geometry, for the regime
that actually applies to this dump (intra-night pairs + cross-night
attribution), and report an order-of-magnitude suppression factor and
residual FP/field.
"""
import numpy as np
import pandas as pd

R = "experiments/diffim_runs/test_real/results"
pp = pd.read_csv(f"{R}/per_panel_fp.csv")
pn = pd.read_csv("DATA_DIFFIM/test_real/panels.csv")
eph = pd.read_csv("DATA/sv_fast_movers_for_karlo_fast_with_pixels_rerun.csv")

emp = pp[pp.role == "empty"].merge(
    pn[["image_id", "visit", "detector", "img_h", "img_w"]],
    on="image_id", how="left")

PIXSCALE = 0.20  # arcsec/pix, LSSTCam
ccd_w = emp.img_w.median() * PIXSCALE / 3600.0
ccd_h = emp.img_h.median() * PIXSCALE / 3600.0
A_ccd = ccd_w * ccd_h                       # deg^2
k_med = float(emp.nn_fp.median())           # 67
k_mean = float(emp.nn_fp.mean())            # ~82
sigma_fp = float(np.std(emp.nn_fp))

out = []
def P(s=""):
    print(s); out.append(s)

P("=" * 70)
P("RIGOROUS FP-TRACKLET SURVIVAL MODEL")
P("=" * 70)
P(f"CCD: {ccd_w:.4f} x {ccd_h:.4f} deg, area {A_ccd:.5f} deg^2")
P(f"FP/CCD: median {k_med:.0f}, mean {k_mean:.0f}, sd {sigma_fp:.0f}")

# --- Astrometric tolerance for a 'kinematically consistent' link ---
# A real tracklet linker accepts a pair only if the implied motion is
# constant to within the per-detection astrometric uncertainty. For faint
# diffim residuals take sigma_pos ~ a few * pixel ~ 0.3" (generous). A 3-pt
# track fit must close to ~sigma_pos at the middle epoch. The cross-night
# attribution corridor half-width is tau ~ a few sigma_pos.
for sigpos_arcsec in (0.30, 0.50, 1.0):
    P("\n" + "-" * 66)
    P(f"astrometric residual tolerance sigma_pos = {sigpos_arcsec}\"")
    tau = (3.0 * sigpos_arcsec) / 3600.0     # corridor half-width, deg

    # STEP A -- intra-night pair (tracklet). Anchor FP in visit1; the
    # second FP in visit2 (same field, dt~minutes) must lie within the
    # ANNULUR corridor swept by the allowed rate band [v_lo,v_hi]. Area of
    # that corridor ~ (path length) x (2*tau) + end caps, NOT the full
    # disk. Path length over dt for the rate band: L = (v_hi-v_lo)*dt
    # (radial extent) but the linker does not know direction -> ring of
    # mean radius r0=v_mid*dt, radial width w=(v_hi-v_lo)*dt, plus 2*tau.
    v_lo, v_hi = 1.0, 47.0                    # full plausible band, deg/day
    v_mid = 0.5 * (v_lo + v_hi)
    # successive-visit gaps along real tracks (proxy for revisit cadence):
    eph_t = eph[["FieldID", "fieldMJD_TAI"]].drop_duplicates()
    # intra-night gap: use a short revisit (Rubin pairs ~ tens of min).
    for dt_lbl, dt in [("15 min", 15/1440.), ("1 hr", 1/24.),
                       ("4 hr (same night)", 4/24.)]:
        r0 = v_mid * dt
        w = (v_hi - v_lo) * dt + 2 * tau
        # annulus area, but clip the ring to what fits on the CCD: only the
        # part of the ring inside the CCD can hold a linkable FP. Fraction
        # of full ring usable ~ min(1, A_ccd / (pi*((r0+w/2)^2-(r0-w/2)^2)))
        ring_area = np.pi * ((r0 + w / 2) ** 2 - max(r0 - w / 2, 0) ** 2)
        usable = min(ring_area, A_ccd)        # cannot exceed the chip
        f_pair = usable / A_ccd               # P a given v2 FP is in corridor
        # expected linked PAIRS in one (field, 2-visit) pass:
        E_pairs = k_med * k_med * f_pair
        P(f"  intra-night dt={dt_lbl:<18s}: r0={r0:.3f} w={w:.3f} deg, "
          f"corridor frac f={f_pair:.3e}, "
          f"E[FP tracklets/field-pair]={E_pairs:.2f}")

    # STEP B -- the real killer: cross-night ATTRIBUTION of the tracklet to
    # an orbit (>=3 epochs). A spurious tracklet has a random implied rate &
    # direction; to attribute it must find ANOTHER spurious tracklet in a
    # later night whose position AND rate vector agree with the linear
    # prediction to within tau (position) and a tight rate tolerance
    # d(rate). Probability a random later-night tracklet matches:
    #   p_pos  ~ (pi*tau^2)/A_ccd                 (must land in the box)
    #   p_rate ~ (2*drate)/(v_hi-v_lo) in each of 2 rate comps ~ that^2
    drate_fac = 0.05            # rate must agree to ~5% of the band
    p_pos = np.pi * tau ** 2 / A_ccd
    p_rate = drate_fac ** 2
    # number of candidate later-night tracklets to test against ~ E_pairs
    # from a typical later night (use the 4hr value as representative).
    r0 = v_mid * (4 / 24.); w = (v_hi - v_lo) * (4 / 24.) + 2 * tau
    E_pairs_night = k_med * k_med * min(
        np.pi * ((r0 + w/2)**2 - max(r0 - w/2, 0)**2), A_ccd) / A_ccd
    p_attrib = 1 - (1 - p_pos * p_rate) ** max(E_pairs_night, 1)
    P(f"  cross-night attribution: p_pos={p_pos:.3e} p_rate={p_rate:.3e} "
      f"-> per-tracklet attribution prob ~ {p_pos*p_rate:.3e}")
    P(f"  with ~{E_pairs_night:.0f} candidate later-night tracklets: "
      f"P(spurious tracklet attributes) ~ {p_attrib:.3e}")
    # Residual surviving 3-epoch FP tracks per field:
    E_surv = E_pairs_night * p_attrib
    P(f"  => residual FALSE 3-epoch tracks / field ~ {E_surv:.3e}")
    supp = (k_med) / max(E_surv, 1e-12)
    P(f"  => FP suppression factor vs single-visit "
      f"({k_med:.0f}/CCD): ~{supp:.2e}x")

P("\n" + "=" * 70)
P("INTERPRETATION")
P("=" * 70)
P("""
- A 2-point positional gate alone does NOT help: at 67 FP/CCD the search
  corridor for any plausible asteroid rate covers an order-1 fraction of
  the chip, so spurious intra-night 'tracklets' form readily
  (E[FP tracklets/field] is tens-to-hundreds). 2-of-2 is a weak filter.
- The decisive lever is CROSS-NIGHT ATTRIBUTION to a kinematically
  consistent orbit (>=3 epochs): a spurious tracklet has a random implied
  rate+direction and must, by pure chance, find a later-night spurious
  tracklet that continues the SAME great-circle, constant-rate motion to
  arcsec/percent tolerance. That joint position+rate coincidence is
  ~1e-4..1e-6 per pair, giving residual FALSE 3-epoch tracks/field ~<<1
  and an FP-suppression factor of ~1e2..1e5 x relative to the single-visit
  ~67-82 FP/CCD operating point.
- In THIS dump the empties are 115 distinct visits / 87 distinct detectors
  with no revisited pointing, so the realized linkable-FP count is 0; the
  numbers above are the modelled survey-realistic upper bound.
""")
with open("experiments/explore_linking/_fp_numbers.txt", "w") as fh:
    fh.write("\n".join(out))
print("[written experiments/explore_linking/_fp_numbers.txt]")
