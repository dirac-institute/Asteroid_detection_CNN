# Same-night 2-visit vs 3-visit NEO linking — 3σ purity threshold

Measured on real off-ecliptic LSST difference images (48k genuine false positives, zero real
asteroids → every false link is unambiguous), with an injected fast-NEO population whose same-night
apparition counts follow the **operational** Rubin cadence (rubin_sim OpSim `baseline_v2.0_1yr.db`
propagated through Sorcha with Granvik-like NEO orbits — **not** DP2 commissioning visits). FP rate
λ_FP(S) measured by the null Monte Carlo in `calibrate_link_fpp.py` (per-visit rigid sky offsets
destroy real continuity → every surviving track is a chance link).

3σ one-sided false-alarm budget: **λ_FP ≤ 1.35×10⁻³ false tracks / field-night.**

## Cadence (operational, fast ≥1°/day NEOs)
WFD is **pair-dominated**: same-night observable k=1 44% · **k=2 38%** · k=3 9% · k≥4 8%
→ **≥2× : 56%**, **≥3× : 17%**. A same-night pipeline must work on PAIRS; the 3rd sighting is
rare by design (it comes on other nights → tracklet→track).

## λ_FP(S) — null MC, Δt≤30 min window
| score S | ρ (FP dets/field) | 2-visit λ_FP (orbit 0.25) | 3-visit λ_FP |
|--------:|------------------:|--------------------------:|-------------:|
| 0.80    | 4719              | 1.89                      | 0.003        |
| 0.85    | 1911              | 0.97                      | 0.005        |
| 0.90    | 608               | 0.58                      | 0.000        |
| 0.95    | 148               | 0.22                      | —            |
| budget  |                   | 1.35×10⁻³                 | 1.35×10⁻³    |

- 2-visit: **λ_FP ∝ ρ^1.15** (shallow). 3-visit: λ_FP ≈ 0 for all S≥0.78 (**ρ^3.5**, steep).
- Same score S=0.80, same density: **2v λ=1.9 vs 3v λ=0.003 (~600–1600×)** — the 3rd point is the
  purity engine; thresholding cannot substitute.

## The 3σ thresholds
Extrapolate the λ∝ρ fit to the budget → required FP density ρ\*, then map ρ\* → S\* via the measured
ρ(S) ≈ exp(25.0 − 20.8·S):

- **3-visit:  S\* ≈ 0.78**  — achievable, full recall. **(shipped discovery tier.)**
- **2-visit (per WFD pair — the operational unit, corrected):  S\* ≈ 0.99 — REACHABLE.** Measured per
  single 2-visit pair (not per 12-visit deep-field night, which inflated λ ~11×): λ/pair = 0.44 (S0.80),
  0.047 (0.90), 0.020 (0.95), 0.005 (0.97); λ∝ρ^1.40 → crosses 1.35×10⁻³ at S≈0.99. The gap at S=0.97 is
  only **~3.7×**, not 130×. **So the FP floor is not the blocker — 3σ purity on a pure pair IS reachable.**
  THE BINDING CONSTRAINT IS RECALL: recovered objects collapse (14→10→3→1 over S=0.80→0.90→0.95→0.97), so
  at S≈0.99 the stream is 3σ-pure but recovers ~0% (only the brightest trails score ≥0.99 on both members).
  The lever that now matters is ADCNN per-detection score on FAINT trails, not more FP cuts. (Collinearity,
  shipped, buys real score headroom: S=0.95 reaches 15× vs ~100× without it.)

## What ships for 2-visit
A purified **candidate / alert stream**, not a 3σ-confirmed tier. Best levers (defaults in
`trail_state_link`): `max_arc_2v_min=40` (Δt window, purity 0.28→0.71), `orbit_rate_tol=0.25`
(bound-orbit velocity-residual; FP pairs can't reproduce both trail velocities — the discriminator
is the residual, NOT the orbital elements, which the short arc leaves degenerate), and
`--score-2v-min ~0.90` (purity ≈0.85, λ≈0.5/field-night — clean on a single field but ~370× over 3σ).

Turning the 56% pair pool into **defensible** discoveries requires multi-night tracklet→track linking.

## Why no per-pair FP cut reaches 3σ (FP-cleaning study, 2026-06-02)
Tested every recall-safe per-pair lever. Best new one: **4-endpoint collinearity** (a pair = 4 trail
endpoints; a real mover's two trail segments lie on ONE line, perp-RMS ~0.08″ vs FP ~0.69″) → `perp<0.30″`
keeps ~100% of real pairs and cuts FP ~3×. Brightness/SNR consistency adds ~2× but costs recall (off by
default). LSST mask planes only ~1.3× (residual FP are *unmasked* subtraction noise). Orbital elements don't
separate (short-arc degeneracy).

**Decisive diagnosis — the FP-rate exponent falls as cuts stack:** λ_FP ∝ ρ^1.34 → 1.15 → 0.96 → **0.84**.
A ρ² population is random coincidences (removable by thinning); a ρ^~1 population is a near-fixed
**structured-artifact** set (unmasked subtraction-residual over-detections that are *locally* valid tracklets:
collinear, consistent velocity, bound orbit). Per-pair physics cannot remove them — and the only handle that
could (stationarity: a real NEO moves away, an artifact recurs at the same sky position) needs *other*
same-night visits, which operational WFD doesn't have (2 visits/night). So the information that certifies a
real heliocentric orbit vs a coincidental/structured alignment is **not in two same-night points** — it's in
the third (triplet, or a second night). Best recall-safe stack cuts FP ~5× at full recall, still ~130× over
3σ at S=0.95. **3σ purity on a pure same-night pair is not achievable by detection/linking cleverness.**
