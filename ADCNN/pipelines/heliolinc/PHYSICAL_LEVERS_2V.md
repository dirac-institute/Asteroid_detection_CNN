# Creative physical levers for faint-fast 2-visit purity — what was tested, what survives

Goal: clean FP so the score floor can be LOWERED → more completeness at fixed purity, for the
faint-fast bin (detection-SNR 2–10, rate>1 deg/day). Going beyond the already-exhausted geometric/
CNN/photometric grid, two genuinely-NEW physical levers were identified and tested, plus a third
dismissed on physics.

## Pivotal fact: the bin is CLOSE
Faint-fast movers sit at median topocentric ρ = **0.27 AU** (93% < 0.5 AU; Sorcha/Granvik). Over the
34-min WFD pair gap the diurnal **parallax is ~4″** (up to 11–22″ for the closest), i.e. ~60× the 0.06″
astrometric precision. The current orbit fit (`orbit_check.fit_orbit`) uses a **geocentric** observer and
its docstring dismisses parallax as "sub-arcsec" — wrong by ~2 orders of magnitude for this bin. So a
topocentric fit looked promising.

## Lever A — topocentric parallax fit (position + velocity): TESTED → DEAD
`/tmp/topo_parallax_test.py`: re-fit each pair with a full topocentric observer (Rubin
`EarthLocation.get_gcrs_posvel`, **position AND velocity**), paired against the geocentric fit on the SAME
pairs (28 TRUE, 1034 FALSE, 3 fields). Metric = Δresid = resid_geo − resid_topo.
- TRUE: median resid 0.475→0.473, **median Δ = +0.0002 deg/day**. FALSE: 3.933→3.932, Δ ≈ 0.
- FALSE-keep at fixed TRUE-keep: identical geo vs topo (0.001–0.005).

**Why:** the fit is Herget/Väisälä — it *anchors exactly on the two positions*, so the large *positional*
parallax is absorbed into the fitted ρ. The only residual channel is velocity; the observer-velocity term
is a near-constant ~0.05 deg/day offset that cancels between trail-vel and chord-vel (~0.25σ). Real
signal, wrong axis. Parallax buys **nothing** here.

## Lever B/C — orbit-physicality / photometric-H prior: TESTED → no orthogonal power
Fitted ρ, a, e, r_helio overlap heavily TRUE vs FALSE (ρ med 0.064 both; r_helio ~0.98 AU both — the
short-arc fit pins ~1 AU regardless; a/e have huge tails for both). Adding a NEO-like orbit cut
(ρ∈[0.03,2], a∈[0.5,4], e≤0.95) on top of the resid cut drops **TRUE-keep 0.90→0.11** while FALSE only
0.005→0.000 — it destroys completeness without removing FALSE the velocity-residual didn't already remove.
Consistent with the earlier ML ceiling (a,e,q were already in it and didn't help).

## Lever (dismissed) — cross-filter color
Off-ecliptic substrate fields are multi-filter, but operational WFD same-night pairs are **same-filter**
(no color), and a SNR 2–10 detection's color error is ~0.1–0.5 mag → overlaps FP. Not viable.

## Root cause (why every per-pair lever caps out)
With exactly 2 epochs the discriminating information is the **velocity residual** of the bound-orbit fit
(3 dof: 4 trail-velocity components − 1 fitted ρ). It already separates ~200–1000× on raw chord pairs
(TRUE resid 0.47 vs FALSE 3.93). Parallax, orbit-physicality, and color are each **redundant with it or
negligible**. The missing ~10–100× to reach discovery-grade purity is a 3rd-epoch / base-rate problem,
not a discriminator we have failed to find. (Confirms [[faint-fast-bin-only-scope]] / the base-rate wall.)

## Partner recovery (the completeness lever) — TESTED

**Ceiling (Part 1, `/tmp/forced_recall_ceiling.py`, 82 fields, 4547 recoverable):** if the CNN-missed
second sighting could be recovered cleanly, completeness ~doubles — at seed score≥0.95, forced thr 5σ,
**13.8% → 28.5% (×2.07)**; at seed≥0.80, **31% → 44%**. (Loose upper bound: `snr_target` is per-object-
constant, so it assumes the undetected partner is as bright as the detected one.)

**Catalog realization (Part 2, `/tmp/asym_sweep.py`, ASYMMETRIC thresholding — seed≥0.95, relax partner
to S_low, chi2≤5, rate[1,8]):** purity at the Granvik base rate (ρ=0.14/deg²/night):

| S_low | C% | FP/field | purity% |
|---|---|---|---|
| 0.95 | 7.1 | 0.13 | **6.9** |
| 0.90 | 10.3 | 0.46 | 3.0 |
| 0.80 | 11.3 | 1.12 | 1.4 |
| 0.70 | 11.5 | 2.90 | 0.55 |

**Relaxing the partner buys ~ZERO completeness at fixed purity** — every step drops purity below the
S_low=0.95 starting point (the (C,purity) frontier point along this family is S_low=0.95 itself). The lever
**fails on the exact ask** (more completeness at held purity). Crucially this IS the trail-localized search:
`pair_chi2`'s pre-gate already forces the partner into a ±20° wedge along the seed's trail (chi2≤5 → ~±10°),
so a "trail-predicted small box" is a strict subset with the same TP/FP ratio — no separate test needed.

**The one card the catalog cannot play:** pixel forced photometry BELOW the CNN detection floor with an
EXACT trail-matched filter (matched filter at known position AND known orientation — selectivity absent
from the catalog). Decisive number to measure (pre-registered): *of the ~1–2′-prediction-box artifacts,
what fraction survive an exact trail-template match (PA ±~5°, length ±~20%) at high matched-filter
significance?* MUST run BOTH arms (TP recovery on injected pixels + FP control at predicted positions from
FALSE seeds on un-injected pixels); scope 1–2 fields; win = adds ≥N pts completeness while FP-arm purity
≥7%. Prior (from [[fp-rejection-investigation]]/[[realbogus-fp-filter-limits]]): unmasked FP are
per-detection-inseparable subtraction noise → likely the same wall, but it is a genuine untested card.

**Pixel forced-photometry spike — TESTED → DEAD (`/tmp/forced_phot_spike.py`, fields 34/38/26, both arms):**
exact variance-weighted trail-matched filter at the seed-trail-PREDICTED position+PA, reaching below the
CNN floor. TP arm = re-inject the partner trail into the real diffim, MF over the prediction box (recovers
3/4 — works, but starved: at S≥0.95 only ~1/field has a single-seed undetected-partner pair). **FP arm (the
decisive number, 213 attempts): 71% of FP-seed prediction boxes yield a chance trail match at ≥5σ (67% at
6σ — flat ⇒ REAL bright elongated artifacts, not noise tail).** → forced phot manufactures ~50 false
links/field (×~390 the current 0.13/field at S≥0.95), and because the forced partner sits on the predicted
line by construction it PASSES the chi2 gate → **purity collapses ~7% → 0.08%.**

**Forced-phot v2 — added the gates the strawman omitted (advisor: brightness + tight direction)
(`/tmp/forced_phot_v2.py`, same 3 fields): STILL DEAD.** Thin search ALONG the predicted motion line
(±300 px along, ±40 px perp — not a square box), match at predicted PA **±3°**, and require partner flux
within **±1 mag** of the seed. Measured seed precision first: even score≥0.95 seeds have ~15% trail-speed
error → ±190 px along-track. Result: **FP false-recovery 169/213 = 79% (WORSE than 71%)**, TP recovery
fell to 31%. Purity 0.07%. The gates do not discriminate.

ROOT CAUSE (now doubly-confirmed, deeper): two pieces. (1) A single trailed seed localizes the partner only
to ±~190 px (trail-speed/PA error over the ~1300 px inter-visit arm) while FP-artifact spacing is ~100 px,
so any search region big enough to hold the real partner spans several artifacts. (2) **At faint flux the
gates carry no information**: the seed is mf_snr 2–10, so the brightness gate (±1 mag) *admits* the abundant
faint artifacts, and a faint short trail template (6–22 px) is not distinguishable from a faint elongated
subtraction residual. Forced photometry adds no information — it only measures flux where you predict, and
faint streak-like FP lie along essentially every predicted line. This is the base-rate/information wall made
concrete at the pixel level (confirms [[fp-rejection-investigation]], [[realbogus-fp-filter-limits]]).

## The FP-COUNT axis (λ∝N_FP²) — the one structural lever, TESTED (`/tmp/fp_count_lever.py`)
A false 2v link is a chord-consistent FP PAIR → λ∝N_FP², so ~50% purity needs only ~3.6× fewer FP
*detections* (not 13×). Decisive test = are the FP that FORM false links separable from true movers?
- **In aggregate, strongly yes** (AUC: score 0.97, mfsnr 0.94): link-forming FP pile at score 0.80–0.87 /
  mfsnr 2–5; true movers at score 0.88–0.98 / mfsnr 5–9. **But that separation IS the score≥0.95 op-point we
  already run** — 96.6% of link-forming FP are <0.90, only 0.6% reach ≥0.95.
- At the operating tail the ROC is **flat**: FP-keep/TP-keep ≈0.011 (0.95) → 0.004 (0.97), i.e. raising the
  floor only walks the same Pareto frontier (more purity, less completeness), no free FP reduction.
- So the detector already extracts ~all the score/mfsnr separation; the ≥0.95 survivors sit at the
  irreducible faint-flux limit (consistent with [[fp-rejecter-1b-result]] per-detection ceiling). The 3.6×
  FP-count cut is NOT reachable by thresholding. λ∝N_FP² unmeasurable here because false links are already
  ~0–1/field at score≥0.90 (FP already cut hard). The only untested action is a HARD-NEGATIVE FINE-TUNE of
  the stage-2 CNN on the score≥0.95 link-forming-FP cutouts — LOW prior (flat ≥0.95 ROC + prior rejecter
  plateau), but the last single-night card.

## FREE WIN — "promote 2v→3v when a 3rd same-night sighting exists" (tier, no purity cost)
**39% of recoverable faint-fast movers have ≥3 same-night sightings** in these deep-cadence fields (≈17% in
real pair-dominated WFD); 3-visit linking is PURE ([[two-vs-three-sighting-fast-neo]]). A tier that promotes
a 2v candidate to a confirmed 3v link when a consistent 3rd detection exists recovers that subset at purity
~1.0 at ZERO cost to the 2v alert tier. Cadence-limited (big in deep-drilling/ToO fields, small in WFD) but
free — the 3v pure-linking path already exists in the linker.

## Hard-negative CNN fine-tune — DECISIVE TEST (`/tmp/extract_cutouts.py`, `/tmp/train_rejecter.py`)
The last untested single-night card. Trained a field-grouped cutout classifier (channel-0 diffim/σ, tiny
CPU net) on 1084 cutouts (271 true movers vs 813 FP) in the 0.80–0.98 score band; asked if the pixels carry
FP/true-mover signal BEYOND the production CNN score.
- cutout-CNN OOF AUC **0.787** < production score **0.888**; **score+cutout combined 0.887 ≈ 0.888 → no
  operational gain**; FP-keep at fixed TP-keep not improved (combined slightly worse — weak/noisy signal).
- residual AUC within a fixed score bin: 0.68 ([0.85,0.90)), **0.80 ([0.90,0.95))** — a WEAK pixel signal
  beyond the score DOES exist, but it lives in 0.85–0.95, NOT at the ≥0.95 operating point (where
  link-forming FP are already ~0.6%), so it can't move operating-point purity.
- VERDICT: fine-tune **not justified** — no operational ROC gain over the existing score, and the residual
  doesn't reach the op-point. CAVEAT: this is channel-0-only + a tiny CPU net (a lower bound); a full
  3-channel GPU fine-tune (incl. the seg-model channels) could in principle exploit the weak 0.90–0.95
  residual for a MARGINAL gain — low prior (consistent with [[fp-rejecter-1b-result]] + the residual not
  reaching ≥0.95). Single-night is effectively closed; this card is the user's call, recommended against.

VERDICT (all cards played — DISCRIMINATION and FP-COUNT axes both exhausted): faint-fast same-night 2-visit
**cannot** reach discovery-grade purity by any tested means — geometry/chi2, CNN score, photometry/mfsnr,
GBM, topocentric parallax, orbit-physicality/H prior, color, restricted sky region, catalog partner-recovery,
pixel forced-photometry (gated), OR FP-count reduction by thresholding. The wall is the base rate + FP being
per-detection-inseparable faint real streaks. Single-night = ALERT/follow-up tier (alerts.jsonl) + the free
3v-promotion tier; **≥3 epochs / multi-night is required for standalone discovery.** Remaining untested
single-night card = targeted hard-negative CNN fine-tune (low prior).

Scripts: `/tmp/topo_parallax_test.py`, `/tmp/forced_recall_ceiling.py`, `/tmp/asym_sweep.py`, `/tmp/forced_phot_spike.py`.
