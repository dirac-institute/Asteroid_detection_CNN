# Can single-night 2-visit linking produce CONFIDENT, PURE NEO alerts?

**Investigation date:** 2026-07-02 · **Data:** real embargo nights 20260629 + 20260630 (prompt ApPipe, S3)
**Adversarially audited 2026-07-02** (self-audit ordered by the user: "be sceptical until proven wrong;
I want to be very economical on how much true alerts I lose"). Every number below is measured on real
data and survives the audit; §7 lists what the audit corrected. Artifacts: `run_embargo_0630/audit/`
(stageA–D CSVs), `run_embargo_0630/audit_pixel_vet.py`, `run_embargo_night/audit_0629_2v_revet.csv`,
null runs in `run_embargo_0630/null2v/`.

---

## 1. The question, decomposed

A "confident pure NEO alert" requires two independent things:

1. **NEO-ness** — if the pair is a real object, is it an NEO?
2. **Realness** — is the pair a real moving object at all (vs a false link)?

These have opposite answers, and conflating them is what made the problem look hopeless.

---

## 2. NEO-ness is nearly free (measured, with two audited caveats)

Monte Carlo: 3M bound non-NEO orbits (q > 1.3 AU; a ∈ [1.5, 4.5], e ≤ 0.6, i ~ Rayleigh 9°),
propagated to the observation epoch, selected in each field's (elongation, ecliptic-latitude) window:

| field | elong | β | non-NEOs in window | median rate | p99.9 | **max** |
|---|---|---|---|---|---|---|
| 20260630 swarm (RA 215.7, −17.1) | 120° | −2.7° | 27,746 | 0.04 °/d | 0.59 | **0.91 °/d** |
| 20260629 cand. (RA 327.0, −8.3) | 132° | +4.7° | 22,972 | 0.07 °/d | 0.53 | **0.90 °/d** |

Zero of ~51k simulated non-NEOs reach 1.0 °/day (max 0.91). Empirical confirmation: SkyBoT returns
344 catalogued asteroids matched to our own detections in the 345/396 field — **all** move < 0.13 °/day
(the field sits near the MBA stationary point). → *If a pair is real and its rate is securely ≥ 1 °/day,
it is an NEO* — with two honest caveats found in audit:

- **Rate error on short arcs.** At Δt = 49 s, endpoint astrometry (~0.3″) gives σ_rate ≈ 0.3 °/day —
  the gate must be `rate − 3σ_rate > 1.0` (effective ~1.9 °/day for 49-s arcs; negligible correction
  for ≥20-min arcs, σ_rate ≈ 0.006 °/day). All four real 49-s alerts (rates 4.5–7.4) pass with margin.
- **Jupiter-family comets near perihelion** (q ~ 1.3–1.5, e > 0.6 — outside the simulated box) can
  reach ≥1 °/day apparent. They are rare and usually active, and are themselves follow-up-worthy;
  "certain NEO" should read "NEO or (rare) JFC near perihelion".

## 3. The false-link background is STRUCTURED, not statistical noise (measured)

The 20260630 night produced 74 raw 2v alerts, 55 from one dense-ecliptic visit pair (345→396,
Δt 39 min). Decomposition:

**(a) Cross-field null runs** — re-link visit 345 against detections of 4 same-night far-away visits
translated onto 396's footprint (real links physically impossible, densities preserved):

| pair | alerts |
|---|---|
| real 345→396 (control, isolated re-run) | **55** |
| null donors (16.5k–32.7k dets) | 6, 19, 0, 15 → **λ_indep ≈ 10 ± 8** |

Independent-FP chance linking explains only ~10 of 55; the ~45 excess is same-sky correlated.
*(Audit caveat: only 4 donors, and donor panel footprints differ from the real visit's — λ_indep is
a factor-~2 estimate, not better. The artifact interpretation does not rest on it; (b)–(c) prove it
directly.)*

**(b) Stationarity: the excess is static subtraction artifacts.** A ≥1 °/day mover is ≥27″ from
itself after 39 min; a repeating artifact is at the *same* position. Catalog test (counterpart
< 3″ in the other visit): **65/74 alerts vetoed** (33 with BOTH members static).
**Measured false-kill cost on real movers** (KD-tree test at 30″-offset positions, i.e. where a real
mover's other epoch would be): 12/1147 = **1.05%** and 5/545 = **0.92%** per member ⇒ **≈2% per
alert** (both members tested). The veto must be motion-aware: valid only when expected displacement
rate×Δt exceeds the match tolerance by a wide margin (at 46-s companion cadence a real mover moves
only ~2–9″; the confirmed 3-visit mover's own 1-min sub-pair would otherwise self-veto).

**(c) Pixel-level stationarity (forced trail-capsule photometry, S3 diffims) on the 9 catalog
survivors.** The audit formalized the statistic — kill only on **`snr_at0`** (flux at the exact
position, no grid search) computed from **mask-clean pixels** (excluding BAD/INTRP/EDGE/NO_DATA/
SAT/CR/SUSPECT), threshold **≥ 5σ**. Under this rule:

| alert | mask-clean static flux | verdict |
|---|---|---|
| 000001 | 5.1σ pair (28.6σ was CR-inflated); **13.8σ / 11.6σ in third visits v129/v107, DETECTED** | KILL (multi-visit) |
| 000057 | **11.0σ**, DETECTED | KILL |
| 000017 | **10.8σ**, DETECTED | KILL |
| 000011 | **6.3σ**, DETECTED | KILL |
| 000010 | 8.3σ raw → **3.3σ** after removing BAD/INTRP/EDGE pixels | ~~kill~~ → FLAG |
| 000049 | 3.2σ | FLAG |
| 000003 | 2.6σ | FLAG |
| 000062 | 0.4σ / 0.4σ | CLEAN |
| 000015 | (off mosaic) | NO_COVERAGE |

So the pixel stage kills **4** (not 5 as pre-audit; one retracted as a defect-pixel artifact), flags 3,
clears 1, can't test 1. The kills are real static sources: DETECTED-masked, hours-stable (000001
repeats over 3.5 h), all **below the catalog floor** — invisible to any catalog method.

Key measured rate, re-derived with the capsule (N=210 + 201 matched controls,
`audit/stageC_fstatic.csv`): **f_static — 60% of score≥0.6 detections show ≥3σ (40% ≥5σ) static flux
at the same position 39 min later**; matched 30″-offset control: 1.0% (≥3σ) / 0.5% (≥5σ). Independent
of CNN score (34–47% ≥5σ across score bins 0.6→1.0). The CNN cannot see "will it repeat"; two epochs
of pixels can.

**(d) Bottom line per night (post-audit).** 20260630: 74 alerts → catalog veto → 9 → pixel veto →
**5 not killed** (1 pixel-clean, 3 flagged-marginal, 1 untestable). 20260629: 4 raw 2v alerts (49-s
arcs) → pixel veto kills 3 (clean static flux 6.0σ, 5.7σ, 5.3σ — the last borderline) → **1 golden
candidate** (000004: 40σ/40σ at its own positions, −0.0σ/1.4σ at cross-positions, equal brightness,
collinear 9.2″/49-s displacement, rate 4.5 ± 0.3 °/day).

## 4. Why 2-point orbits stay degenerate (prior result, context)

The 2-point fit is flat in topocentric distance (1.2e-5 °/d variation vs 0.5 °/d trail noise) —
alerts now publish admissible-region ranges (schema 1.2). No amount of vetting fixes *orbit*
information; vetting fixes *realness*. Confidence and orbit quality are separate axes.

## 5. The answer

**Yes — conditionally. Confidence comes from vetoes + geometry, not from the linker's score/χ² gates.**
The base-rate wall as previously measured is mostly *removable structure*, not irreducible chance:

| lever | measured effect | measured cost to real movers |
|---|---|---|
| motion-aware catalog stationarity veto | 74 → 9 alerts (×0.12) | **≈2% per alert** (1.0%/member) |
| pixel stationarity veto (mask-clean snr_at0 ≥ 5σ) | 9 → 5; and 4 → 1 on 0629 | **≈1–3.5% per alert** (0.5%/test typical, 1.75%/test densest panels) |
| third-image kill/confirm at predicted position | decisive where covered (000001: 13.8σ third-visit kill) | ~0 (same 0.5–1.75%/test blend risk) |
| rate ≥ 1 °/day (− 3σ_rate) + geometry (MC §2) | classification purity ≈ 100% (mod JFCs) | none (it's the target bin) |
| per-alert FPP tier: λ = k·n₁·n₂ (null-calibrated), Δt² law | ranks candidates honestly | none |

**Total measured completeness cost of the full veto stack at the recommended operating point: ≈3–5%
of true alerts.** Direct true-mover verification (N=5 movers, 6 valid STAT tests: 1997 UT25 ×2,
2014 WC551, 2016 GL48, golden-candidate ×2): **0 false kills**, all ≤ 2.1σ.
What the audit ruled OUT (this is where "economical with true alerts" bites): killing on the
grid-max statistic (null median 2.5σ, 24% of blind positions ≥3σ — would cost ~10–25% of true
alerts), or killing at 3σ (2–6% per alert, and the 3 flagged alerts show 3σ is noise territory).

**Operating envelope for CONFIDENT (per-alert FPP ≲ 1e-2) same-night 2v NEO alerts:**
- **Short-gap pairs (≈1-min mosaic revisits):** chance annulus scales as Δt² → 49-s pairs have
  ~2300× smaller chance area than 39-min pairs; trails (≈5″) nearly tile the 9–15″ path. Pixel
  stationarity remains valid for rate ≳ 2.4 °/day (disp > L/2 + halfw + 3″). The 20260629 golden
  candidate is this class. Rate gate must include the σ_rate term (§2).
- **Any candidate whose track crosses a third covered image** (even 46 s away): forced photometry
  decides — and it is the *strongest* kill evidence too (000001). Tonight's commissioning mosaics
  were too sparse (44–108 of 189 panels); steady-state WFD coverage makes this the default path.
- **Ordinary-density fields:** post-veto λ scales with n₁n₂ — at 10× lower density product than
  the swarm patch, a clean survivor is already at FPP ≈ few %.
- **NOT reachable:** long-gap pairs in the densest ecliptic patches with no third-image coverage —
  there the honest product remains a ranked follow-up trigger (exactly what the alert stream is).

**Both nights, full accounting (post-audit):** 78 raw 2v alerts → veto stack → **2 pixel-clean
candidates** (0629 golden + 0630 000062) + 3 flagged-marginal + 1 untestable. Purity transformation
~13–39× at ≈3–5% completeness cost, same-night, from 2 visits + archival pixels only.

## 6. What to implement (proposed, in order) — **IMPLEMENTED 2026-07-02 (alert schema 1.3)**

1. **Motion-aware catalog stationarity veto in `link_2visit`** (counterpart < 3″ in the other member
   visit AND expected displacement > 10″ ⇒ veto): kills 88% of false alerts for ≈2%/alert
   completeness. Add `vetoStationary` flag to tracks/alerts rather than silent drop.
   ✅ `link_2visit.stationarity_check` (`--stat-tol-arcsec` / `--stat-min-disp-arcsec` /
   `--no-stationarity`); counterpart trees over the FULL pre-floor catalog (ADCNN-vs-ADCNN, §8);
   `stationarity` block on alerts + `veto_stationary`/`stat_testable` track columns; `write_alerts`
   demotes vetoed alerts below every clean one (rank class 1), never drops.
2. **Pixel vetting stage** (`pixel_vet` productionized). Formal rule from the audit:
   - statistic: `snr_at0` from mask-clean pixels only (exclude BAD/INTRP/EDGE/NO_DATA/SAT/CR/SUSPECT);
   - **KILL at ≥ 5σ; FLAG at 3–5σ (never kill); never kill on the grid-max statistic**;
   - validity guard: expected displacement > L/2 + halfw + 3″;
   - capsules dominated by SAT_TEMPLATE or defect bits → FLAG unless corroborated by a third visit;
   - annotate alerts `pixelVet: CLEAN | STATIC_e1 | STATIC_e2 | FLAGGED | CONFIRMED | NO_COVERAGE`
     with the measured SNRs. ~1–3 s/alert (S3 in-memory reads, panel cache).
   ✅ `ADCNN/linking/pixel_vet.py` (`--in-place` preserves `alerts_prevet.jsonl`); STAT stacks
   across ALL valid covering same-night visits with the §8 combined-OR-single rule (kill =
   snr_comb ≥ 5σ **or** any single valid visit ≥ 5σ — the OR keeps a legit 6σ single-visit static
   from being diluted by quiet visits); defect-dominated capsules (badfrac > 0.5) may only FLAG and
   never enter the stack; third-visit CONFIRMED at the predicted position (aperture widened to the
   prediction error — still no grid search); wired into `run_night` as the `pixel_vet` stage
   (graceful pass-through without `fits_path`).
3. **Per-alert FPP field**: λ̂ = k·n₁n₂ with k calibrated from the cross-field nulls (≥8 donors
   recommended; the current 4 give k to a factor ~2); `confident` tier = CLEAN + FPP below threshold.
   ✅ `fpp_2v_chance.json` (k = 3.273e-7/det² @ Δt_ref 39.2 min, Δt² law) + `fpp_block` /
   `_finalize_fpp` → `fpp.lambdaPair` / `fpp.perAlertShare` / `fpp.nAlertsPair` per alert;
   top-level `confident` = CLEAN/CONFIRMED + no catalog veto + perAlertShare ≤ 0.01 (set by pixel_vet).
4. **Prefer short-gap pairs in scoring**: priorityScore bonus ∝ 1/Δt² for the chance term, with the
   σ_rate-aware rate gate.
   ✅ `priority_score(..., dt_min)` bonus = 0.04·min(1, (10 min/Δt)²) — bounded so 2v NEW max
   2.99 < 3.0 keeps tier order BY CONSTRUCTION (39-min WFD pairs get +0.003: the 2026-06-10 ranking
   recalibration is essentially untouched); `motion.rate_sigma_degday` (√2·max(rms, 0.4″)/Δt) +
   `motion.neoRateGate` (rate − 3σ_rate > rate_lo) annotations on every 2v alert.

Items 1–4 change no frozen op-point keys (they act after the existing gates; new CLI flags with
safe defaults, additive columns/fields only). Unit-tested in
`ADCNN/pipelines/heliolinc/tests/test_veto_stack.py` (19 tests: motion-guard self-veto, OR-rule
dilution, √N stack kill, defect demotion, mask-clean flux exclusion, tier preservation,
demotion-aware cap, graceful no-op).

## 7. What the adversarial audit changed (2026-07-02)

Ordered by the user; all pre-audit claims re-measured with saved artifacts. Corrections:

1. **The pre-audit "hard kill" SNRs were grid-search maxima** (search ±3″ = ~30 correlated trials):
   under pure null the grid max has median 2.5σ and 24% of blind positions exceed 3σ
   (`audit/stageB_null.csv`, N=400). Killing on it would have cost ~10–25% of true alerts —
   the user's fear was justified. Kill statistic re-based on unbiased `snr_at0` (null median 0.0).
2. **One of five pixel kills retracted** (000010: 8.3σ → 3.3σ after excluding BAD/INTRP/EDGE pixels)
   and one downgraded then re-confirmed via third visits (000001: pair test 5.1σ after CR removal,
   but 13.8σ/11.6σ DETECTED static in v129/v107). Two "weak kills" (2.6σ, 3.2σ) demoted to flags.
3. **Completeness costs corrected upward and made explicit**: catalog veto ~1% → measured **2%/alert**;
   pixel veto "≈0" → measured **1–3.5%/alert at 5σ** (blend channel, 0.5–1.75% per test, field-
   dependent); full stack **≈3–5%/alert**, not "~1%".
4. **f_static revised** 38% → **60% @3σ / 40% @5σ** (the trail capsule is more sensitive than the
   pre-audit circular aperture); score-independence re-confirmed.
5. **Direct true-mover safety test added** (was missing entirely): 6 valid STAT tests on 5 real
   movers → 0 kills, max 2.1σ (`audit/stageD_truemovers.csv`, `audit_0629_2v_revet.csv`).
6. **0629 kill margins are thinner than implied**: 5.3–6.0σ (one borderline), not "7–35σ"; the
   golden candidate itself re-verified clean with the corrected statistic (40σ/40σ POS, −0.0/1.4 STAT).
7. **NEO-ness softened** with the JFC caveat and the short-arc σ_rate term (§2).
8. **λ_indep flagged as factor-~2** (4 donors, footprint mismatch) — supporting, not central, evidence.

## 8. The sub-5σ regime (the detector's design point) — measured 2026-07-02

ADCNN exists to work *below* the LSST 5σ source-detection floor. Does a 5σ kill threshold
re-import the floor it was built to beat? **No — but the veto layers behave very differently
with member faintness, and the stack's architecture matters. Measured:**

**First, the design invariant: no 5σ requirement is ever applied to the mover.** The mover
enters at the ADCNN score floor and is never asked to clear any forced-photometry bar. The 5σ
appears in exactly one place — as the confidence the *artifact* evidence must reach before we
are allowed to DELETE an alert. The asymmetry is deliberate: hard to kill, cheap to flag. And
mover safety is brightness-independent — the false-kill channel is blends (0.5–1.75%/test),
which does not grow as the member gets fainter.

**Pixel-veto kill power collapses in the faint bin** (single STAT test, stageC, 39-min repeat;
by fainter-member `mf_snr`):

| fainter member mf_snr | N | static ≥5σ (KILL) | 3–5σ (FLAG only) | <2σ |
|---|---|---|---|---|
| [0, 7) | 124 | **7%** | 48% | 44% |
| [7, 10) | 23 | 74% | 22% | 4% |
| [10, 20) | 45 | 93% | — | — |
| [20, ∞) | 18 | 89% | — | — |

Faint artifacts repeat *faintly* — below what 30-s forced photometry can confirm at 5σ. The
crucial point: this failure is **conservative**. A faint alert the pixel veto can't confirm is
*kept* (flagged at 3–5σ, clean below), never eaten. The cost of the collapse lands on the
faint stream's *purity*, not its *completeness* — exactly the economical direction.

**The catalog veto is the layer that carries kill power into the faint bin, because it is
ADCNN-vs-ADCNN.** Its counterpart catalog is ADCNN's own score≥0.5 detections — it goes as deep
as the detector itself, not to the LSST 5σ floor. On the 74 raw 0630 alerts, by fainter-member
`mf_snr`: **[0,7): 83% vetoed (N=35)**, [7,10): 89% (N=28), [10,∞): 100% (N=11). The detector's
sub-5σ capability *is* the faint-bin veto: ADCNN re-detects the faint static in the other visit
at score≥0.5 even when 5σ forced photometry cannot. Nowhere in the stack does LSST source
detection appear.

**Multi-visit stacking gains √N but N=2 is not enough** (stageE, v106→v129 + small-tile third
epochs, N=89 dets with 2 valid covering visits; `audit/stageE_stack.csv`):
- kill power single → stacked(2): mf_snr [0,7) 1%→4%; [7,10) 50%→60%; [10,∞) 56%→56%;
- the statics stack coherently — corr(snr₁, snr₂ | snr₁≥2) = 0.83 — so the √N math is real;
- missed zone (2–5σ single): 29% rescued to ≥5σ by one extra visit (N=7);
- quiet zone (<2σ): 2.8% false-promoted to ≥5σ (N=71) — the cost channel, still small at N=2.

Commissioning mosaics give mostly 0–1 extra covering visits, so tonight stacking is a marginal
lever; at steady-state WFD cadence (N≈4–8 same-night+archival covering visits) the faint-bin
pixel kill power scales back up. The production rule already accommodates this: **kill on the
COMBINED mask-clean `snr_at0` across all valid covering visits** — same 5σ bar, deeper data.

**Net for the sub-5σ science case:** the confident-alert machinery (catalog veto → pixel
flag/kill → FPP tier) degrades gracefully with member faintness instead of gating on it. The
faint bin keeps ~83% of its FP removal (catalog layer), loses most *pixel-kill* certainty
(7%), and compensates with flags + honest per-alert FPP. Fix the artifact test by making it
deeper (stacking), never looser (3σ kills) and never by raising the score floor.
