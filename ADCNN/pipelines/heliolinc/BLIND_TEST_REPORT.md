# Blinded injection-on-real test — REPORT (round 5, final)

**Date:** 2026-06-11 · **Contract:** `EVALUATION_CONTRACT.md` (pre-registered configs, metrics,
failure criteria, diagnosis order) · **Verdict: PASS** (pre-registered floor; see §4)

**No tuning was performed at any stage.** All operating points were frozen before the blind set was
selected (`op_2v_alert.json`, `op_3v_confirm.json` — read-only throughout). Every number below is the
first and only evaluation at the frozen ops on these fields.

**The result in one paragraph (the paper framing):** on this blind substrate the standard stack wins
raw per-sighting recovery (4σ > 5σ > ADCNN@0.80 in every SNR bin); ADCNN wins selectivity and
alertability (8.5% detection purity at 9.7 dets/panel with trail state — 5× the 5σ stack at 1/7 the
load); and the two are complementary (5σ ∪ ADCNN reaches stack-4σ faint completeness without 4σ's
purity collapse). **ADCNN is not a replacement for Rubin source detection; it is a specialized,
low-load, trail-aware complement that turns faint-fast trail detections into ranked same-night
follow-up alerts.** The frozen 2v alert point transfers: lower completeness than validation
(explained quantitatively by the detector-recall² ceiling, §4), with no purity or ranking collapse.
This document separates three things that must not be blended: *validation threshold selection*
(chose S≥0.80, mfsnr≥5 — done before this test), *blind transfer measurement* (this report), and
*interpretation* (detector recall under domain shift is the main limitation, §4).

**Metric taxonomy (every table below is tagged with these types; they are never interchangeable):**

| type | name | definition |
|---|---|---|
| T1 | per-sighting completeness | fraction of injected *sightings* recovered by a detector (§3, §3.1) |
| T2 | injection-set purity | TP/(TP+FP) among detections or pairs *at the injected truth density* (§2, §3.1) — NOT a real-sky number |
| T3 | base-rate-corrected real-sky purity | C·ρ/(C·ρ+λ) at the real faint-fast base rate ρ≈0.36/field-night (THRESHOLD_PROTOCOL.md) — the only number that may be called "real-sky purity"; remains low for 2v, hence alert-stream-not-discovery |
| T4 | alert-stream / top-N truth fraction | TP fraction among the N highest-priorityScore alerts (§2 ranked columns) |

Object-level pair completeness (§2) is a fifth, distinct quantity: distinct injected faint-fast
*objects* with ≥1 accepted pair over recoverable objects — not per-sighting (T1).

## 1. Blind data set

- **26 field-nights** on the DRP collection `LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195`:
  20 off-ecliptic (clean-FP substrate: no real asteroids, every unmatched pair is definitionally false)
  + 6 ecliptic (dense sky; contains REAL asteroids — see §5 caveat). 20,360 diffim panels,
  nights 2025-05-01 … 2025-07-28.
- **Selection by rule, not by hand** (contract §3): tract-disjoint from ALL validation/training fields
  (strict tract exclusion), night window disjoint where applicable, ≥10 visits/night.
- **Injection:** 300 synthetic NEO-like movers per field (`sim_orbits.py`, seeds 3000+k), rates 1–8°/day,
  target detection-SNR log-uniform 2–30 (the faint-fast science bin = SNR 2–10 AND rate ≥1°/day),
  visits re-timed to the OpSim baseline same-night cadence (`--retime-map`, the v1 lesson). 9,803
  injected sightings; 1,374 recoverable faint-fast objects (≥2 sightings on the retimed cadence).
- **Detection:** ADCNN (seg_v2 iter08 + focal-cutout CNN) at the contract's S≥0.50 retention floor
  (GPU array 28300666, 26/26 COMPLETED) + `mask_flags`. Stack baselines on byte-identical injected
  pixels (`stack_detect.py`, 5σ and 4σ, 52/52 runs clean).

## 2. Product A — same-night 2-visit alert tier at the FROZEN op

Frozen op: `S≥0.80 ∧ mf_snr≥5 ∧ pair_chi2≤5 (gate) ∧ rate∈[1,8]°/day`, priorityScore = base +
0.95·weakest-member score, top-50/night cap. Reducer identical to validation
(`Evaluation/threshold_selection_plots.py` conventions; field-bootstrap 16–84% bands).

| split | pairs@op | TP | FP | injection-set pair purity (T2)¹ | faint-fast object completeness | alerts/field-night | top-5 truth (T4) | top-50 truth (T4) |
|---|---|---|---|---|---|---|---|---|
| **ALL (26)** | 324 | 279 | 45 | **86.1%** [79.6–92.9] | **3.64%** [2.66–4.53] (50/1374) | 12.5 | 92.9% | 85.9% |
| off-ecliptic (20) | 228 | 226 | 2 | **99.1%** [98.2–99.6] | 3.55% [2.32–4.68] (42/1183) | 11.4 | 98.7% | 99.1% |
| ecliptic (6)² | 96 | 53 | 43 | 55.2% [47.9–65.1] | 4.19% [3.41–4.82] (8/191) | 16.0 | 72.7% | 55.2% |

¹ "validation injected-truth fraction": TP/(TP+FP) at the injected truth density — NEVER to be
presented as real-sky alert purity. At the real faint-fast base rate (ρ≈0.36/field-night) the
base-rate-corrected 2v purity C·ρ/(C·ρ+λ) remains low (THRESHOLD_PROTOCOL.md), which is exactly why
this product is a ranked alert stream requiring follow-up, not a standalone discovery claim.
² Conservative lower bound — see §5.

**vs validation (82 injection fields, same reducer):**

| metric | validation | blind | transfer |
|---|---|---|---|
| faint-fast completeness | 6.07% (159 objs) | 3.64% [2.66–4.53] | **60% of validation** (band excludes 6.07) |
| alerts/field-night | 15.2 | 12.5 | consistent |
| ranked stream truth fraction (top-N) | 76.9% | 85.9% (ALL top-50) / 99.1% (off-ecl) | **transferred HIGHER** |

## 3. Baselines (identical injected pixels)

| detector | per-sighting recovery (all) | per-sighting (faint-fast) | faint-fast 2v-able objects (≥2 sightings rec.) |
|---|---|---|---|
| Stack 5σ | 51.3% (5032/9803) | 31.1% | 26.1% (359/1374) |
| Stack 4σ | 59.0% | 40.7% | 35.2% (484/1374) |
| ADCNN (S≥0.50) | 44.5% | 30.3% | — |
| ADCNN at alert floor (S≥0.80) | 36.6% | 22.2% | — |

Note what each column buys: the stack numbers are raw 5σ/4σ peaks with **no** purity mechanism at
faint-fast rates (the validation campaign measured the 2-visit FP wall at ~10⁵:1); the ADCNN S≥0.80
detections carry the score+trail state that lets the frozen op deliver the §2 purity. At the retention
floor ADCNN ≈ stack 5σ on the faint-fast bin per-sighting; the alert tier trades ~8 points of
per-sighting recall for the ranked, high-purity stream.

### 3.1 SNR-binned completeness and detection purity (5σ vs 4σ vs ADCNN vs union)

Per-sighting completeness (T1) by injected target-SNR bin (9,803 sightings, 4,360 injected panels,
identical pixels in every config):

| config | SNR 2–5 (faint) | SNR 5–10 | SNR 10–31 | all |
|---|---|---|---|---|
| stack 5σ | 20.0% | 46.4% | 80.0% | 51.3% |
| stack 4σ | 28.8% | 57.1% | 84.8% | 59.0% |
| ADCNN S≥0.50 | 22.7% | 41.0% | 64.5% | 44.5% |
| ADCNN S≥0.80 | 15.5% | 31.5% | 57.0% | 36.6% |
| **5σ ∪ ADCNN@0.50** | **28.8%** | 56.2% | **88.2%** | **60.1%** |
| 5σ ∪ ADCNN@0.80 | 23.5% | 49.7% | 84.9% | 55.4% |

**EXACT deduplicated detection-level table (T2 purity)** — full 5σ/4σ peak catalogs
(`stack_detect --full-catalog`), union = stack peaks + ADCNN detections positionally deduplicated
(10 px) per panel, each physical detection counted once (`run_blind/exact_union.py`,
`exact_union_table.csv`). FP carry no injected SNR, so purity is per-config global. (The table-1
completeness uses either-detector-hit semantics; the C columns here are against the deduplicated
catalog and differ by ≤0.5 pt where dedup replaces a near detection with a farther peak.)

| config | C 2–5 | C 5–10 | C 10–31 | C all | TP sightings | TP dets | FP dets | purity (T2) | dets/panel | ΔTP vs 5σ | ΔFP vs 5σ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stack 5σ | 20.0% | 46.4% | 80.0% | 51.3% | 5,032 | 4,916 | 277,192 | 1.74% | 64.7 | — | — |
| stack 4σ | 28.8% | 57.1% | 84.8% | 59.0% | 5,781 | 5,683 | 948,802 | 0.60% | 218.9 | +749 | +671,610 |
| ADCNN S≥0.50 | 22.7% | 41.0% | 64.5% | 44.5% | 4,364 | 4,324 | 546,656 | 0.78% | 126.4 | −668 | +269,464 |
| ADCNN S≥0.80 | 15.5% | 31.5% | 57.0% | 36.6% | 3,589 | 3,588 | 38,504 | **8.52%** | **9.7** | −1,443 | −238,688 |
| 5σ ∪ ADCNN@0.50 | 28.2% | 55.6% | 86.7% | 59.2% | 5,800 | 5,768 | 742,265 | 0.77% | 171.6 | +768 | +465,073 |
| **5σ ∪ ADCNN@0.80** | 23.0% | 49.2% | 83.7% | 54.6% | 5,352 | 5,318 | 289,559 | 1.80% | 67.6 | **+320** | **+12,367** |
| 4σ ∪ ADCNN@0.80 | 30.0% | 58.9% | 87.8% | 61.1% | 5,988 | 5,978 | 953,275 | 0.62% | 220.0 | +956 | +676,083 |

Four readings: (1) **the marginal-cost argument, now exact** — adding ADCNN@0.80 to the standard 5σ
stack buys +320 TP sightings for +12,367 FP = **38.6 FP per added TP**, while relaxing the stack to 4σ
buys +749 TP for +671,610 FP = **897 FP per added TP**: per recovered sighting, ADCNN@0.80 is ~23×
more FP-efficient than lowering the stack threshold — and union purity *rises* (1.74→1.80%) because
the additions are 8.5% pure; (2) **the union is complementary, not redundant** — 5σ∪ADCNN reaches
stack-4σ-class faint completeness and beats both detectors alone at high SNR, without 4σ's purity
collapse; (3) **at the alert floor ADCNN is a different kind of object** — 8.5% detection purity (5×
the 5σ stack, 14× 4σ) at 9.7 dets/panel with score+trail state, the only input from which a ranked
linked alert stream can be built; (4) **the stack's mid/high-SNR per-sighting edge on this substrate
is real** — the same domain shift as §4's diagnosis; deployment answer = run both: the stack carries
the bright end, ADCNN adds the faint tail and the purity mechanism (maximum-completeness option
4σ∪ADCNN@0.80 exists at 4σ-like load for surveys that can afford it).

## 4. Failure criterion and diagnosis (contract §5 — in order, NO tuning)

Pre-registered criterion: product fails if completeness or purity falls below **half** the validation
value at the frozen op → C floor = 3.0%. **Blind C = 3.64% [2.66–4.53] → PASS.** Purity transferred
above validation. The completeness gap (3.64 vs 6.07) is real (band excludes the validation point) and
was diagnosed in the contract's fixed order:

1. **Per-sighting recall (the recall² funnel) — CONFIRMED as the driver.** Blind faint-fast
   per-sighting recall: 30.3% raw, **22.2% at S≥0.80**. The pair bound recall² ≈ 4.9% (before op-cut
   and cadence losses) already sits at the observed C≈3.6%; validation C=6.07% implies validation
   per-sighting recall ≈ 28% at the op. A ~5-point per-sighting recall drop on the blind substrate
   (different DRP reprocessing d_2025_11_10, different nights/seeing/airmass mix) fully accounts for
   the completeness transfer gap.
2. Score transfer / 3. mfsnr transfer / 4. domain shift beyond recall / 5. thresholds — **not reached**:
   no anomaly beyond (1); ranking and purity transferred at or above validation, so the score and
   photometric axes are healthy.

Consequence (allowed by contract, not applied here): none. Thresholds stay frozen; the paper quotes
the blind numbers.

## 5. Ecliptic split caveat (honest reading)

The 6 ecliptic fields contain **real asteroids**; injection-based labeling counts every non-injected
pair as "fp", so 55.2% is a **worst-case lower bound** on that split's purity. Two unseparated
components: (a) real movers in the rate∈[1,8] window (genuine alerts, mislabeled by construction);
(b) chance FP pairs at ~10× the off-ecliptic candidate-pair volume (97,591 vs ~10,160 post-gate
pairs). Per post-gate pair the ecliptic FP-at-op rate (4.4×10⁻⁴) is ~2× the off-ecliptic rate
(2.0×10⁻⁴, n=2 — Poisson-compatible), so the chance-FP component alone may explain most of it.
Separating (a) from (b) needs a known-object crossmatch on these fields (queued as follow-up; not done
here to keep the blind reduction free of post-hoc edits).

## 6. Product C — same-night ≥3-detection confirmation tier (frozen `op_3v_confirm.json`)

Run on all 26 fields (linker on re-timed detections, known = injected truth):
see `run_blind/productC_summary.json`. **Interpretation caveat (apples-to-apples):** the validation 3v
anchor λ_3v≈0.0025/night is a NULL-SKY false-track rate; the blind "3v fp" here are tracks at the
INJECTED truth density (mixed triplets — e.g. 2 injected sightings + 1 FP — count as fp), a different
and strictly harsher quantity. The pair-level §2 comparison is the like-for-like one; the 3v tier's
null rate was and remains calibrated on real nights (link_fpp_calib_3visit.json).

**Off-ecliptic (20 field-nights, clean-FP substrate):** 110 3v tracks = **62 TP / 48 pure-false**
(match_frac=0 for all 48 — genuine chance triplets, not mixed tracks), 60 distinct injected objects
confirmed, 2.4 false 3v tracks/field-night.

**The 2.4/night vs the validation λ_3v≈0.0025/night anchor is a cadence-regime difference, not a
model failure:** the null anchor was calibrated on pair-dominated (≈2-visit) WFD nights, where
same-night triplets barely exist combinatorially; the blind fields were *selected* for dense same-night
cadence (12–17 visits/night, median 12 → C(12,3)≈220 triplet slots vs ~0 on a pair night). The
λ_3v(N_visits) surface for dense cadences is follow-up calibration work — pre-registered here, not
retro-fitted.

**The 3v geometry still separates cleanly (measured, not tuned):** TP linear-fit RMS median 0.136″
[16–84%: 0.048–0.46] vs pure-false median 0.70″ [0.46–0.89] — the frozen op's max_rms=1.0″ admits the
dense-cadence chance triplets, and the measured distributions show a future dense-cadence operating
point exists without touching the score axis. Recorded for the next calibration round; **not applied
to any number in this report.**

**Ecliptic (6 fields) — a scalability finding, with the three giant fields terminated:** the three
small ecliptic fields completed normally (their 3v tracks: 40, of which 1 tp / 39 conservative-fp —
real asteroids count as fp under injection labeling). On the three giant ones (4,100–4,900 panels,
300–900k detections/field-night) the frozen `op_3v_confirm` linker was **terminated after ~19.7 h
CPU per field with no output** — vs minutes per off-ecliptic field. py-spy localized all three to
`extend_to_triplets`: the 3v-first seeding scans chord pairs in a 180-minute arc window (~4.5× the
40-min alert window) over 12–17-visit nights at ecliptic FP density, producing millions of raw
pairs, each paying a Python-level per-visit attach query — the same unculled-input scalability wall
measured for the multi-night chain (heliolinx post-proc explosion) and for S<0.8 make_tracklets.
**Conclusion: the frozen 3v configuration is computationally infeasible on ecliptic-density
same-night fields without an O(N log N) pre-cull** — recorded as a deployment constraint of the
tier, alongside the dense-cadence false-rate finding above. Completed-fields aggregate
(`productC_summary.json`, 23 fields incl. the 3 small ecliptic): 150 3v tracks, 63 tp / 87 fp at
injected density (off-ecliptic split: 62 tp / 48 fp as analyzed above).

## 7. Product D — multi-night discovery: honestly DEFERRED

The blind set is single-night by construction (same-night re-timed cadence), so the multi-night
product cannot be scored on it. Deferral, not failure. **Proposed multi-night blind window** (next
campaign): a tract-disjoint field with ≥4 consecutive retained nights in DM-53195, Sorcha-propagated
Granvik orbits (real multi-night arcs, as in the H2H campaign) injected at the same faint-fast
magnitudes, hybrid router at the frozen `op_multinight_discovery.json` (S≥0.80 default; S≥0.60
reservoir behind the length-split router) — one shot, no tuning, same contract discipline.

## 8. Incident log (everything that went wrong, in order)

1. **v1: retime-map omission.** First blind injection ran `sim_orbits.py` WITHOUT `--retime-map` while
   the evaluation chain re-timed — injected motion inconsistent with evaluated cadence → C≈0 artifact.
   Caught by the contract's diagnosis order (per-sighting recall was healthy 26–40% → not the
   detector). Full redo; the v1 field set is quarantined in `run_blind_v1_purged/`.
2. **Datastore purge race (DM-53881/stage4).** The original collection was being decommissioned
   mid-campaign: older nights deregistered, recent files purging between selection and injection
   (panel counts shrinking between stages). Mitigation: per-field purge canary (≥70% of manifest
   panels readable) + migration (4).
3. **Blind-pool exhaustion.** Strict tract-disjoint selection on the shrinking DM-53881 pool collapsed
   (275→4 candidate fields at ≥20 visits); relaxed to ≥10 visits and a two-tier exclusion mode
   (`--exclude-mode tract|tract-night`) — then made moot by (4).
4. **Migration to `…/d_2025_11_10/DM-53195`** restored a stable, strict-tract-disjoint 26-field pool
   (FITS verified on disk).
5. **Pixel-'A'-WCS (the big one).** DM-53195 diffim FITS headers carry NO astropy-readable sky WCS:
   the primary header has none and the 'A' key is a CTYPE='PIXEL' bookkeeping transform; the exact
   SkyWcs lives only in LSST archive HDUs. `WCS(hdr, key='A')` silently returned a pixel-identity
   transform → round-4 injection placed "sky" coordinates in pixel units → 24/26 fields with 0
   sightings. A count-only verification ("N panels readable") missed it. **Fix:**
   `annotate_manifest_wcs.py` Butler-reads `difference_image.wcs` per panel; most DM-53195 SkyWcs have
   no attached FITS approximation (`getFitsMetadata()` raises), so a TAN-SIP is FIT to the exact
   transform on a 24×24 pixel grid (`astropy fit_wcs_from_points`) and **validated end-to-end** per
   panel: an astropy WCS is rebuilt from the stored JSON cards (the exact consumer code path) and
   compared to the exact SkyWcs on a held-out 17×17 grid. Measured residuals 12–98 mas across all
   20,360 panels (worst: field 24 at 99.8 mas ≈ 0.5 px, vs the 127 mas pair-perpendicular sigma);
   tolerance gate 0.1″. Injection, detection and linking all consume the same approximation
   (self-consistent). Readers (`sim_orbits`, `discover_stream`, `ephem_to_inject`) now prefer manifest
   `wcs_json`, **fail loud** on non-celestial header WCS, and sanity-check panel centers
   (finite, |dec|≤90).
6. **BLAS oversubscription.** The first annotation run let each TAN-SIP fit grab every core × 8
   processes → ~53 CPU-s/panel and a ~3 h stall on the large manifests. Restarted with
   `OMP/OPENBLAS/MKL_NUM_THREADS=1` and per-row parallelism: the 4,315-panel manifest annotated in
   ~90 s.
7. **Process-kill self-match.** Relaunching product C in parallel, the `kill` pattern matched the new
   run's own shell command line (it contained the same script name) and killed it (exit 144). Relaunch
   without an embedded kill; tracked artifacts (`tracks3v_*.csv`) made every restart resumable.

None of these incidents involved threshold or science-parameter changes.

## 9. Provenance

- Fields/manifests/retimes/injects/truths/dets/caches: `run_blind/` (quarantines: `run_blind/_bad_wcs/`,
  `run_blind_v1_purged/`). WCS annotation: `annotate_manifest_wcs.py` (manifest column `wcs_json`).
- Detection: SLURM array 28300666 (ada, rubin:commissioning, `--cnn-thr 0.50`), 26/26 COMPLETED.
- Pair tables: `exact_lowS_pairs.py --dir run_blind --smin 0.5` → `run_blind/_nomfsnr_cache/*_smin0.5_v3exact.json`
  (post-`physical_check` rows incl. chi2≤5 gate; the S≥0.5 retention also provides the contract's
  appendix-control data).
- Reduction: `run_blind/reduce_blind.py` → `run_blind/blind_frozen_op_reduction.json` (this report's §2).
- Stack baselines: `run_blind/stack_all.sh` → `stack_dets_s{5,4}_*.csv`.
- Product C: `run_blind/run_productC.py` → `tracks3v_*.csv`, `productC_summary.json`.
- Frozen configs: `op_2v_alert.json`, `op_3v_confirm.json` (untouched; see git history).

## 10. Next sprint (pre-registered here, after the blind result — in order)

1. **Exact deduplicated union baselines** (paper table): rerun `stack_detect --full-catalog` (5σ, 4σ)
   so stack peaks carry positions, positionally match stack↔ADCNN catalogs, and report for
   {5σ, 4σ, ADCNN@0.50, ADCNN@0.80, 5σ∪ADCNN@0.50, 5σ∪ADCNN@0.80, optionally 4σ∪ADCNN@0.80}:
   completeness per SNR bin, exact detection purity, dets/panel, and **incremental TP and FP vs
   stack 5σ**. (§3.1's union row is bounded, fine for interpretation, not for the paper.)
2. **Stack-seeded trail-measurement branch** (exploit the complementarity instead of just reporting
   it): stack 5σ detections → trail-morphology preselection → Veres trailed fit / ADCNN local scoring
   → trail state (length, PA, mfsnr, score) → merge into the candidate stream → the SAME frozen 2v
   linker. Goal: convert the stack's mid/high-SNR recovery edge into alert-capable candidates, NOT
   feed raw stack catalogs to the linker.
3. **Detector miss audit** (before any retraining): three-way error census on the blind set —
   stack-found/ADCNN-missed (by SNR, trail length, band, seeing, background, detector, subtraction
   quality), ADCNN-found/stack-missed (the true value region), and high-score ADCNN FP (artifact
   taxonomy). Output = the retraining objective, designed from measured failure modes (domain shift
   vs trail morphology vs stage-2 over-rejection vs calibration), not guessed.
4. **Only then** define/execute retraining.
5. **3v dense-cadence recalibration: deferred.** The 3v geometry separates (§6) but the validation
   operating point was calibrated on ~2-visit nights; dense fields need a separate λ_3v(N_visits)
   calibration. Do not ship the current 3v threshold for dense cadences.

Explicitly NOT next: changing the frozen 2v thresholds post-blind; claiming ADCNN-alone beats the
stack in raw detection; presenting injection-set purity as real-sky purity; 2v low-threshold
reservoirs; forced-counterpart search (closed); retraining before the miss audit.
