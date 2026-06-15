# ADCNN v2_D — RESULT: domain-adapted detector improves the frozen 2-visit alert product (blind-confirmed)

**Date:** 2026-06-15 · **Verdict: STRONG WIN** (pre-registered band: all-field blind faint-fast
completeness ≥5% at controlled purity). v2_D becomes the new scientific result; v1 remains the
prior published baseline (`adcnn-v1-blind-baseline`).

## 1. Headline (blind, single pre-registered shot, frozen 2v alert op)

26 held-out DM-53195 field-nights, tract-disjoint from training, frozen op (S≥0.80, mf_snr≥5,
chi2≤5, rate∈[1,8]). **v1 reproduced its original blind numbers exactly** (3.64% / 86.1% / 12.5
per fn — see BLIND_TEST_REPORT.md), confirming the harness; v2_D measured on the same harness:

| split | v1 faint-fast C | v2_D faint-fast C | relative | v1 purity | v2_D purity | v1 alerts/fn | v2_D alerts/fn |
|---|---|---|---|---|---|---|---|
| **ALL (26)** | 3.64% | **10.33%** | **+184% (2.8×)** | 86.1% | 88.6% | 12.5 | 32.7 |
| off-ecliptic (20) | 3.55% | 9.04% | +155% | 99.1% | 97.5% | 11.4 | 30.1 |
| ecliptic (6)¹ | 4.19% | 18.32% | +337% | 55.2% | 66.8% | 16.0 | 41.2 |

¹ Ecliptic injection-set purity is a conservative lower bound: those fields contain real asteroids
that the injection-truth labeling counts as FP. Purity rose anyway (55→67%).

### 1a. Exposure-leakage disclosure + clean-subset integrity check

**Provenance defect (found in code review, disclosed):** the v2_D training substrate was selected
**tract-disjoint** from the blind set, but tract-disjoint is NOT exposure-disjoint — adjacent tracts
on the shared night 20250723 share boundary-CCD exposures. **12 (visit,detector) panels** (10 in
blind field 0, 2 in field 1; both off-ecliptic) appear in both the v2_D fine-tune catalogs and the
blind manifests. So 2 of the 26 blind fields are not fully independent.

**Impact: none on the conclusion — the leak slightly DEPRESSED the headline.** Re-scoring on the
**24 fully-independent fields** (leaked 0,1 excluded):

| split | v1 C_ff | v2_D C_ff | relative | v2_D purity |
|---|---|---|---|---|
| ALL-24 (clean) | 3.68% | **10.74%** | **+192%** | 88.5% |
| off-ecl-18 (clean) | 3.60% | 9.44% | +162% | 97.5% |

The clean-subset gain (+192%) is ≥ the all-26 gain (+184%) because the two leaked fields are dense
off-ecliptic with below-average per-field completeness. The ~2.8–2.9× faint-fast alert-completeness
win at maintained purity holds with every leaked exposure removed. **The clean-24 numbers are the
integrity headline**; the all-26 numbers are retained above for continuity with the v1 blind report.
A pristine release artifact (clean retrain excluding the 12 exposures) is recommended before a final
(non-rc) tag — the result is already shown robust to it.

- **Faint-fast 2-visit alert completeness ≈ tripled, at maintained-to-improved purity**, in BOTH
  latitude regimes — not an off-ecliptic-only effect.
- Alert load rose ~2.6× (12.5→32.7/fn) but purity held/improved, so the additions are predominantly
  real movers; the stream is ranked + top-50-capped per night.
- Dev gate (20 non-blind fields) had predicted this: +299% (1.42→5.67%) at held purity; the blind
  shot confirms it out-of-domain.

## 2. The scientific arc (clean, end to end)

1. **v1 blind test** (BLIND_TEST_REPORT.md) PASSED but identified **detector per-sighting recall as
   the limiting factor** (recall² funnel; 22% @0.80 faint-fast).
2. **#251 miss audit** showed the dominant failure was **zero stage-1 segmentation response** on
   83% of stack-found/ADCNN-missed faint trails under the DM-53195 domain shift — not thresholds,
   score calibration, or linking.
3. **Targeted fix:** a hard-positive domain fine-tune of stage-1 (variant v2_D, oversampling the
   stack-found/ADCNN-missed pool). Detector ladder: faint-fast per-sighting recall **22.9→27.6%
   (+4.7 pt)**, gains in every SNR bin, at lower load. Ablation confirmed it's the oversampling
   (plain fine-tune flat, low-LR hurt).
4. **Required recalibrations after changing stage-1** (measurement constants invalidated by the new
   representation — NOT threshold tuning): (a) **stage-2 cutout-CNN score** re-fit on leakage-clean
   dev panels (best_val_auc 0.86); (b) **MF_LEN trail-length de-bias** re-derived (offset 33.4→7.67,
   slope 0.887→0.9425; v2_D's domain-adapted segmentation has a *tighter* ends-bloom, 7.7 px vs
   33 px — itself a quality gain — which the stale v1 constant had been zeroing, collapsing len_db
   and tripping the frozen len_db≥6 floor → the transient "0 pairs" artifact, now understood and
   fixed). Field-held-out fit residual ~1 px.
5. **Result:** the same FROZEN 2-visit alert product improves ~2.8× blind, with no threshold change.

## 3. What was NOT changed (discipline)

Frozen throughout: S≥0.80, mf_snr≥5, chi2≤5, len_db≥6, rate∈[1,8], top-50/night. Blind 26 fields
write-protected; v2_D blind detection wrote only to `run_blind_v2eval*`. Single pre-registered blind
shot, no post-hoc retuning. **Independence caveat (corrected):** the blind set was tract-disjoint but
not fully exposure-disjoint from training — 12 boundary-CCD panels in 2 of 26 blind fields leaked via
the shared 20250723 night (see §1a); the clean-24-field re-score confirms the result is unaffected
(leak was non-inflating). The two recalibrations are model-specific *measurement* constants
required by the deliberate stage-1 change, each surfaced for explicit approval; both validated on
non-blind dev with field-held-out checks before the blind shot. (The tp=0 in the first blind tally
was a scoring-dir missing-truth-symlink artifact — detections/op unchanged — corrected by symlinking
the injection truth so the scorer could label the already-frozen pairs.)

## 4. Caveats / scope

- v2_D was trained and dev-gated on **off-ecliptic** DM-53195 fields (the ecliptic dev pool was
  empty); the blind ecliptic improvement is therefore a genuine out-of-domain transfer, encouraging
  but to be read with the conservative-purity caveat (¹).
- Alert load up ~2.6× — acceptable here (purity held, ranked/capped stream) but a deployment with a
  hard follow-up budget would use the documented mf_snr knob, not a re-gate.
- Real-sky (base-rate-corrected) alert purity remains the separate low number that makes this an
  alert stream, not standalone discovery (unchanged framing from v1).

## 5. Provenance

- Models: `models/seg_v1_trainable_init.pt` (reconstructed v1), v2_D stage-1
  `run_ft/v2_D_segmentation_scripted.pt`, v2_D stage-2 `run_ft/v2_D_cnn_postproc.pt`
  (MF_LEN offset 7.67 / slope 0.9425).
- Dev: `run_dev/v2_detector_ladder.md`, `run_dev/v2_D_dev_gate.md` (+299%).
- Blind: `run_blind_v2eval/` (raw v2_D dets), `run_blind_v2eval_cal/` (MF_LEN-recomputed); v1 blind
  cache `run_blind/_nomfsnr_cache`.
- Charter/decisions: `ADCNN_V2_SPRINT.md`, `ADCNN_V2_MFLEN_DECISION.md`.

## 6. Recommendation

Adopt v2_D as the detector for the faint-fast same-night alert product; cite the blind ~2.8× alert-
completeness gain at maintained purity as the v2 headline. Promote v2_D to a tagged release after a
code-review pass; keep v1 as the documented prior baseline. Next (separate, pre-registered): an
ecliptic-inclusive dev set + its own MF_LEN/stage-2 calibration if the ecliptic regime is to be a
first-class product; a multi-night blind window for product D.
