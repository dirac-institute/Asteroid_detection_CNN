# ADCNN v2 — detector-domain-adaptation sprint (charter, pre-registered)

**Question:** can a DM-53195-domain-adapted ADCNN improve stage-1 segmentation recall without
destroying the low-load alert stream?

**Status of v1 (unchanged by anything below):** frozen thresholds, blind PASS (C 3.64%, off-ecl
injection-set pair purity 99.1%), stack/ADCNN complementarity established, detector recall identified
as the limiting factor. Tag `adcnn-v1-blind-baseline`. The v1 blind-test conclusion is the completed
paper baseline; v2 is the next experiment.

## Rules (non-negotiable)

1. **The blind 26 fields are untouchable** — no training, no threshold tuning, no model selection on
   them (`run_blind/BLIND_SET_FROZEN`, files write-protected). Used exactly ONCE at the end:
   ADCNN v1 frozen baseline vs ADCNN v2 frozen candidate.
2. **Training data = synthetic trails injected into REAL DM-53195 difference images** with known
   injection truth/masks. No real-asteroid labels, no synthetic backgrounds.
3. **Thresholds stay fixed during all detector development:** S≥0.80, mfsnr≥5, chi2≤5, score_min
   ranking, top-50/night. Any improvement must come from the detector, not retuning.

## Why now (the #251 diagnosis)

83% of stack-only losses have zero segmentation response even at S≥0.50 → the failure is stage-1
recall under domain shift, not thresholds/stage-2/mfsnr/linking. Ceiling math (pair product ≈
recall²): 22%→30% per-sighting ≈ 4.8→9.0%; →35% ≈ 12.3%; →41% ≈ 16.8%. The only factor-level lever.

## Phases

**Phase 0 — tag/freeze (DONE):** tag `adcnn-v1-blind-baseline` pushed; BLIND_TEST_REPORT.md, configs,
checkpoints, evidence caches frozen + write-protected.

**Phase 1 — non-blind DM-53195 development set:** fields/nights strictly tract-disjoint from the
blind 26; mix of off-ecliptic + ecliptic, varied seeing/background; same injection population
(sim_orbits faint-fast definition) + same retiming logic; wcs_json annotation as in the blind chain.
**Train/val split by FIELD/NIGHT, never by tile.** Also run v1 detection + stack 5σ on the dev set to
label the hard-positive pool (stack-found / ADCNN-missed injected trails) for variant D.

**Phase 2 — stage-1 fine-tune FIRST (no stage-2, no ch3, no architecture search):** current
checkpoint as initialization; conservative: low LR, short schedules, field-grouped validation, early
stopping on ALERT-level metrics. Variants:
- A. baseline current checkpoint (control)
- B. fine-tune all layers, low LR
- C. fine-tune decoder/head heavier than encoder
- D. fine-tune with hard-positive oversampling (stack-found/ADCNN-missed injected trails)
Target: lift segmentation response on the missed trails — especially SNR 5–10, lengths ~10–40 px —
while preserving the ADCNN-only long-trail (20–41 px) recoveries.

**Phase 3 — three-level evaluation ladder for every checkpoint (dev set, frozen op):**
1. Detector: per-sighting recall by SNR bin / trail length / band / field type; detection-level
   purity; detections/panel.
2. Alert: 2v faint-fast completeness at the frozen op; alerts/field-night; top-5/top-50 truth;
   injection-set purity; base-rate-corrected real-sky alert purity.
3. Complementarity: stack 5σ / ADCNN v2 / 5σ∪v2; incremental TP and FP vs stack 5σ.
The product metric is the ALERT metric, never segmentation F1 / pixel AUC.

## Success criteria (ALL required before v2 replaces v1; assessed on dev, then ONE blind shot)

- per-sighting recall @S≥0.80: 22% → ≥28–30%
- 2v alert completeness: 3.64% → ≥~5%
- load: dets/panel ≤ ~2× v1 (9.7 → ≤~20) unless alert purity improves
- ranking healthy: top-5 truth fraction does not collapse
- ADCNN-only long-trail value preserved (20–41 px regime)
- union efficiency: added-FP-per-added-TP vs stack 5σ remains ≪ stack-4σ's 897
A checkpoint that lifts raw recall but explodes candidate load is REJECTED (that recreates 4σ).

## Explicitly NOT in this sprint

Full architecture redesign; stage-2-only retraining; ch3/mask wiring (kept as a later ablation:
"can a mask/context plane cut high-score FP at fixed recall?" — relevant only if v2 recall rises
with FP load); threshold changes; training on real asteroid labels; optimizing pixel AUC.
