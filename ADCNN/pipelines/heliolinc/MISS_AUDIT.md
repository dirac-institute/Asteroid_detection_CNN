# #251 — Detector miss audit on the blind set → retraining objective (DEFINED, not executed)

**Date:** 2026-06-11 · **Data:** all 26 blind fields (`run_blind/miss_audit.py` →
`run_blind/miss_audit.json`), 9,803 injected sightings, ADCNN masked catalogs + stack-5σ per-sighting
recovery. **Measurement only — no model, threshold, or config was changed.**

## Census (per injected sighting, ADCNN at the S≥0.80 alert floor vs stack 5σ)

| category | n | share |
|---|---|---|
| both | 3,194 | 32.6% |
| stack-only | 1,838 | 18.7% |
| ADCNN-only | 395 | 4.0% |
| neither | 4,376 | 44.6% |

At the S≥0.50 retention floor: both 3,500 / stack-only 1,532 / ADCNN-only 864 / neither 3,907.

## (a) Stack-found / ADCNN-missed — the decomposition that defines the objective

Of the 1,838 stack-only sightings: **1,532 (83%) have NO ADCNN detection at all** — segmentation
never fired, even at the S≥0.50 retention floor — vs only 306 (17%) detected-but-sub-threshold
(median score 0.71). **The dominant failure is stage-1 segmentation recall under the DM-53195
domain shift, NOT stage-2 score calibration.** Corroboration: #250's seeded-MF showed these
stack-only additions are genuine mid-SNR trails (median true length 17.9 px, snr_target ~14,
PA recoverable to 4.5°) — bright enough that per-sighting flux is not the limitation.
By SNR the stack-only losses concentrate at 10–31 (1,133 of 1,838); by trail length they are flat
(709 / 518 / 554 across 6–12 / 12–20 / 20–41 px) — i.e. NOT a short-trail morphology gap but a
broad substrate shift (newer reprocessing: different subtraction residual statistics/PSF handling).

## (b) ADCNN-found / stack-missed — the value region, confirmed

395 sightings @0.80 (864 @0.50) that stack 5σ misses. By length: **184 of 395 are LONG trails
(20–41 px) vs only 11 short (6–12 px)** — trail dilution kills the stack's point-source significance
exactly where matched segmentation wins. By SNR: 116 in the faint 2–5 bin. ADCNN's unique
contribution is the long-faint-trail population — the faint-fast science bin — as designed.

## (c) High-score FP taxonomy (228,651 FP vs 3,588 TP detections at S≥0.80, all panels)

| feature | FP | TP |
|---|---|---|
| len_db median | 9.5 px | 14.4 px |
| mf_snr median | 3.9 | 8.0 |
| art_frac > 0 | 18.8% | 2.9% |
| m_DETECTED_NEGATIVE | **12.7%** | 0.1% |
| m_BAD / m_INTRP | 2.4% / 2.7% | 0.2% / 0.2% |

The dominant high-score FP class: **short, low-mf_snr, disproportionately negative-paired
subtraction residuals** (dipoles) — a learnable hard-negative class, distinct from real trails on
every measured axis.

## Also measured: the joint floor

44.6% of injected sightings are invisible to BOTH detectors (2,549 of them in the faint SNR 2–5
bin) — the photon-noise/cadence floor no retraining reaches; keeps expectations honest.

## RETRAINING OBJECTIVE (defined here; execution is a separate, future decision)

1. **Primary — stage-1 segmentation domain adaptation to the DM-53195 reprocessing.** Evidence:
   83% of stack-only losses have zero segmentation response; losses are SNR-broad and
   length-flat (substrate shift, not morphology). Recipe: the blind campaign's own injection
   machinery (sim_orbits + wcs_json manifests) on **non-blind** DM-53195 fields → fine-tune
   the segmentation model; the 26 blind fields stay untouched as the held-out eval.
   Expected gain ceiling: recovering the stack-only pool would lift per-sighting recall
   22.2%→~41% at the alert floor, i.e. roughly double pair completeness (recall²).
2. **Secondary — stage-2 hard-negative emphasis on the dipole/short-trail FP class**
   (m_DETECTED_NEGATIVE-rich, len_db~9.5, mf_snr~3.9): a targeted hard-negative set could raise
   the 8.5% detection purity without touching recall.
3. **Exploratory (#250 handoff) — a stage-2 variant NOT conditioned on segmentation channels**
   (diffim-only input): the only path by which stack-seeded candidates could ever be scored;
   decisive test = can it separate the #250 stack-only true trails from 5σ artifacts?
4. **Explicitly NOT justified by the data:** moving the score threshold (only 17% of losses are
   sub-threshold, median 0.71 — a threshold change buys little and breaks the frozen op);
   mfsnr cut changes; any 2v gate retuning.

**Guards:** blind set = eval-only forever; any retrain is validated on it once, blind-style.
