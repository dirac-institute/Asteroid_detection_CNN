> **Provenance / historical record.** `v2_D` is the development name of what is now the **current** default pipeline (`models/current/`, frozen `adcnn-v2_D-rc1`). For the active workflow use the repo-root `REPRODUCE.md` / `TRAINING_PROTOCOL.md` / `EVALUATION_PROTOCOL.md` and `python -m ADCNN.pipelines.run_experiment`. This doc is kept for the development record.

# ADCNN v2_D — decision report: the trail-length de-bias recalibration (MF_LEN)

**Date:** 2026-06-14 · **Phase:** v2 Phase 3 (dev alert gate) · **Status:** BLOCKED on a user decision,
mirroring the (already-approved) stage-2 refit decision. Blind set untouched; frozen thresholds untouched.

## 1. One-paragraph summary

v2_D's detector is genuinely better than v1 (stage-1 domain adaptation lifted faint-fast recall +4.7 pt,
and at the alert floor it detects **more** pairable real injected objects than v1). But the first dev
alert-product read showed v2_D producing **zero** 2-visit alert pairs. The cause is **not** a v2_D failure:
it is a *third* v1-calibrated post-processing constant — the **trail-length de-bias** `MF_LEN_OFFSET/SLOPE`
in `ADCNN/inference/catalog.py` — that the new stage-1 invalidated, exactly like it invalidated the stage-2
score (which we already re-fit). With the old constants, v2_D's trail-length column `len_db` collapses to ~0,
so the linker's **frozen** `len_db ≥ 6 px` floor deletes ~83% of v2_D's high-confidence detections *before*
they can pair. The alert eval as-run therefore measured a calibration mismatch, not v2_D's product. To get a
valid v2_D alert number we must re-derive the two trail-length constants (a 2-parameter linear fit on the dev
injections) and re-detect — the same class of required recalibration as the stage-2 refit. This document lays
out the full evidence so the decision (approve the MF_LEN re-derivation, or stop v2 here) can be made cleanly.

## 2. Background: how we got here

- #251 audit → stage-1 segmentation recall is the limiting factor under the DM-53195 domain shift.
- v2 sprint: 3 fine-tune variants; **v2_D (hard-positive oversampling) won** the detector ladder:
  per-sighting recall @0.5 36.6→43.1% (all), **22.9→27.6% faint (+4.7 pt)**, gains every SNR bin, lower load.
- Alert floor (@0.80) was unmeasurable for the v2-stage1 + v1-stage2 *chimera* → you approved the **stage-2
  refit** (the required end-to-end recalibration after changing stage-1). Stage-2 refit completed:
  CNN trained, best_val_auc 0.86, score scale verified sane at S=0.80.
- Dev alert eval of the full v2_D pipeline (v2_D seg + v2_D stage-2) began → surfaced the problem below.

## 3. The symptom

Dev alert ladder at the frozen op (S≥0.80, mf_snr≥5, rate∈[1,8]), fields 0–2:

| | v1 | v2_D (full) |
|---|---|---|
| physical-check-pass pairs (S≥0.5) | 25,824 | **99** |
| true alert pairs at frozen op | 39 | **0** |
| faint-fast completeness | 4.65% | **0.00%** |

260× fewer pairs and zero alerts looked, at first, like the pre-registered "detector gain didn't propagate to
product" FAIL. **It is not** — see §4.

## 4. The decisive diagnostic (why it is NOT a v2_D failure)

Per injected faint-fast object, count those with **≥2 sightings detected at score ≥0.80** (the pairable
condition), independent of the linker — pure detector question (10 px positional match, fields 0–3):

| | v1 | v2_D |
|---|---|---|
| pairable faint-fast objects @≥0.80 | 63/477 (13.2%) | **100/477 (21.0%)** |

**v2_D detects 59% MORE pairable real faint-fast objects at the alert floor than v1.** The detector is
strictly better at exactly the thing the alert product needs. So the 0-pairs result must come from a linker
stage *after* detection — which it does.

## 5. Root cause: the trail-length de-bias is a v1-fit constant

`len_db` (trail length, px) at score ≥0.80, per field — the column the linker's `len_db ≥ 6 px` floor reads:

| field | v1 median / ≥6px | v2_D median / ≥6px |
|---|---|---|
| 0 | 15.7 px / 74% | **0.0 px / 17%** |
| 1 | 15.8 / 75% | 0.0 / 15% |
| 2 | 15.3 / 73% | 0.0 / 14% |
| 3 | 16.5 / 76% | 0.0 / 17% |

v2_D's trail lengths collapse to ~0, so the frozen `len_db≥6` floor removes ~83% of its high-score detections
*before* chord-seeding/pairing. That fully explains 99 vs 25,824 pairs.

**Why the collapse:** trail length is de-biased in `ADCNN/inference/catalog.py` as
`len_db = clip((mf_length − MF_LEN_OFFSET) / MF_LEN_SLOPE, 0, ∞)` with **`MF_LEN_OFFSET = 33.4`,
`MF_LEN_SLOPE = 0.887`** — fit to **v1's** segmentation, which over-extends ("ends-bloom") trail ends by a
characteristic amount. The new stage-1 (v2_D) has a different raw `mf_length` distribution, so subtracting
v1's 33.4 px offset drives most v2_D lengths below 0 → clipped to 0. The constant is simply mis-matched to
the new detector; it is not measuring v2_D's trails.

This is **the same failure class as the stage-2 score chimera** you already diagnosed and approved fixing —
a v1-calibrated post-processing constant that any stage-1 change invalidates. There are exactly three such
constants downstream of stage-1: (a) stage-2 CNN score [re-fit ✓], (b) the FP-budget threshold [informational,
not needed — we detect at 0.5 and gate at 0.80], (c) **the MF_LEN trail-length de-bias [NOT re-fit ← this]**.

## 6. The fix (what "approve" entails)

Re-derive the two constants for v2_D: fit `mf_length ≈ SLOPE·L_true + OFFSET` on the dev injections (we have
truth `trail_length` per injected sighting and v2_D's raw `mf_length`), giving `MF_LEN_OFFSET_v2D`,
`MF_LEN_SLOPE_v2D`. Apply them v2_D-specifically (not globally — v1 keeps its constants), then re-run the dev
detection so `len_db` is on the correct scale. Cost: a quick linear fit + one more dev-detection pass
(~the pass already running). **Caveat to verify:** the raw `mf_length` must be recoverable — if the catalog
only stored the already-clipped `len_db`, the re-detect is mandatory (cannot post-hoc invert the clip); we
also confirm `length` isn't independently usable. The frozen `len_db≥6` linker floor is **unchanged**; only
the measurement feeding it is corrected — exactly analogous to keeping S=0.80 fixed while re-fitting the
stage-2 score.

After the fix: valid dev alert ladder vs v1 → the pre-registered DEV GATE (≥20% rel faint-fast C gain, or
similar C at substantially lower load, or better 5σ∪ADCNN efficiency; and no purity/top-N/long-trail
collapse) → conditional single blind shot. Given v2_D already detects +59% more pairable faint-fast objects
at the floor, there is a real prior that the corrected alert number beats v1 — but that is exactly what the
gate must measure, not assume.

## 7. The decision

- **Option A — approve the MF_LEN re-derivation for v2_D** (recommended): the third and last required
  recalibration after a stage-1 change; it is the only way to obtain a valid v2_D alert-product number.
  Small fit + one dev-detection pass, then gate, then conditional blind shot. No frozen threshold or blind-set
  change.
- **Option B — stop v2 here**: close with the honest, defensible result — "stage-1 domain adaptation improved
  the detector (+4.7 pt faint recall; +59% pairable faint-fast detections at the alert floor) and stage-2 was
  successfully re-fit, but converting the gain to the 2-visit alert product additionally requires re-deriving
  the trail-length de-bias, which is deferred." Ship v1 as the paper baseline; v2_D documented as a validated
  detector improvement pending one more recalibration.

## 8. Guardrails honored throughout

Blind 26 fields untouched and write-protected. Frozen linker thresholds (S≥0.80, mf_snr≥5, chi2≤5,
len_db≥6, top-50/night) unchanged. No retuning. The only changes contemplated are recalibrations of
v1-fit *measurement* constants made necessary by the deliberate stage-1 change — each surfaced for explicit
approval, one at a time, as with the stage-2 refit.

## 9. Evidence files

- Detector ladder: `run_dev/v2_detector_ladder.md`
- v2_D stage-2 refit: `run_ft/v2_D_cnn_postproc.{pt,json}` (best_val_auc 0.86)
- Dev detections: `run_dev/v2_D_s2/adcnn_dets_masked_*.csv` (array draining; 4/21 at writing)
- MF_LEN constants: `ADCNN/inference/catalog.py` (MF_LEN_OFFSET=33.4, MF_LEN_SLOPE=0.887)
- Charter: `ADCNN_V2_SPRINT.md`
