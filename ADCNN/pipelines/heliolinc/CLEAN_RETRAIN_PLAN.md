# Optional staged clean retrain (all-26 hardening) — DO NOT run unless needed

> **Status: OPTIONAL hardening, not required.** The shipped, defensible blind result is the
> leakage-free **CLEAN-24** (`3.68% → 10.74%`, +192%, purity 86.0→88.5%) — enough for paper prep.
> This retrain only recovers the 2 excluded fields to quote an *all-26* number. Run it only if the
> manuscript/collaborators need the "all 26 fields" framing. Tagged state: `adcnn-v2_D-rc2`.

## Manuscript wording for the CLEAN-24 headline (no retrain needed)
> The final leakage-free blind evaluation uses the 24 fields with no exposure overlap between
> training/calibration and evaluation. The two contaminated fields (which share night-20250723
> `(visit,detector)` exposures with the dev set) were excluded from the headline metric and retained
> only in an audit table (`leakage_audit/leakage_audit.json`).

## If the all-26 number is needed — staged, gated, resource-conscious

**Stage A — FILTER the caches, do NOT rebuild from Butler.** *(verified feasible — CPU, minutes)*
The cached H5s are panel-indexed and the CSV `image_id` is a 1:1 panel index, so the leaked
`(visit,detector)` panels can be dropped directly — no Butler reads, no re-injection, no GPU
re-detection. Drop counts (from `leakage_audit/leakage_audit.json`): stage-1 train **8** of 1429
panels, stage-2 train **4** of 490, MF_LEN/threshold dev pool **99** of 13414 (filter the dev
*detections* used for the fit). Only rebuild from Butler if the cached pixels were themselves
contaminated — they are not (injection is on real diffims; dropping whole panels is sufficient).

**Stage B — train ONLY the winning recipe.** No A/B/Blow sweep, no architecture search. Stage-1
hard-positive oversampling (stk-balance 0.85), init from `models/seg_v1_trainable_init.pt`, the v2_D
recipe. Checkpoint at epochs 3/5/7/10; stop early if the faint-recall / product proxy plateaus.
(GPU ~4–6 h.)

**Stage C — cheap dev gate FIRST.** On a small leakage-free dev subset require: faint per-sighting
recall improves meaningfully, dets/panel does not explode, len/morphology distributions sane. **If it
does not reproduce the v2_D signal, STOP — no stage-2, no blind shot.** (CPU / short GPU.)

**Stage D — refit stage-2 + MF_LEN on SAMPLES (only if C passes).** MF_LEN is a 2-param fit: a
stratified sample (trail 6–60 px; SNR bins 2–5 / 5–10 / 10+; bands/seeing), field-held-out. Stage-2
on a controlled candidate sample, not every panel. (GPU ~2–4 h + CPU.)

**Stage E — ONE all-26 blind shot (only after C passes).** No threshold tuning, no post-blind edits.
(GPU ~8 h — the single biggest block.)

## Cost
With Stage A as a filter (not a rebuild), the heavy cost collapses to ≈ stage-1 (~5 h) + blind
(~8 h) ≈ **~13 h GPU** (plus stage-2 if the gate passes), Butler-free — versus the ~1–1.5 GPU-days
of a full from-Butler rebuild. Still gated: B/C must reproduce the signal before D/E spend anything.

Driver: `python -m ADCNN.pipelines.run_experiment` (stages train-stage1 → … → detect → report); the
`data` stage's leakage guard enforces exposure disjointness on whatever manifests are passed.
