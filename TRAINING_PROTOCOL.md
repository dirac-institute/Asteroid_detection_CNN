# ADCNN training protocol

The rules every ADCNN training run must follow. Driver: `python -m ADCNN.pipelines.run_experiment`
(stages `data`, `train-stage1`, `train-stage2`, `calibrate-mflen`). Full SLURM recipe for the
current model: `ADCNN/pipelines/heliolinc/TRAIN_V2_D_E2E.md`.

## Training data — always synthetic trails on real diffims
- Supervised targets are **simulated asteroid/NEO trails injected into REAL Rubin/LSST difference
  images** (`inject_trails.add_trails` / re-subtraction). The background is always real.
- **No** pure-synthetic backgrounds. **No** real-asteroid labels as supervised targets (real
  asteroids are eval-only truth, never training targets).
- Faint-fast scope: detection-SNR 2–10, rate > 1°/day, trail ≥ 6 px is the science target; the
  injection magnitude/rate distributions cover it.

## Splits — grouped, not random; leakage-checked
- Split by **field/night**, not by random tile — random crops from the same exposure leak.
- Blind/test fields are disjoint from training at the **(visit, detector) exposure** level, not
  just by tract: adjacent tracts share boundary CCDs on a shared night (the rc1 12-panel leak).
- The `data` stage enforces this: `ADCNN/pipelines/leakage_guard.py:assert_disjoint(train, blind)`
  raises `LeakageError` if any `(visit,detector)` is shared. Pass `--train-manifests` /
  `--blind-manifests` to `run_experiment --stage data`.

## Stage 1 — segmentation (the win: hard-positive domain adaptation)
- `UNetResSEOrientHough` (widths [24,48,96,192,384], kernel_lens [11,21,41], n_angles 12),
  init from the v1 trainable checkpoint, low LR, oversample the stack-found/ADCNN-missed pool.
- Export to TorchScript → the pipeline's `segmentation` model.

## Stage 2 — cutout-CNN refit (REQUIRED after any stage-1 change)
- A changed stage-1 changes the seg channels the stage-2 CNN reads, so stage-2 MUST be refit on
  **leakage-clean** panels (disjoint from stage-1 training). Calibrate the retention threshold to
  the combined 5σ-stack ∪ ADCNN FP budget on validation.

## MF_LEN trail-length de-bias — REQUIRED after any stage-1 change
- A domain-adapted stage-1 has a different matched-filter "ends-bloom", so the de-bias
  `len_db = clip((raw_mf_length − offset)/slope, 0)` must be **re-fit** (field-held-out, on
  non-blind dev injections). It is **model-specific** and lives in the pipeline config so it
  travels with the model (mixing models and de-biases silently corrupts `len_db`).
- Fit by emitting raw length (`ADCNN_MF_LEN_OFFSET=0 ADCNN_MF_LEN_SLOPE=1`) and regressing raw vs
  truth length; write the result into `models/<pipeline>/pipeline.json`. Apply to a detection run
  with `run_experiment --stage calibrate-mflen` (uses the active pipeline's constants).

## Validation vs blind
- **Validation** is where threshold selection, model selection, stage-2 calibration, MF_LEN fit,
  and dev gates happen. **Blind/test** is eval-only: a single pre-registered shot, no threshold
  tuning afterwards, outputs to separate `run_blind*` dirs, evidence caches preserved.

See also `EVALUATION_PROTOCOL.md`, `ADCNN/pipelines/heliolinc/THRESHOLD_PROTOCOL.md`.
