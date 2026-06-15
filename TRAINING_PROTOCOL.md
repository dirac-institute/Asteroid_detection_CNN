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
- The fit IS code: `ADCNN/calibration/calibrate_mflen.py` matches injected sightings to detections
  (score≥0.80, 10 px) and OLS-fits `length_raw ~ slope·trail_length + offset`, then **confirms** the
  re-fit reproduces the frozen pipeline values (fail-loud on drift). Level-1 reads the committed
  `ADCNN/calibration/mflen_fit_pairs.csv`; Level-2 extracts it from the on-disk dev dirs
  (`--src/--inj/--out-csv`). Run as a `train_and_validate` stage (`--stage calibrate-mflen`); the
  apply step (recompute len_db on a detection run) stays `run_experiment --stage calibrate-mflen`.

## Threshold selection — the operating point is a FORMAL output, not a constant
- `ADCNN/calibration/threshold_selection.py` regenerates the validation completeness/purity curves
  from the committed 82-field per-pair caches and applies a **pre-declared decision rule**:
  - **score S — purity-floor:** lowest S whose in-sample purity at mfsnr≥5 is ≥ 75% → **S=0.80**
    (stable for any floor in ~(67%, 77%]). Documented-but-rejected framings: "largest S on the
    J-plateau" → 0.825, "completeness-knee" → 0.85 — neither yields 0.80; only purity-floor does.
  - **mfsnr — completeness-retention:** largest mfsnr retaining ≥80% of the uncut faint-fast
    completeness → **mfsnr=5** (stable for retention ~(0.73, 0.87]).
- It then **asserts** the selection equals the frozen `op_2v_alert.json` (tol 0). A disagreement is
  a FINDING surfaced to the user, never a knob to retune. Run as a `train_and_validate` stage
  (`--stage threshold-select`); `freeze` writes `thresholds.json` / `validation_report.json` /
  `threshold_sweep.csv` / `threshold_plots/` into the release dir.

## Validation vs blind
- **Validation** is where threshold selection, model selection, stage-2 calibration, MF_LEN fit,
  and dev gates happen. **Blind/test** is eval-only: a single pre-registered shot, no threshold
  tuning afterwards, outputs to separate `run_blind*` dirs, evidence caches preserved.

See also `EVALUATION_PROTOCOL.md`, `ADCNN/pipelines/heliolinc/THRESHOLD_PROTOCOL.md`.
