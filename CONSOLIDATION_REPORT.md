# Consolidation report — `heliolinc-discovery`

Promoted the ex-`v2_D` domain-adapted detector to the **single current/default** ADCNN pipeline,
behind one entry point, with the frozen science **unchanged**. No retraining, no threshold tuning,
no GPU re-run, no change to blind evidence.

## What changed

### 1. One config system + default flip (`ADCNN/config.py`)
- New `load_pipeline()` resolves the active pipeline from `models/<name>/pipeline.json`
  (`ADCNN_PIPELINE` selects; default `current`). A pipeline bundles model paths **and** the
  model-specific MF_LEN de-bias so they always travel together.
- `models/current/` (→ pointers into the frozen `models/v2_D/` release) + `pipeline.json`
  (MF_LEN 7.67/0.9425, provenance `adcnn-v2_D-rc1`). `models/legacy_v1/` (→ v1.0 files) +
  `pipeline.json` (33.4/0.887).
- Default `--seg-model`/`--cnn` and the MF_LEN de-bias in `catalog.py`, `discover_stream.py`,
  `make_eval_catalogs.py`, `stream_real_inference.py`, `evaluation/architecture.py` now resolve
  from the active pipeline. **No active Python default resolves to a v1 path** (only the explicit
  `legacy_v1`).

### 2. Single entry point (`ADCNN/pipelines/run_experiment.py`)
- `python -m ADCNN.pipelines.run_experiment --stage {data,train-stage1,train-stage2,
  calibrate-mflen,detect,alert-eval,report}` / `--stages all` / `--dry-run`. CPU stages run
  in-process; GPU/Butler stages print the exact `sbatch` command (SLURM backend
  `heliolinc/train_v2_D_e2e.sh` + `TRAIN_V2_D_E2E.md`), submitting only with `--submit`.

### 3. Exposure-level leakage guard (`ADCNN/pipelines/leakage_guard.py`)
- `assert_disjoint(train, blind)` raises `LeakageError` if any `(visit,detector)` appears in both
  training and blind/test inputs (tract-disjoint is not enough — the rc1 12-panel leak). Wired
  into the `data` stage.

### 4. Notebooks
- `Evaluation/Evaluation.ipynb` = canonical **current** pipeline (defaults → `models/current`
  + `Evaluation/catalogs_current/`), with a banner separating the **product headline**
  (3.64→10.33%, sourced from the release artifact) from the **cross-domain detector diagnostic**
  (legacy DM-53881 test set; recall lower by design). `Evaluation/Evaluation_legacy_v1.ipynb` =
  prior baseline pinned to `legacy_v1`. Both re-render from existing catalogs with **no env edits**
  (verified clean: current 0.603 / legacy 0.709 @ S≥0.80; zero tracebacks). The old
  `Evaluation_v2D_test.ipynb`/`.html` were folded into the canonical notebook (content also in
  `models/v2_D/v2_D_release.json` + `ADCNN_V2_RESULT.md`).

### 5. Tests + docs
- `pytest.ini` (root discovery) + `ADCNN/pipelines/tests/test_pipeline_config.py`: default→current
  + MF_LEN 7.67/0.9425, legacy→33.4/0.887, env selection/override, **frozen op-point golden
  values**, leakage guard (pass/raise/missing-col), entry-point stage table. **36 tests pass**
  (25 existing + 11 new).
- New: `REPRODUCE.md`, `TRAINING_PROTOCOL.md`, `EVALUATION_PROTOCOL.md`; updated root `README.md`
  (active pipeline + single entry point). One-line provenance banner added to the `ADCNN_V2_*` /
  `REPRODUCE_V2_D` / `TRAIN_V2_D_E2E` / `BLIND_TEST_REPORT` / `ALERT_SWEEP_DECISION` docs (kept).

## Science preserved (evidence)
- `op_2v_alert.json` / `link_op_point.json` / `op_multinight_discovery.json` **byte-unchanged**
  (md5 identical pre/post); `pipeline.json` references the op, not vice versa. Golden-value test pins them.
- `run_experiment --stage report` prints the headline identically: **3.64% → 10.33% (+184%)**,
  purity 86.1→88.6%.
- `models/v2_D/` untouched (md5s = immutable release identity).

## Remaining technical debt (deferred, non-blocking)
- The `ADCNN/pipelines/heliolinc/run_*/` sprawl (h2h, band, box, lambda, night, realfp, test2)
  is evidence/scratch from the research campaign; left in place (referenced by caches/docs). A
  future pass could archive the truly inert ones.
- `regen_v2_report.py` keeps its name (it is the prior-baseline-vs-current comparison generator —
  a historical-comparison tool, which is allowed to retain old labels).
- The frozen `op_*.json` `model_version`/`data_release` strings still describe the shared
  architecture + the DM-53881 calibration release (factually correct); intentionally not rewritten.
- FINAL (non-rc) tag still wants an exposure-disjoint clean retrain (the leakage guard now enforces
  this at build time); ecliptic-inclusive dev set; multi-night blind window — all pre-registered,
  separate from this consolidation.
