# ADCNN release / reproducibility checklist

Two reproducibility levels (run from a clean checkout in the `asteroid_cnn` conda env):

- **Level 1 — reproduce the reported result (mandatory for the paper).** Frozen model + frozen
  configs + frozen eval caches; one command regenerates the CLEAN-24 numbers, the integrity checks,
  and the Evaluation notebooks/HTML:
  ```bash
  python -m ADCNN.pipelines.run_experiment --stages report,evaluation-notebooks
  ```
  Produces: (1) leakage audit confined to blind fields 0,1; (2) active pipeline = `current`;
  (3) model md5s verified vs `models/v2_D/v2_D_release.json`; (4) MF_LEN 7.67/0.9425 verified;
  (5) frozen `op_2v_alert` golden values verified; (6) CLEAN-24 blind verdict regenerated;
  (7) `Evaluation/Evaluation.ipynb` + `.html` (and the legacy appendix) regenerated; (8) final
  result table written to `Evaluation/results.json`. The report stage **fails loud** on any drift.

- **Level 2 — reproduce the training (documented, expensive, not run every time).** The full
  train → calibrate → evaluate chain:
  ```bash
  python -m ADCNN.pipelines.run_experiment --stages all          # GPU stages print sbatch; --submit to run
  ```
  GPU/Butler stages print the exact `sbatch` (backend `heliolinc/train_v2_D_e2e.sh` +
  `TRAIN_V2_D_E2E.md`). The exposure-disjoint clean retrain is the OPTIONAL hardening variant —
  `heliolinc/CLEAN_RETRAIN_PLAN.md` (staged/gated; Stage A filters cached panels, no Butler rebuild).

## Clean-checkout verification (what was run)

1. **Fresh checkout** — `git worktree` at the release commit (symlinks + caches only; no 14 GB h5).
2. **Report regeneration** — `run_experiment --stage report` → CLEAN-24 `3.68 → 10.74%` (+192%).
3. **Notebook regeneration** — `--stage evaluation-notebooks` renders both notebooks + HTML (the
   stack-FP counts come from the committed `Evaluation/catalogs_*/stack_fp_counts.json`, so the
   14 GB `DATA/test.h5` is NOT required on a clean checkout; the h5 is a fallback for full rebuilds).
4. **Exact numbers** — current 0.603 recall @ S≥0.80 (cross-domain detector diagnostic); CLEAN-24
   blind 10.74% / +192%; ALL-26 10.33% / +184% (flagged not-strictly-blind).
5. **Leakage artifacts** — `ADCNN/pipelines/heliolinc/leakage_audit/leakage_audit.json` produced;
   contamination confined to blind fields 0,1.
6. **Active defaults never touch legacy** — clean-env `load_pipeline()` → `models/current/pipeline.json`;
   no active default resolves to `legacy_v1` or old MF_LEN (only explicit selection).
7. **Notebooks need no manual edits** — defaults resolve from the active pipeline + `catalogs_current`.
8. **Full training command documented** — `--stages all` (Level 2) + `TRAIN_V2_D_E2E.md`.
9. **All model/config md5s recorded** — `models/v2_D/v2_D_release.json`; verified by the report stage.
10. **Reproducible release candidate tagged** — see the tag below.

## Frozen state (release identity)
- Models: `models/v2_D/` (md5s in `v2_D_release.json`); `models/current/` points into it.
- De-bias: MF_LEN 7.67 / 0.9425 (in `models/current/pipeline.json`).
- Frozen op: `op_2v_alert.json` (S≥0.80, mfsnr≥5, chi2≤5, rate[1,8], top-50) — golden-value tested.
- Result: `Evaluation/results.json`; headline CLEAN-24 `3.68 → 10.74%` (+192%), purity 86.0→88.5%.
- 36 unit tests pass (`pytest`); NY2 linker regression anchor recovered.
