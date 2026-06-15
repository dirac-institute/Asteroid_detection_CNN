# ADCNN evaluation protocol

Where each reported number comes from, and the rules for producing it. Two distinct evaluations —
do not conflate them.

## 1. Product result (the headline) — heliolinc alert chain
- **Metric:** faint-fast same-night 2-visit **alert completeness** and injection-set purity at the
  frozen op-point (`op_2v_alert.json`: S≥0.80, mf_snr≥5, chi2≤5, rate[1,8], top-50/night).
- **Where:** `python -m ADCNN.pipelines.run_experiment --stage report` (→ `regen_v2_report.py`),
  over the committed per-field pair caches on the DM-53195 **blind** fields.
- **Headline:** `3.64% → 10.33%` (+184%, clean-24 +192%), purity `86.1 → 88.6%` (held).
- **Baselines** (paper): Stack 5σ / Stack 4σ / ADCNN / Stack ∪ ADCNN — deduplicated union tables;
  ADCNN is a **complement** (adds stack-missed faint-fast), not a raw-recovery replacement.
- **Threshold selection** (frozen before the blind shot): S=0.80 sits on a completeness plateau;
  mfsnr=5 is set by the night-level top-50 alert budget — `Evaluation/threshold_selection_plots.py`,
  `ADCNN/pipelines/heliolinc/THRESHOLD_PROTOCOL.md` / `ALERT_SWEEP_DECISION.md`.

## 2. Detector diagnostic — the Evaluation notebooks (catalog-based)
- **Metric:** object-wise catalog recall + FP/panel vs the LSST 5σ/4σ/3σ stack, parameter recovery,
  completeness overlays. Pure analysis of pre-built catalogs (no NN inference in the notebook).
- `Evaluation/Evaluation.ipynb` — the **current** detector. It runs on the LEGACY DM-53881 `test`
  set, which is **out of domain** for the current model (domain-adapted to DM-53195), so standard
  recall here is *lower* than the prior baseline (0.603 vs 0.709 @ S≥0.80) — the expected cost of
  specialization, **not** a product regression. The notebook says so loudly and sources the product
  headline from `models/v2_D/v2_D_release.json`, not from this cross-domain recall.
- `Evaluation/Evaluation_legacy_v1.ipynb` — the prior v1.0 baseline (provenance / comparison),
  pinned to `legacy_v1` + `Evaluation/catalogs/`.
- `Evaluation/Evaluation_Real.ipynb` — real-sky evaluation.
- Defaults resolve from the active pipeline (`ADCNN/config.py`); no manual path/version edits.
  Regenerate: `jupyter nbconvert --to notebook --execute --inplace Evaluation/Evaluation*.ipynb`
  (always commit notebooks WITH outputs — the GitHub viewer errors on empty notebooks).

## Hygiene rules
- Never call injection-set purity "real-sky purity" — base-rate-corrected real-sky purity is low;
  the 2-visit product is an **alert** stream, not standalone discovery.
- Blind is eval-only: one pre-registered shot, **no** threshold tuning after results, outputs to
  separate `run_blind*` dirs, evidence caches preserved.
- The cross-domain detector recall is a diagnostic, never the product result.
