# Can a gradient-boosted reranker beat the promoted FT RandomForest?

**Verdict: NO-GO.** Boosted-tree rerankers (HistGradientBoosting, with hard-neg
mining / monotone constraints / calibration) produce a spectacular *in-sample*
FP@posR improvement that is a **metric-degeneracy + overfitting artifact**, not
a real second-stage gain. They do **not** justify a GPU re-dump. The promoted
FT RandomForest remains the right model.

All work is CPU-only, under `experiments/explore_reranker/` (gitignored), using
on-disk dumps. No tracked file modified, no GPU/SLURM, no commits.

---

## 1. Apples-to-apples table (same `_split`, same `_fp_gen`/`_pos_recall`)

Baseline reproduced **exactly** (`01_reproduce_baseline.py`): the FT RF trained
via `train_rf_v2` on the seed-0 2/3 panel-disjoint split reproduces the
`fp_fix2.txt` FT columns to ≤0.05 FP/CCD at every thr (match check: **YES**).

### 1a. Fixed-THRS table (the established metric, as in fp_fix2.txt)

genuine FP/CCD on the 50 held-out real-empty CCDs | synthetic posR (label_v2==1)

| thr  | FT RF (promoted) | RF_w8 | RF_deep | HGB | HGB_mono | HGB_hnm | HGB_mono_hnm | GB_sk |
|------|------------------|-------|---------|-----|----------|---------|--------------|-------|
| 0.05 | **68.9** / 1.000 | 30.0/1.000 | 25.1/1.000 | 1.8/1.000 | 4.0/1.000 | 0.9/1.000 | 1.6/1.000 | 253.8/1.000 |
| 0.10 | **37.8** / 1.000 | 11.1/0.999 | 14.3/1.000 | 1.2/1.000 | 2.3/1.000 | 0.6/1.000 | 1.2/1.000 | 148.8/1.000 |
| 0.20 | **18.2** / 1.000 | 3.3/0.981 | 7.2/1.000 | 0.8/1.000 | 1.4/1.000 | 0.5/1.000 | 0.7/1.000 | 82.1/0.998 |
| 0.30 | **11.7** / 1.000 | 1.4/0.941 | 4.2/1.000 | 0.6/1.000 | 1.1/1.000 | 0.4/1.000 | 0.6/1.000 | 55.2/0.992 |
| 0.50 | **5.6** / 0.999 | 0.3/0.765 | 1.3/1.000 | 0.4/1.000 | 0.8/1.000 | 0.3/1.000 | 0.4/1.000 | 30.0/0.953 |
| 0.70 | **1.4** / 0.746 | 0.0/0.567 | 0.4/0.693 | 0.3/1.000 | 0.4/1.000 | 0.3/1.000 | 0.2/1.000 | 15.5/0.869 |

`RF_balanced` (retrained here) == FT RF to 0.1 FP/CCD (identity sanity OK).
Isotonic-calibrated HGB at fixed THRS looks awful (396 FP/CCD @0.05) purely
because isotonic remaps the score scale — **proof the fixed-THRS table is not a
valid cross-model comparison** (different models put their boundary at
different numeric scores). Hence the calibration-free curve below.

### 1b. Calibration-free Pareto (`04_pareto_curve.py`) — the honest comparison

For each model, sweep its OWN score; at the cut that *just* holds the target
synthetic recall, read genuine FP/CCD on the disjoint held-out empties.

| target posR | FT RF: FP/CCD (AUC) | HGB: FP/CCD (AUC) | HGB_mono_hnm: FP/CCD |
|-------------|---------------------|-------------------|----------------------|
| 1.000 | **6.18**  (AUC 0.99792) | 0.04 (**AUC 1.00000**) | 0.04 |
| 0.990 | 3.90 | 0.02 | 0.04 |
| 0.950 | 3.06 | 0.02 | 0.02 |
| 0.900 | 2.74 | 0.02 | 0.02 |

On its face this is a ~150× FP reduction at posR=1.0. **It is not real.** See §2.

---

## 2. Is it a strictly better Pareto point? NO — it is an artifact

Three independent skeptical checks (`05_leakage_audit.py`,
`06_domain_features.py`, `03_oob_recall_check.py`) all converge:

**(a) Synthetic-pool AUC = exactly 1.00000.** Boosted trees *perfectly*
separate the 877 synthetic positives from negatives. Real noisy detectors never
hit 1.0 — this only happens when positives/negatives come from systematically
different generative processes.

**(b) The held-out-empty FP proxy is metric-degenerate for high-capacity
models.** A *domain classifier* trained only to tell "row from the synthetic
pool" vs "row from a real-empty CCD" (asteroid label ignored) gets **AUC =
1.0000** — and still **0.84** after dropping the 5 most domain-skewed features.
The synthetic-injection and real-empty candidate populations are perfectly
separable in the 72-D space. A 63-leaf boosted model wins the FP@posR metric by
keying on *"does this look like a synthetic-pipeline candidate"*, **not** on
*"is this a faint asteroid trail"*. `frac_real_label_overlap` is **not** the
leak (zeroing it changes nothing: FP 0.04→0.04); the separability is
distributed across many weak features, so the proxy cannot be "fixed" by
dropping a column.

**(c) What HGB keys on.** HGB permutation importance (syn-pool ROC) collapses
onto a *single* feature, `or_agg_max` (≈0.005; everything else ≈0). `or_agg_max`
alone separates synthetic-positives from real-empty candidates at **AUC 0.91**
(94% of syn positives have `or_agg_max>0.9` vs 52% of empties — clean injected
trails vs messy real residuals). The promoted RF spreads importance across
`or_agg_max, top5_mean_p, max_p, mean_p, lmf_flux_30, integrated_logit, …`;
bagging + depth-14 + balanced weights stop it collapsing onto the domain axis —
**which is exactly why the project's RF proxy is the trustworthy one and the
boosted model is not.**

**(d) The decisive test — panel-disjoint OOB recall** (`03`): scoring true
trails on *held-out panels* (honest generalization), HGB_mono's synthetic
recall **collapses from in-sample 1.000 to 0.665 → 0.539**. RF_balanced OOB
recall stays 0.96 → 0.88. The boosted model has memorized the synthetic
positives + the synthetic-vs-real split; out of fold it would **lose 33–46% of
real asteroid trails** — catastrophic for a recall-critical 2nd stage. The
in-sample posR≈1.0 in §1 is illusory; the baseline metric reads posR in-sample
(established convention), which masks this for boosted models but not for the
regularized RF.

Negative results stated plainly: hard-negative mining, monotone constraints,
and isotonic calibration **do not help** — they ride the same degenerate axis.
sklearn shallow `GB_sk` (depth-3, subsample) does *not* domain-overfit and is
correspondingly *worse* than RF (55–254 FP/CCD), confirming the "win" requires
high capacity to memorize the domain split.

---

## 3. Honest go/no-go

**NO-GO. Do not GPU-re-dump the real `test_real` asteroid panels for this.**

- The held-out-empty FP@posR proxy — validated and reliable for the
  *regularized RF* the pipeline ships — becomes **degenerate for high-capacity
  boosted models** because synthetic-injection vs real-empty candidates are
  perfectly separable independent of the asteroid label. Any "win" a GBM shows
  on it is the model exploiting that domain shift.
- The one honest generalization test we *can* run on disk (panel-disjoint OOB
  synthetic recall) shows the GBM **loses 33–46% of true trails out of fold**.
  Expected real-data effect of deploying it: **fewer** stack-missed asteroids
  recovered (a model that scores "looks synthetic" high will *reject* the
  faint, messy real trails that are the entire point), at the cost of a much
  larger objectwise-bar regression. Expected objects-gained vs the promoted FT
  RF: **negative**, not positive.
- Confirming objects-gained on real asteroid data would require the GPU
  re-dump of the real `test_real` panels (out of scope here). That re-dump is
  **not warranted**: the on-disk evidence (OOB recall collapse + domain
  AUC=1.0) predicts a regression, and the held-out-empty FP@posR proxy — the
  decision proxy used throughout this project — cannot validate a boosted
  model because it is degenerate for that model class.

**Recommendation:** keep the promoted FT RandomForest. The genuine remaining
lever is **not** a fancier classifier on these 72 features — it is reducing the
synthetic-vs-real domain gap (more realistic injections / real-background
training negatives) so the proxy stops being degenerate, *or* validating any
candidate directly via objects-gained on a real-asteroid GPU dump. Within the
current data/features, no reranker beats the promoted pipeline on a trustworthy
metric.

---

## Repro

```
cd <repo>; export PYTHONPATH=.
PY=/sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn/bin/python
$PY experiments/explore_reranker/01_reproduce_baseline.py   # baseline + _arrays.npz
$PY experiments/explore_reranker/02_model_zoo.py            # 10-model fixed-THRS table
$PY experiments/explore_reranker/03_oob_recall_check.py     # panel-disjoint OOB recall (decisive)
$PY experiments/explore_reranker/04_pareto_curve.py         # calibration-free FP@posR
$PY experiments/explore_reranker/05_leakage_audit.py        # fro-leak + domain probe
$PY experiments/explore_reranker/06_domain_features.py      # which features leak the domain
```

`harness.py` reuses `ADCNN.evaluation.fp_analysis` (`FEATS`, `EPS_GENUINE`,
`THRS`, `_split`, `_dedup`) and `ADCNN.inference.diffim_postproc_v2`
(`train_rf_v2`, `RF_FEATURES_V2`) so every number is directly comparable to
`fp_fix2.txt`. Artifacts: `_arrays.npz`, `_zoo_rows.json`, `_zoo_fimp.json`,
`_pareto.json`, `_leakage.json`, `_domain.json`, `_fimp.json`.
