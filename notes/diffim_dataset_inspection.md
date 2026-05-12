# Diffim dataset inspection (2026-05-11)

This is a read-only inspection of the dataset that already exists on disk at
`DATA_DIFFIM/`. It was produced by `ADCNN/data/dataset_creation/simulate_inject_diffim.py`
via `slurm_inject_diffim.sh` (train) and `slurm_inject_diffim_threshold.sh` (test).
Nothing was regenerated. The dataset is good enough to start Experiments 1 and 2 on
without rebuilding it.

The JSON dump backing this report is at
`experiments/diffim_runs/sanity/reports/dataset_inspection.json`.

---

## 1. Storage layout

```
DATA_DIFFIM/
├── train.h5         88 GB     800 panels
├── train.csv        4.3 MB    15,920 rows (one row per injected trail)
├── test.h5          2.7 GB     50 panels
├── test.csv         272 KB    1,000 rows
├── test_5sigma/test.{h5,csv}  same 50 panels, LSST stack detection at 5σ
├── test_4sigma/test.{h5,csv}  same 50 panels, LSST stack detection at 4σ
└── test_3sigma/test.{h5,csv}  same 50 panels, LSST stack detection at 3σ
```

The `test_Xsigma/` directories are **three threshold variants of the same test
set**, generated with the same seed but with `--stack-detection-threshold` set
to 5.0 / 4.0 / 3.0 respectively. Their `images` and `masks` are essentially
the same set of injected diffims (small numerical jitter aside) — what differs
is the LSST classical-detection flag (`stack_detection`) per injection. These
are exactly the right test artefacts for the LSST-threshold-sweep baseline.

`test.h5` (the unsuffixed one) and `test_5sigma/test.h5` were both built at 5σ;
treat the suffixed directories as the canonical test set going forward.

---

## 2. HDF5 schema (`train.h5` and all four `test*.h5`)

All four HDF5 files share the same three top-level datasets and the same
spatial shape `(H, W) = (4004, 4096)` (a single LSST CCD).

| dataset | shape (train) | dtype | chunks | semantics |
|---|---|---|---|---|
| `images` | (800, 4004, 4096) | float32 | (1, 128, 128) | **The injected difference image.** One channel only — diffim flux, not normalized. Zero-median by construction. |
| `masks` | (800, 4004, 4096) | bool | (1, 128, 128) | **Truth label.** Per injection, a drawn line of width `psf_width/2` joining the injection endpoints (computed from x, y, β, trail_length). It is the *analytic geometry*, **not** an empirical injected-minus-clean diffim residual. |
| `real_labels` | (800, 4004, 4096) | uint16 | (1, 128, 128) | **Footprints from `DetectAndMeasureTask` run on the clean (no-injection) diffim.** Marks real residual signals: variable stars, kernel mismatches, dipoles, subtraction artefacts. Pixel value = 1+footprint_index; treat as a boolean for now. |

`real_labels` is misleadingly named in the diffim build — it does **not** mark
real astrophysical sources in the science image. In the diffim, "real" means
"already present in the clean diffim before injection". The density is high:
sampled per-panel coverage ranges from ~10% to ~46%, which reflects diffim
subtraction artefacts and DIA-source footprints, not faint pre-injection stars.

### 2.1 Image statistics (sampled panels)

The `images` arrays are raw diffim flux. Sampled medians are essentially zero;
99th-percentile pixel values are typically ~60–100 (in diffim flux units),
while extreme outliers reach ±10⁴–10⁵ (bright-star residuals, subtraction
artefacts, saturation cores). The pipeline's old
`H5TiledDataset` clipped to `±5σ` around the **sky median**, which is wrong
on a zero-mean diffim. `ADCNN/data/diffim_norm.py` (already present in the
repo, unused so far) provides the correct three normalizations
(`normalize_diffim_mad`, `normalize_diffim_variance`, `normalize_variance_channel`).

### 2.2 Label statistics

| | train | test_5sigma | test_4sigma | test_3sigma |
|---|---|---|---|---|
| panels | 800 | 50 | 50 | 50 |
| frac positive pixels | 0.00122 | 0.00125 | 0.00124 | 0.00122 |
| mean positive pixels / panel | 20,076 | 20,498 | 20,334 | 19,947 |
| injections / panel (mean) | 19.9 | 20.0 | 20.0 | 20.0 |

Positive pixel fraction ≈ 1.2 e-3. This is ~5× sparser than typical
direct-image trail labels, as expected — only thin trail residuals remain
after template subtraction. Loss weighting must reflect this; the
direct-image `pos_weight` is too low.

---

## 3. CSV schema (`train.csv`, all four `test*.csv`)

22 columns, all `test*.csv` files share the same schema as `train.csv`:

| column | dtype | role |
|---|---|---|
| `ra`, `dec` | float64 | injection sky position |
| `x`, `y` | int64 | injection pixel position on the CCD |
| `trail_length`, `beta` | float64 | injection geometry (px, degrees) |
| `source_type` | object | always `"Trail"` |
| `mag`, `integrated_mag`, `PSF_mag` | float64 | injection brightness in three conventions |
| `SNR` | float64 | injection SNR target (since `--mag-mode snr`, this is the *commanded* SNR; range 2–8) |
| `SNR_estimation` | float64 | post-injection SNR estimate at the injection site |
| `m5_local`, `m5_detector` | float64 | local / detector-mean 5σ depth |
| `physical_filter` | object | u,g,r,i,z,y (all six bands present) |
| `visit`, `detector` | int64 | provenance |
| `image_id` | int64 | **HDF5 panel index** — joins to row `image_id` of the `images`/`masks` datasets |
| `stack_detection` | bool | **LSST classical detector recovered this injection** (`DetectAndMeasureTask` footprint overlapped the drawn-line truth by ≥ 1 pixel). Threshold = build-time `--stack-detection-threshold`. |
| `stack_mag`, `stack_mag_err`, `stack_snr` | float64 | photometry of the recovered footprint (NaN if not recovered) |

### 3.1 Train (`stack_detection_threshold=5.0`)

- 15,920 injections across 800 panels (≈ 20 injections per panel).
- 674 unique visits, 113 unique detectors, **796 unique (visit, detector) pairs**
  packed into 800 panels — i.e. some (visit, detector) re-injected, most unique.
- All 6 bands appear (u, g, r, i, z, y).
- SNR range 2.0 – 8.0, median 5.0 (uniform between 2 and 8 as commanded).
- Trail length 6 – 60 px, median 33 px.
- β 0 – 180°, median 90°.
- **LSST 5σ recovery rate: 8,039 / 15,920 ≈ 50.5 %.** This is the headline
  classical baseline; the NN is competing against this on (~50 % of the
  injections) the LSST-missed remainder.

### 3.2 Test thresholds — recovery rate sweep

LSST classical recovery rate per injection (same 1,000 injections per test split):

| split | `stack_detection=True` | recall |
|---|---|---|
| test_5sigma | 475 / 1000 | 47.5 % |
| test_4sigma | 682 / 1000 | 68.2 % |
| test_3sigma | 879 / 1000 | 87.9 % |

This is the classical-completeness ceiling the NN has to **complement**, not
replace. Concretely:

- ~52 % of injections in the test set are missed by LSST 5σ → that is the
  "LSST-missed-but-injected" pool the NN should recover.
- Dropping the classical threshold to 3σ already recovers 88 %, so the
  scientific bar is "NN must add a meaningful number of LSST-5σ-missed
  recoveries **at a smaller candidate burden than what you would pay by
  lowering the stack threshold to 3σ**". This is the test that the
  direct-image NN failed.
- Note: at 3σ, `stack_snr` ranges down to 0.47 — the threshold isn't strict
  on photometric SNR, it's on detection-image SNR. Be careful interpreting
  it as "SNR ≥ 3" in the photometric sense.

---

## 4. Train/test split — leakage check

| | count |
|---|---|
| Train visits | 674 |
| Test visits | 50 |
| **Visits in both train and test** | **8** |
| Train (visit, detector) pairs | 796 |
| Test (visit, detector) pairs | 50 |
| **(visit, detector) pairs in both** | **0** |

The split is by **(visit, detector)** rather than by **visit**. No
(visit, detector) leaks between train and test, but 8 visits do appear in
both (with different detectors). This is acceptable in practice — different
CCDs see different sky and have different per-detector systematics — but it
is **not** the visit-disjoint split recommended by the plan, and we should
say so explicitly in any final write-up. For Experiment 1's pilot scope it
is fine; for the final result, build a strictly visit-disjoint test set.

---

## 5. Critical gotchas discovered during inspection

1. **Truth is geometric, not empirical.** `masks` is the drawn-line label,
   not the post-subtraction injected residual. The plan recommended the
   empirical (`diffim_injected − diffim_clean > τ·σ_diff`) construction.
   Consequences:
   - Low-SNR injections whose residual is buried in noise still have positive
     truth pixels. The NN will be penalized for failing to see them, even
     when the residual is physically below noise.
   - Trail width = `psf_width / 2`, which is narrower than the actual smeared
     residual on the diffim. The NN will be penalized at the edges of real
     residuals where they fall outside the geometric line.
   - This is fixable post-hoc by **gating the truth mask with the diffim
     variance plane at training time**, but we do not have the variance
     plane saved.
2. **No variance / mask channel saved.** Only `images` (the diffim) is on
   disk. `diffim_norm.normalize_diffim_variance` cannot be used as-is; we
   are limited to `normalize_diffim_mad` (whole-panel MAD around zero) for
   the input channel until/unless we extend the builder to also persist the
   diffim's variance plane.
3. **`real_labels` is dense.** 10–46 % of pixels per panel are flagged.
   Most of this is subtraction-artefact / DIA-source footprint, not real
   astrophysics. In a diffim setting this becomes the *ignore-mask* — pixels
   where the NN may fire on legitimately interesting non-injected signal
   and should not be marked as false positives.
4. **Class imbalance.** Mean positive fraction is 1.22 × 10⁻³. With 128×128
   tiles that's ~20 positive pixels per tile and many tiles fully negative.
   `pos_weight` needs to be retuned (current direct-image config is too
   small) and a stratified-by-tile-positives sampler is essential.
5. **Train/test split is by (visit, detector), not by visit.** 8 visits are
   shared between splits with different detectors. Document.
6. **`SNR` vs `SNR_estimation` vs `stack_snr`** are three different things:
   commanded SNR at injection time, in-place SNR after injection, and PSF
   photometry SNR of the matched LSST source footprint. Use `SNR` and
   `SNR_estimation` for binning by injection brightness; use
   `stack_detection` and `stack_snr` for the LSST baseline.

---

## 6. What is immediately usable

- All ~800 train panels and 50 test panels (× 3 thresholds) are well-formed.
- LSST 5σ / 4σ / 3σ baselines are precomputed per-injection (`stack_detection`,
  `stack_snr`) — no need to re-run any LSST detector for the comparison.
- The test threshold sweep is the right artefact for the
  "NN ∪ stack-Xσ" complementarity plots.
- `ADCNN/data/diffim_norm.py` is ready for the input-side normalization.
- `ADCNN/core/model.py` (`UNetResSE`, `UNetResSEASPP`) needs no changes for
  single-channel input; it already takes `in_ch`.

## 7. What is NOT yet usable

- A diffim-aware dataloader (current `H5TiledDataset` does sky-median MAD;
  must replace with zero-centered MAD or pixel-SNR).
- A candidate-extraction module (two-stage: t_low for geometry + score_thr
  for confidence; emit per-candidate metadata).
- Object-level evaluation that ingests `stack_detection` and produces the
  LSST-only / NN-only / union recovery curves.
- Visual panel scripts (covered next in this work session).

---

## 8. Recommended immediate sequence

1. Generate sanity panels (this session).
2. Add `DiffimH5Dataset` that reads `images` / `masks` / `real_labels`, applies
   `normalize_diffim_mad`, exposes `real_labels` as an ignore-mask channel and
   to the loss for don't-penalize logic.
3. Pilot Experiment 1: train `UNetResSE` (`in_ch=1`) on a subset of train
   panels for a few hours on 1 GPU. Use masked BCE with retuned pos_weight.
   Eval on `test_5sigma` only.
4. Pilot Experiment 2: candidate extraction + matched object-level metrics
   producing the four headline numbers (LSST-only / NN-only / union / NN
   recovery of LSST-missed) and a FROC curve.
5. Then re-run eval against `test_4sigma` and `test_3sigma` to produce the
   three-threshold sweep.

Only after these four steps do we have a defensible answer to the
"does NN add scientific value over LSST 5σ on diffims?" question.
