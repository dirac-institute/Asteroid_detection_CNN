# Difference-Image Training — Direction Plan

Branch: `experiment/diffim-dataset`
Repo: `/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN`
Environment: SLAC/SDF Rubin, LSST stack `w_2026_09`, Butler repo `dp2_prep` (`/sdf/group/rubin/repo/dp2_prep`)

This document is the output of the first-phase investigation and design work for a pivot
from direct-science-image training to difference-image (diffim) training. It does **not**
describe finished code. It is the reference for the first real implementation steps.

---

## 1. Problem statement

### 1.1 Why direct-image training is running out of headroom
The current pipeline trains a U-Net-family segmentation network on single-visit
`preliminary_visit_image` (PVI) data with synthetic asteroid trails injected via
`lsst.source.injection.ExposureInjectTask` and a binary trail-pixel mask as the label.
The scientifically interesting result so far is:

- The NN contains real detection signal: it recovers trails that the classical
  5σ `SingleFrameDetectAndMeasure` / stack threshold misses.
- That gain shrinks substantially once the classical stack threshold is lowered to
  4σ or 3σ — i.e. most of the extra objects are also findable classically if one is
  willing to tolerate the false-positive load of a lower threshold.
- The direction has therefore reached a strategic ceiling: the problem formulation
  (detect faint trails in a single calibrated science image) rewards the network for
  doing what a threshold scan can almost do, not for doing something qualitatively new.

### 1.2 Why difference images are the right next step
The scientifically relevant downstream product for moving-object detection on Rubin is
the *difference image* (science minus PSF-matched template). It differs from direct
images in ways that matter:

- All static sky structure (stars, galaxies, fixed diffuse emission) is subtracted.
  Trails — which are by definition not in the template — appear as almost-pure
  residuals against a noise-dominated background.
- The signal-to-distractor ratio for moving objects is vastly higher than on PVI,
  because there is no population of real point-source / galaxy footprints to compete
  with. The remaining structure is: new transients, subtraction artefacts
  (dipoles, CR ghosts, bad masking), and noise.
- This aligns the project with Rubin Alert Production (AP): AP's entire job is to
  detect transients and moving objects on difference images and emit `dia_source`
  records. A trail detector operating on diffims is directly comparable to AP's
  detection performance and naturally extends it toward streaks, which AP currently
  under-catches.
- The DRP `preliminary_visit_image` / single-frame catalogues the previous phase used
  are *calibration products*, not a transient-detection source — a point that caused
  confusion earlier. Moving to diffims removes that confusion by construction.

### 1.3 What the new direction is, in one sentence
Train a network whose input is a Rubin difference-image-family product (with its
variance and mask planes as auxiliary channels) and whose label describes the
injected moving-object residual in diffim space. Evaluate completeness, false-positive
burden and comparison to `dia_source` at the level of recovered objects, not pixels.

---

## 2. Current codebase inventory

All paths are relative to the repo root unless absolute.

### 2.1 What currently exists and what is tied to direct images

| Area | File(s) | Role | Direct-image-specific? |
|---|---|---|---|
| **Injection entry point** | `ADCNN/data/dataset_creation/simulate_inject_fill_deterministic.py` (primary), `simulate_inject.py` (legacy) | Fetch PVI, inject trails, save HDF5 panels + CSV | **Yes.** Reads `preliminary_visit_image` + `_background`; uses `ExposureInjectTask` on the PVI directly; computes a pre-injection source mask via `SingleFrameDetectAndMeasureTask` at 5σ on PVI. |
| **Pipetask wrappers** | `ADCNN/data/dataset_creation/pipetasks.py` | `isr()`, `calibrate()`, `fetch_from_butler()`, `source_detect()` | Mostly direct-image. `source_detect()` is specialized to the PVI → stars path; `calibrate()` is the CalibrateImage path. **Reusable but narrow.** |
| **Trail drawing** | `ADCNN/data/dataset_creation/common.py` | Geometry: `draw_one_line`, SNR↔mag conversions via `getPhotoCalib` | Line geometry is reusable. SNR/mag conversions use the **science-image** `photoCalib` and variance — not directly applicable to diffims. |
| **Ephemeris / real-mover injection** | `real_ephemerides.py`, `slurm_ephem.sh` | Real Solar-System orbits for realistic injection | Reusable — geometry happens pre-injection. |
| **SLURM launcher** | `slurm_inject.sh` | Sources `w_2026_09`, runs deterministic fill | Reusable pattern; command line must change. |
| **Storage format** | HDF5 `images` / `masks` / `real_labels` (all `(N, H, W)`, N tiles ≈ 4072×4072) + `train.csv` / `test.csv` metadata | Large mmap-friendly panels with per-panel truth | Format is fine for diffims, but the single image plane will need to become a multi-plane tensor (science, diffim, variance, mask, …). |
| **Dataset** | `ADCNN/data/datasets.py` (`H5TiledDataset`, `robust_stats_mad`) | HDF5 → 128×128 tiles, per-panel MAD normalization, ±5σ clip | **Normalization is tied to PVI.** Sky-dominated median + MAD is wrong for a zero-mean diffim. Needs replacement. |
| **Soft masks** | `ADCNN/data/soft_masks.py`, `ADCNN/scripts/precompute_soft_masks.py` | Gaussian-blurred centerline masks from CSV truth | Reusable — truth geometry is the same in diffim space as long as the injected trail is wholly new. |
| **Model** | `ADCNN/core/model.py` (`UNetResSE`, `UNetResSEASPP`) | 1-in/1-out U-Net with residual + SE blocks | Reusable. Only `in_ch` changes to accept multi-plane diffim tensors. |
| **Losses** | `ADCNN/core/losses.py` | masked BCE, Dice, Focal-Tversky, ASL | Reusable. pos_weight and class balance must be re-tuned for diffim class distribution. |
| **Training loop** | `ADCNN/train.py`, `ADCNN/main.py`, `ADCNN/training/*`, SLURM in `ADCNN/scripts/slurm_train_frontier_batch.sh` | DDP 4-GPU, EMA, stratified sampler, rescue validation | Reusable. Stratified-sampler bucket definitions (`frac_missed` / `frac_detected` / `frac_background`) need rethinking because the "detected" bucket is defined against the **stack detection on the science image**, not diffim detection. |
| **Evaluation** | `ADCNN/evaluation/*`, `ADCNN/inference/*`, notebooks in `Notebooks/` | Pixel ROC, connected-component → object match, comparison to stack detections | Pixel path is reusable. Object-level comparison target must change: instead of `SingleFrameDetectAndMeasure` catalog, compare to `dia_source` / `fakes_dia_source`. |
| **Real-label handling** | `core/config.py:105–112`, mask plane `real_labels` | Down-weight / ignore real pre-injection source pixels | **Most of this becomes unnecessary on diffims** — static sources are already subtracted. What remains is subtraction-artefact pixels, which have a different failure mode. |

### 2.2 What is **reusable with zero or minor change**
- U-Net model code (change `in_ch`).
- Loss library and EMA.
- Trail geometry (`draw_one_line`), soft-mask generator.
- SLURM job structure, conda env (`asteroid_cnn`), `w_2026_09` loadLSST.
- HDF5 tile-streaming dataset pattern (layout changes; streaming logic is fine).
- Stratified sampler and DDP plumbing.
- Evaluation connected-component matcher (geometry is the same).

### 2.3 What must be replaced or abstracted
- **Data fetch**: replace `fetch_from_butler` with a diffim fetcher that pulls
  `difference_image` + variance + mask + (optionally) `visit_image` + `template_image`
  + per-detector `dia_source_detector`.
- **Normalization**: replace `robust_stats_mad` with a zero-centered, variance-aware
  scheme. Diffims are already roughly zero-mean; the correct scale is
  `sqrt(variance)` (pixel-wise) or a MAD on the full panel that does not assume
  positive sky. Clip symmetrically.
- **Pre-injection forbidden-mask strategy**: on a diffim, "pre-existing real sources"
  are not the blocker. What to avoid is injecting on top of known subtraction-artefact
  regions (mask plane bits, edge regions, saturation residuals). The `real_labels`
  panel becomes a **DIA-source proximity mask** instead.
- **Truth label**: for a newly injected moving source that is **not** in the template,
  the diffim residual is a positive trail whose geometry is the same injection
  polyline. The pixelwise mask is therefore unchanged in principle, but **only the
  pixels where the convolved-injected-flux minus 0-template exceeds a noise threshold
  should count as positive**. Labels should be generated from the injected
  post-injection minus pre-injection diffim at a threshold of a few σ of the diffim's
  pixel variance, not from the bare geometry line.
- **Stratified-sampler buckets**: switch the "detected-in-stack" bucket to
  "detected-in-`fakes_dia_source`" or to a simple SNR threshold in diffim space.
- **Evaluation baseline**: the classical competitor is `dia_source_detector`
  (AP-style 5σ on the diffim) with or without the trailed-source plugin
  `ext_trailedSources_*`. Comparing to `single_visit_star` is no longer meaningful.

---

## 3. Candidate diffim workflows

Four concrete workflows were considered. Each is rated on realism (how close to
production science), compute cost, implementation complexity, and whether truth
is cleanly defined.

### Candidate A — Train on existing `difference_image` products (no injection)
Use the `difference_image` and `dia_source_detector` outputs of the DP2 stage3
DRP run as-is. Label pixels as positive if they belong to a `dia_source` footprint
whose centroid matches a known moving-object orbit (Sorcha cache at
`/sdf/group/rubin/repo/dp2_prep/sorcha_cache` or MPC orbits at `.../mpcorb`).

- **Benefits**: zero extra compute for image generation; every input is a real
  product; results directly generalize to real AP.
- **Drawbacks**: truth is now external-catalog-driven. Recall metrics are limited
  to what MPC / Sorcha gives. Most real DP2 diffims contain few or no asteroids
  per detector, so the positive-class density is extremely low and the problem
  is dominated by artefacts. No control over SNR / trail-length distribution.
- **Realism**: highest (real diffims, real noise, real artefacts).
- **Compute**: minimal (read-only butler access).
- **Complexity**: medium — needs an orbit→pixel projection module and careful
  handling of trail length (visit exposure time × apparent motion).
- **Truth**: noisy. Good **evaluation** set, bad sole **training** set.

### Candidate B — Inject synthetic moving-object catalogues into `post_isr_image`, then re-run the DRP/AP subtraction
Use `lsst.source.injection` through the supported pipetasks
(`inject_visit` / `injectExposure` / `injectVisit`) to insert trails into the
`post_isr_image`, let the **existing** `calibrateImage` + image-subtraction
pipeline run, and harvest the resulting `fakes_difference_image` and
`fakes_dia_source` products.

- **Benefits**: this is the officially supported injection route. The vocabulary
  (`injection_catalog`, `injected_post_isr_image`, `fakes_difference_image`,
  `fakes_dia_source`, `assocDiaFakesDetectorVisitCore_metrics`) is already
  registered on this stack and the analysis side (`analyzeAssocDiaFakes*`,
  `matchObjectToInjected*`) is wired up. The residual in the fakes diffim is the
  **physically correct** diffim residual of the injected trail, including
  subtraction artefacts from PSF-matching, at the same cost as the real pipeline.
- **Drawbacks**: most expensive path — requires running subtraction. For a trail
  injected on-visit against a coadd template there is no "old position" of the
  asteroid to worry about (coadds average sidereal positions across many nights
  so any one moving object is diluted); this simplifies the residual to "injected
  trail minus (small) background".
- **Realism**: very high. Produces artefacts and mask bits identical to production.
- **Compute**: non-trivial. One fakes diffim per `(visit, detector)` requires
  one subtraction task + associated DRP prerequisites (template warp, PSF
  matching). Target cadence ≈ minutes per detector on one CPU. SLURM-friendly,
  embarrassingly parallel.
- **Complexity**: medium. No new image math to write — we drive existing
  pipetasks. The work is mostly pipeline wiring and butler output collections.
- **Truth**: clean. The injection catalogue defines ground truth in pixel space
  (centre, length, angle, flux) and the residual mask can be derived either
  analytically from the injection catalog or empirically by differencing two
  runs (with / without injection) of the same visit.

### Candidate C — Inject synthetic moving-object catalogues into `preliminary_visit_image`, then hand-subtract against the coadd template
Same spirit as (B), but skip the full DRP replay: pull `preliminary_visit_image`
and `template_coadd`, subtract the (PSF-matched) template from the injected PVI,
and use the resulting residual as the NN input.

- **Benefits**: cheaper than (B). Produces a diffim without needing to drive the
  full AP pipeline.
- **Drawbacks**: reimplements PSF matching / warping / scaling with in-repo code.
  Unless `lsst.ip.diffim`/`AlardLuptonSubtractTask` is called correctly, the
  noise properties of the synthetic diffim will diverge from production's
  `difference_image`. If we do call the stack subtraction, this collapses into
  (B) minus the last detect-on-diffim step.
- **Realism**: medium. Correct if we use `AlardLuptonPreconvolveSubtractTask` on
  real templates; degraded if we bypass it.
- **Compute**: lower than (B). Still per-visit per-detector.
- **Complexity**: **higher** than (B) if done right — PSF matching is its own
  rabbit hole. Lower than (B) if we cheat and do a straight subtraction, but
  then realism suffers.
- **Truth**: clean (same as B).

### Candidate D — Inject the trail image *directly into the diffim*
Take a real `difference_image`, add a PSF-convolved synthetic trail at a chosen
SNR, and treat that as both input and truth.

- **Benefits**: trivial to implement; fastest iteration.
- **Drawbacks**: non-physical. The real diffim already contains any residual
  asteroids, real subtraction artefacts, and template imperfections; we pile a
  synthetic trail on top. SNR in diffim space is defined against the **variance
  plane of the diffim**, which we respect, but we skip all of the PSF-matching
  physics. We also cannot measure realistic subtraction-artefact false positives
  because we are not subjecting the fake to them.
- **Realism**: low for training-side claims; fine for quick sanity checks.
- **Compute**: negligible.
- **Complexity**: trivial.
- **Truth**: trivial.

### Candidate summary

| Candidate | Realism | Compute | Complexity | Truth quality | Role |
|---|---|---|---|---|---|
| A (real diffims, orbit truth) | **Highest** | Low | Medium | Weak | Evaluation set |
| B (inject at post-ISR, re-run subtraction) | High | Medium-high | Medium | Clean | **Main training source** |
| C (inject at PVI, hand-subtract) | Medium | Medium | Medium-high | Clean | Fallback if (B) too expensive |
| D (inject directly into diffim) | Low | Trivial | Trivial | Clean | Sanity check / loss-curve debugging |

---

## 4. Recommended workflow

**Use Candidate B as the production generator, Candidate D as a smoke-test generator, and Candidate A as the ultimate evaluation overlay.**

Concretely:

1. **Primary training data**: inject trails at the `post_isr_image` level using
   `lsst.source.injection`'s `injectExposure` / `injectVisit` tasks, then drive
   the stack's image-subtraction through to `fakes_difference_image` and
   `fakes_dia_source`. Tile the resulting diffim into HDF5 panels in the same
   layout already used for direct images, but with more channels.
2. **Smoke-test dataset (Candidate D)**: a one-hour pilot that takes
   a few hundred real `difference_image` patches, adds synthetic trails, and
   trains a small U-Net. This is there to prove that the NN pipeline, loss,
   and evaluation still converge in diffim space before we commit SLURM hours to
   the real generator.
3. **Scientific evaluation overlay (Candidate A)**: on a held-out set of real
   DP2 `difference_image` + `dia_source_detector` products, run both the NN and
   AP's native detection, and compare object-level recall/FPR against an
   MPC/Sorcha-derived truth catalogue.

### 4.1 Justification
- **Truth quality**: (B) gives per-pixel, per-object truth that is derivable
  directly from the injection catalogue. That makes the loss well-defined and
  keeps the pixelwise + object-level evaluation consistent.
- **Realism**: (B) preserves all diffim-specific noise and artefact structure
  because the trail passes through the real subtraction. This is what lets us
  claim the final NN will transfer to real AP diffims.
- **Existing infrastructure**: the stack already provides the vocabulary
  (`fakes_*`, `injectedMatchDiaSrc_*`, `analyzeAssocDiaFakes*`). We do not need
  to write the image math.
- **Cost control**: SLURM-parallelizable per `(visit, detector)`; we can cap the
  initial run to ~100–200 visits and a subset of detectors to build the pilot.
- **Fallback**: (D) gets us through week-1 development. (C) is the fallback if
  (B) turns out to require storage we cannot support.

---

## 5. Data specification

### 5.1 Input tensor
Per tile `(H, W) = (128, 128)` (same tile size the current pipeline uses),
as a stack of channels:

1. **`diffim`** — the `difference_image` pixel values.
   Normalization: per-tile zero-centered clip at `±k · σ_diff` where `σ_diff` is
   the median of `sqrt(variance_plane)` over a central crop. `k=5`.
2. **`diffim_var`** — the variance plane of the diffim.
   Transform: `log1p(variance / median_variance)` (compresses the dynamic range
   while preserving spatial structure).
3. **`diffim_mask`** — a bitfield reduced to a single-channel float.
   Concretely, build a binary layer `bad = any(mask & (EDGE | BAD | NO_DATA | SAT | CR)) != 0`,
   float32. This tells the NN where pixels are unreliable.
4. (Optional) **`science`** — the `preliminary_visit_image` (or `visit_image`),
   MAD-normalized as the current pipeline does. Same injection shares WCS with
   the diffim, so no resampling needed. Useful because the NN can learn to use
   the science image as a soft prior for "is there local structure".
5. (Optional) **`template`** — the warped `template_coadd` patch used in the
   subtraction, with the same normalization as `science`.

Start with channels 1–3. Channels 4–5 are ablation-only and may be toggled on
with a config flag.

### 5.2 Label
Pixelwise binary mask, `(H, W)` float32 in `[0, 1]`, defined as:

> a pixel is positive iff the **injected-only** signal at that pixel, in diffim
> space (i.e., same PSF-matching convolution as applied to the science image),
> exceeds `τ · σ_diff` where `τ = 2` by default.

Two ways to build this:

- **Analytic**: re-project the injection catalogue geometry, convolve with the
  PVI PSF, scale by the subtraction normalization, and threshold at `τ · σ_diff`.
  This is fast but approximate.
- **Empirical**: run the same visit through subtraction *with* injection and
  *without* injection, and take `diffim_with - diffim_without` as the clean
  injected residual. Threshold at `τ · σ_diff`. This is exact but doubles the
  compute per visit.

Start with the empirical path for the pilot (because doubled compute is fine on
O(100) visits). Move to the analytic path for the full-scale generator once its
accuracy is validated against the empirical one.

The soft-mask channel (from `ADCNN/data/soft_masks.py`) remains available and
is recommended for training — the hard label above is the ground-truth
reference for evaluation.

### 5.3 Metadata (CSV, one row per injected object)

- `panel_id`, `visit`, `detector`, `band`, `physical_filter`, `day_obs`
- `x`, `y`, `trail_length_px`, `beta_deg` — injection geometry
- `inject_mag`, `inject_snr_diffim` — input SNR in **diffim** space, computed as
  `injected_integrated_flux / sqrt(sum(psf^2 * variance_plane))` at the
  injection site
- `ap_detected` — did `fakes_dia_source` (or `dia_source_detector` on the
  `fakes_difference_image`) recover this injection (via
  `injectedMatchDiaSrc`-style matcher or a simple 2-pixel footprint overlap)
- `ap_snr` — flux/flux_err from the recovered `diaSrc` if matched
- `classical_diffim_detected` — same as `ap_detected` but at a reduced (3σ)
  detection threshold, for the 3σ/4σ/5σ sweep
- `inside_bad_mask`, `inside_edge` — whether any of the injection pixels
  overlap unreliable mask planes

### 5.4 Panel layout and storage

- HDF5 file per split (`train.h5`, `test.h5`, `val.h5`).
- Datasets: `diffim`, `diffim_var`, `diffim_mask`, `masks` (truth),
  `dia_source_pixels` (precomputed AP-detection mask, for rescue-validation
  comparison). All shape `(N, H_panel, W_panel) = (N, 4072, 4072)`. `float32`
  for images, `bool`/`uint8` for masks.
- Chunk size `128×128` in the spatial dims (matches tile size).
- `gzip`/`lzf` compression is usually a loss on our I/O pattern — leave panels
  uncompressed; we read random tiles.
- Metadata CSV: one row per object, one row-group per panel.

### 5.5 Train/val/test split and stratification

- **Primary split**: by `(visit, detector)`. A visit×detector tuple is in exactly
  one split. This prevents leakage of per-CCD systematics.
- **Target sizes**: for the pilot, `train ≈ 160 visit×detector`,
  `val ≈ 20`, `test ≈ 20`. Full-scale later: ~2000 / 250 / 250.
- **Stratification buckets** (used by the sampler, not the split):
  - by `inject_snr_diffim` in {<3, 3–5, 5–8, >8}
  - by `trail_length_px` in {6–12, 12–30, 30–60, >60}
  - by `ap_detected` ∈ {0, 1} — this is the "missed-by-AP" bucket that drives
    training focus
- Ensure every `band` appears in every split.

---

## 6. Training proposal

### 6.1 Model family
- **Keep the U-Net family.** `UNetResSE` with `in_ch = 3` (diffim, variance,
  mask) is the first real target. `UNetResSEASPP` is a drop-in upgrade.
- Output stays single-channel logits over pixels.
- Expect `in_ch = 5` for the ablation that adds science + template channels.

### 6.2 Task framing
- **Keep segmentation** as the primary output.
- Add a **"candidate scoring" post-processing** head on top of connected
  components: for each connected component in the thresholded segmentation,
  compute (component integrated logit, component length, component area,
  aspect ratio, max-pixel, diffim-variance at component centroid). Use these as
  inputs to a small classifier (logistic regression or shallow MLP) that
  outputs a per-candidate score. This is the moral equivalent of the AP
  detect+measure+classify pipeline and is what object-level metrics are
  computed against.
- **Do not move to anchor-based detection (YOLO/RetinaNet/Faster R-CNN) yet.**
  Trails are long, thin, have arbitrary aspect ratios, and span many tiles —
  segmentation + connected components is a better base. Revisit only if
  segmentation underperforms after serious tuning.

### 6.3 Baselines to beat
- `dia_source_detector` in the `fakes_difference_image` collection, at 5σ and
  at 3σ, including the trailed-source plugin where applicable.
- On real DP2 data (Candidate A evaluation): `dia_source` at nominal threshold.
- The old direct-image U-Net, to show the diffim direction is better, not just
  different.

### 6.4 Losses
- Primary: masked BCE with class-balanced `pos_weight`, computed from the
  diffim-space positive-pixel density (typically ~5–50× more sparse than in
  direct images, because only the thin residual contributes).
- Auxiliary: Focal-Tversky (already in repo). Ramp its weight up between
  epoch 8 and 18 as the current code does.
- Optional: small CE on the candidate-scoring head once that head is present.

### 6.5 Metrics
- **Pixel-level** (training convergence signal): BCE, masked ROC-AUC.
- **Object-level**: per-(visit, detector, band, SNR bucket):
  - recall of injected trails vs `inject_snr_diffim`
  - false-positive rate per `(4072×4072)` panel
  - comparison curve: NN recall vs AP 5σ recall vs AP 3σ recall, at matched
    false-positive rates
  - F1 and completeness-vs-purity curve over candidate score
- **Scientific end-to-end metric**: *fraction of injected trails recovered that
  AP alone at 5σ misses, at a tolerable false-positive cost of ≤ N candidates
  per detector per visit*. `N` is a tunable parameter reflecting candidate
  vetting capacity; a sensible first anchor is `N=10`.

---

## 7. SLURM execution plan

This is the operational skeleton. It is **SLURM-first**: nothing here is meant
to run interactively beyond smoke tests.

### 7.1 Job tree

```
experiments/diffim/
├── stage1_generate/            # run the fakes diffim pipeline
│   ├── driver.py               # per-(visit, detector) worker
│   ├── slurm_generate.sh       # sbatch --array=0-N, --cpus-per-task=4
│   └── manifests/              # one JSON per shard, list of (visit, detector)
├── stage2_pack/                # gather per-detector outputs into HDF5 panels
│   ├── pack_panels.py
│   └── slurm_pack.sh
├── stage3_train/               # training
│   ├── main.py                 # (reuses ADCNN/main.py)
│   └── slurm_train_diffim.sh
├── stage4_eval/                # AP comparison
│   ├── run_eval_ap.py
│   └── slurm_eval.sh
└── logs/
```

### 7.2 Stage 1 — dataset generation (SLURM-heavy)

- One `sbatch --array` job, one array task per shard of ~20 (visit, detector)
  tuples. Each task:
  - Loads `lsst_distrib w_2026_09` (matches current `slurm_inject.sh`).
  - For each (visit, detector):
    1. Fetch `post_isr_image` (or `injected_post_isr_image` if chaining),
       calibrate it, inject synthetic moving objects using
       `lsst.source.injection`, write to a scratch butler collection.
    2. Run the subtraction task (`AlardLuptonPreconvolveSubtractTask` or the
       current stack default) against `template_coadd`. Persist
       `fakes_difference_image` + `fakes_dia_source` to the scratch collection.
    3. (Empirical labels) run the subtraction a second time on the same
       uninjected `post_isr_image` to get the "clean" diffim; subtract the two
       to obtain the injected-only residual. Threshold → label mask.
    4. Write the tile-ready numpy arrays + metadata rows to
       `$SLURM_TMPDIR/<visit>_<detector>.npz`.
  - At the end of the task, copy all `.npz` files to the shared project
    directory (GPFS), not to `/tmp`, and mark them done.
- Resources per task: ~4 CPUs, 16 GB RAM, 1 h walltime is a reasonable first
  ceiling (tighten after benchmarking).
- Required files: `bad_visits.csv` to skip known-bad entries.

### 7.3 Stage 2 — packing (short SLURM job)

- Walk the `.npz` files produced by stage 1, stack them into the three-channel
  HDF5 panels plus the CSV metadata. One job, few hours wall-time, one CPU
  core.
- Deterministic ordering by `(visit, detector)` so splits are reproducible.

### 7.4 Stage 3 — training

- Reuse `ADCNN/scripts/slurm_train_frontier_batch.sh` as a template; change
  `--in-ch`, HDF5 paths, and sampler bucket config. Keep the DDP + EMA setup.
- Partition `ada`, 4 GPUs, same walltime profile.

### 7.5 Stage 4 — AP comparison

- One SLURM job that iterates over the held-out (visit, detector) set and
  produces three recovery curves: NN, AP at 5σ, AP at 3σ. Output a CSV and
  a notebook-ready JSON summary.

### 7.6 What stays local / debug

- Butler queries for dataset-type probing.
- Single-detector smoke runs (`--test-only`-style).
- Tile-level plotting.
- Anything in `notebooks/` that runs in < 5 min on a login node.

### 7.7 Logging, checkpoints, output layout

- SLURM stdout/err to `experiments/diffim/logs/<stage>-<jobid>-<arraytask>.log`.
- Training checkpoints to `checkpoints/diffim/<run_name>/{best,last}.pt`
  plus a `rescue_history.jsonl` file.
- Dataset shards named `diffim_shard_XXXX.h5` under a dedicated DATA dir
  (`/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/diffim/`).

---

## 8. Minimal pilot experiment

**Goal**: answer "can a U-Net on diffim channels produce an object-level
recovery curve that *substantially* beats AP 5σ at a matched candidate budget,
on even a tiny amount of data?" in under one week of wall time.

### 8.1 Pilot parameters
- Data: 20 `(visit, detector)` pairs selected from
  `LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2` across the 6 bands.
  100 injected trails per detector, trail length 6–60 px, SNR 2–8 in diffim space.
- Channels: `diffim`, `diffim_var`, `diffim_mask` (three channels).
- Network: `UNetResSE` with `in_ch=3`, no ASPP, widths = default.
- Training: 20 epochs on a single GPU, warmup 3, cosine decay.
- Eval: on 5 held-out detectors, compare NN recovery vs AP 5σ vs AP 3σ at
  candidate-budget `N ∈ {5, 10, 20}`.

### 8.2 Pilot success criteria
- **Pass**: NN recovers at least 15% more injected trails than AP 5σ at
  candidate-budget `N=10`, **and** at least 5% more than AP 3σ at the same
  budget, at SNR 3–5.
- **Probe**: even if the pass bar is not met, artefacts of the pipeline
  (normalization, label generation, mask-plane handling) should look correct in
  qualitative tile visualizations. If the qualitative picture is bad, fix the
  pipeline before scaling.
- **Fail-fast**: if a hand-picked high-SNR injection is completely invisible in
  the recovered diffim, (B) is broken and we fall back to (C) or (D) to
  localize the bug.

### 8.3 Pilot compute budget
- Stage-1 (generation): 20 array tasks × ~30 min = ~10 CPU-hours.
- Stage-2 (packing): < 1 CPU-hour.
- Stage-3 (train): 1 GPU × 4 hours.
- Stage-4 (eval): 1 CPU × 1 hour.
- Total: one working day, one GPU, modest CPU.

---

## 9. Main failure modes and scientific risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Template subtraction artefacts dominate the positive class on diffims | The NN ends up learning "where are dipoles / ringing / edge ghosts" instead of "where are trails". False positives blow up on real AP diffims. | Use `diffim_mask` channel so the NN can condition on bad-mask regions. Include synthetic artefact-region training panels. Stratified-sample by `inside_bad_mask`. Verify on real-data eval that FPR on real diffims is not driven by artefacts. |
| Pilot statistics too low to conclude | 20 detectors is ~ 2k injected trails. SNR bucket counts become noisy. | Use injected-SNR curves (continuous fit), not bin-wise histograms. Replicate the pilot on a second 20-detector set if the result is marginal. |
| Label mismatch (analytic vs empirical) | Analytic labels miss convolution subtleties; empirical labels cost 2× compute. | Start with empirical. Once a few hundred paired panels exist, benchmark analytic against empirical and only switch after < 0.5-pixel disagreement. |
| Normalization drift between tiles | Zero-mean MAD clip can amplify structure inside tiles that hit bad mask regions. | Compute MAD on a central 512×512 sub-crop **masked** by `diffim_mask`. |
| `preliminary_visit_image` vs `visit_image` vs `calexp` mismatch | DP2 stack version (w_2026_09) uses the `*_visit_image` vocabulary; older code paths used `calexp`. The butler registry shows both. | Pin to `preliminary_visit_image` / `post_isr_image` / `difference_image` consistently in stage-1 driver; assert butler dataset types at job startup. |
| Injections that overlap a *real* asteroid already in the diffim | Would double-count or confuse the truth mask. | Skip injection positions where the nearby real `dia_source_detector` centroid is within N pixels (reuse `MAX_PRE_SOURCES` guard from `simulate_inject_fill_deterministic.py` in spirit). |
| Pixel-only ROC-AUC looks good but object recovery is bad | Known hazard of tiny positive classes. | Primary metric is object-level recovery vs AP baseline. Pixel AUC is a training-time sanity check, not a headline. |
| Over-fitting to DP2 noise realization | DP2 is a specific stack version with its own artefacts. The NN may not transfer to e.g. AP commissioning data. | Hold out whole visits (not random detectors) to force visit-level generalization. Include multiple `physical_filter`s in training. |
| Storage pressure | Adding variance + mask channels triples panel storage. 2000 × 4072×4072 × 3 × 4 B ≈ 400 GB. | Use `float16` for `diffim_var` and `bool` for `diffim_mask`. Shard HDF5 by 100 panels/file. Monitor DATA-dir usage. |

---

## 10. Concrete next steps (prioritized)

The numbering below is the order these should be done. Each item is sized as
a single focused work session unless noted.

### 10.1 Immediate (code): scaffold + dry-run driver
1. `experiments/diffim/stage1_generate/driver.py`: a single-detector driver
   that takes `(visit, detector)` on the CLI, imports the already-existing
   helpers, fetches the required butler products, runs the source-injection
   + subtraction tasks into a scratch butler collection, and writes one
   `.npz` with channels + truth. Work against a single visit locally first;
   the script must not assume SLURM.
2. Add `experiments/diffim/stage1_generate/slurm_generate.sh` that launches
   `driver.py` as an array. Keep `--array=0-4` at first to limit the blast
   radius; ramp up only after one end-to-end success.
3. Add `notes/diffim_direction_plan.md` (this file). ✓

### 10.2 First pilot dataset task (SLURM, small)
1. Pick 5 `(visit, detector)` pairs across bands, drive them through
   `driver.py` via `slurm_generate.sh --array=0-4`. Inspect the resulting
   diffim + truth panels visually in a short notebook cell.
2. If the result looks right, extend to 20 pairs and run the full pilot
   generation.

### 10.3 First pilot training task
1. Extend `ADCNN/data/datasets.py` with a `DiffimH5Dataset` subclass (or
   parameterize `H5TiledDataset`) to read multi-channel HDF5, do zero-centered
   per-tile normalization using the variance channel, and yield
   `(C, H, W)` tensors.
2. Run `ADCNN/main.py --in-ch 3 --data DATA/diffim/train.h5 ...`, one GPU,
   20 epochs. Use the existing stratified sampler with redefined buckets
   (`ap_detected` replaces `stack_detected`). Dump rescue-validation curves.

### 10.4 First scientific comparison
1. Write `experiments/diffim/stage4_eval/run_eval_ap.py`: given a held-out
   set of `(visit, detector)`, produce for each injection:
   - `ap_5sigma_recovered`, `ap_3sigma_recovered`, `nn_recovered_at_N=10`.
2. Plot NN vs AP recovery curves over `inject_snr_diffim`, per band.
3. Write a 1-page summary (notebook + markdown) whether the pilot **passes**
   the success criterion in § 8.2. **Stop and revisit the plan before
   scaling if it fails.**

### 10.5 Beyond the pilot (not part of phase 1)
- Add science + template channels as an ablation.
- Move from empirical to analytic label generation once validated.
- Evaluate on real `dia_source` (Candidate A path).
- Revisit object-head (candidate-scoring) design.

---

## Appendix A — Butler probe results (run 2026-04-24 on dp2_prep, w_2026_09)

Read-only probes confirmed the following dataset types are **registered** on
this stack:

- Image products: `preliminary_visit_image`, `visit_image`, `calexp`,
  `difference_image`, `template_coadd`.
- DIA catalogs: `dia_source`, `dia_object`, `dia_source_detector`,
  `dia_source_visit`.
- Legacy names (still present): `goodSeeingDiff_differenceExp`,
  `goodSeeingDiff_templateExp`, `goodSeeingDiff_diaSrc`,
  `goodSeeingDiff_diaSrcTable`.
- Fakes infrastructure: `fakes_difference_image`, `injection_catalog`,
  `injected_post_isr_image`, `injected_post_isr_image_catalog`,
  `injectExposure_{config,log,metadata}`, `injectVisit_{config,log,metadata}`,
  `inject_coadd_{config,log,metadata}`, `consolidate_injected_catalogs_*`.
- Fakes analysis glue: `analyzeAssocDiaFakes*`, `analyzeDiaFakes*`,
  `assocDiaFakesDetectorVisitCore_metrics`, `injectedMatchAssocDiaSrc_*`,
  `injectedMatchDiaSrc_*`, `matchObjectToInjected_*`.

Stage2 collection used by the current direct-image pipeline:
`LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2`. Stage3 collections (which
contain `difference_image` / `dia_source_*` data) include
`LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage3` and several dated
sub-collections under `v30_0_6_rc1/DM-53881/stage3/*`. The exact collection to
use for the pilot must be selected by `butler query-datasets difference_image`
at the moment of running, because DP2 collections are still being produced.

## Appendix B — Explicit direct-image assumptions that become invalid

- `datasets.py::robust_stats_mad`: assumes positive-sky median; invalid on
  diffim.
- `datasets.py::H5TiledDataset.__getitem__`: clips to `[-5σ, +5σ]` around the
  sky median; must center at zero and respect variance plane.
- `simulate_inject_fill_deterministic.py`: pre-injection 5σ `source_detect` on
  PVI defines the forbidden mask; on diffim we instead want to forbid regions
  flagged by the diffim mask plane + a buffer around existing
  `dia_source_detector` centroids.
- `common.py::mag_to_snr`, `estimate_m5_local_from_psf_var`: use the PVI's
  `photoCalib` and variance. In diffim space the variance plane has a
  different scale (template contribution added); SNR conversions must take the
  diffim's variance plane as input.
- `pipetasks.py::source_detect`: hardcoded to `SingleFrameDetectAndMeasureTask`
  on PVI. For diffim evaluation we need `DetectAndMeasureTask` / whatever
  `ImageDifferenceTask`'s detection subtask is, or — more cleanly — we read
  `dia_source_detector` directly from the butler.
- `core/config.py::real_label_train_mode`: down-weighting real-source pixels
  is the solution to a problem that mostly vanishes on diffims; this config
  key becomes a no-op and should be dropped in the diffim code path.
