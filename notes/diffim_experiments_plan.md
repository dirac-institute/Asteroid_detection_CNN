# Diffim experiments — minimal code changes for Experiments 1 & 2

Status: proposal. Awaiting confirmation before implementation.

This is intentionally narrow. The goal is to answer the **single** scientific
question — "does the NN add object-level recovery beyond the LSST 5σ diffim
detector?" — with the dataset that already exists. No extra dataset
regeneration. Channel ablations, model ablations, complementarity-optimized
sampling are deferred to Experiments 3–5.

---

## 0. What we already have on disk

- `DATA_DIFFIM/train.h5` + `train.csv` (800 panels, 15,920 injections, 6
  bands, SNR 2–8, threshold built at 5σ).
- `DATA_DIFFIM/test_{5,4,3}sigma/test.{h5,csv}` (50 panels each, identical
  injections, three classical thresholds).
- `ADCNN/data/diffim_norm.py` (the correct zero-centered MAD normalization).
- `ADCNN/core/model.py::UNetResSE` (already accepts `in_ch=1`).
- `ADCNN/core/losses.py::BCEIoUEdge`, `AFTL` (reusable; `pos_weight` will need
  to be retuned).
- `ADCNN/evaluation/detection.py`, `metrics.py`, `inference/postprocess.py` —
  TBD whether reusable; pilot will probably want a fresh, narrow candidate
  module to avoid inheriting direct-image semantics.

The existing `ADCNN/main.py` is the full DDP training entry point with a
stratified mixture sampler keyed off CSV `stack_detection`/missed/touched
masks. It will work for diffim with minimal patching, but is overkill for
the pilot. The proposal below uses a separate, tight pilot script and only
reuses `main.py` if the pilot succeeds.

---

## 1. New code (5 small files)

All new files live under `experiments/diffim_pilot/`. Nothing in `ADCNN/`
changes for the pilot; if the pilot succeeds we'll fold winners back into
`ADCNN/`.

```
experiments/diffim_pilot/
├── dataset.py        # DiffimTiledDataset (HDF5 streaming + zero-centered MAD)
├── train_pilot.py    # 1-GPU training, UNetResSE, BCE+SoftIoU, EMA
├── candidates.py     # two-stage candidate extraction (t_low + score_thr)
├── evaluate.py       # NN-only / LSST-only / union object-level metrics
└── slurm_pilot.sh    # SLURM wrapper for an ada-partition 1-GPU run
```

Plus one shared output directory per run:

```
experiments/diffim_runs/<run_name>/
├── config.json
├── train.log
├── ckpts/{best.pt,last.pt}
├── preds/<panel_id>.npz   # logits per test panel (only saved at eval time)
├── candidates/<panel_id>.csv
├── metrics/
│   ├── per_injection.csv
│   ├── per_panel.csv
│   ├── summary.json
│   ├── froc.png
│   └── completeness_vs_snr.png
└── panels/tp_fp_lsst_missed_nn_recovered/*.png
```

### 1.1 `dataset.py` (≈ 80 LOC)

- `class DiffimTiledDataset(Dataset)` — derived from `H5TiledDataset` but
  with zero-centered MAD normalization (`normalize_diffim_mad`, clip ±5) and
  no median-subtraction.
- Reads only `images` and `masks` from HDF5 (1-channel input). Optionally
  reads `real_labels` and exposes it as a third per-tile array (used to
  build a per-tile **ignore mask**: pixels overlapping real_labels are
  excluded from both BCE and the candidate-level FP count).
- Tile size 128, stride = tile (no overlap during training).
- Yields `(x[1,H,W], y[1,H,W], ignore[1,H,W], panel_id, r, c)`.
- Per-panel MAD is computed once and cached. No per-tile stat.

### 1.2 `train_pilot.py` (≈ 200 LOC)

- 1 GPU. No DDP. `torchrun` not required.
- `UNetResSE(in_ch=1, out_ch=1)`.
- Loss: `BCEIoUEdge(lambda_bce=0.6, pos_weight=???)`. **`pos_weight` to be
  set from the actual diffim positive-pixel fraction** (= ~800 instead of
  the direct-image default 8).
- Adam, lr=2e-4, weight decay=1e-5, batch size 16, ~20 epochs, cosine LR.
- EMA (reuse `ADCNN/training/ema.py`).
- Stratified-by-tile-positive sampler: 75% positive tiles, 25% negative.
  Within positives, balance equally between (`stack_detection=True`,
  `stack_detection=False`) — this is the simplest possible
  complementarity bias.
- Validation = a 10-panel slice of train (held back by panel index) since
  the test sets are reserved for final evaluation. Pixel BCE + masked AUC
  as the in-loop metric.
- Outputs: `ckpts/{best,last}.pt`, `train.log`, `config.json`.

### 1.3 `candidates.py` (≈ 200 LOC)

Two-stage extraction from a probability map `p[H,W]` ∈ [0,1] (the
sigmoid of full-resolution stitched logits):

1. `t_low` (default 0.05) binarizes the map; connected components found
   with 8-connectivity (`scipy.ndimage.label`).
2. Per-component features: max p, top-5 mean p, area, bounding-box
   aspect ratio, PCA-elongation (λ1/λ2), bbox center xy, total integrated
   logit.
3. `score_thr` defined as a sweep over candidate score, NOT a single
   threshold. The candidate set at score s = {components with max_p ≥ s}.
   Default score is just `max_p`; the per-tile features are emitted so we
   can switch to a learned candidate scorer later without re-running the
   network.
4. Stitching: the dataset's tile stride is the tile size, so panel-level
   probability map is just a no-overlap mosaic. No need for overlap blending
   for the pilot.

### 1.4 `evaluate.py` (≈ 300 LOC)

This is the scientifically load-bearing piece. It produces, per test split
(`test_5sigma`, `test_4sigma`, `test_3sigma`):

- `per_injection.csv`: one row per CSV injection with columns
    `panel_id, x, y, SNR, SNR_estimation, trail_length, beta, band,
     stack_detection, stack_snr, nn_recovered@score=S, nn_candidate_id,
     nn_candidate_max_p, ...`
  for a grid of scores `S` so a FROC can be plotted without re-running.
- `per_panel.csv`: per-panel candidate counts; per-panel injected-recovered
  counts; per-panel LSST-detected counts; per-panel unmatched-NN counts.
- Matching: a candidate matches an injection if the candidate's footprint
  overlaps the truth mask of that injection by ≥ 1 pixel (consistent with
  the dataset's own `stack_detection` matcher). For pure-pixel arguments
  we also record IoU.
- Unmatched candidates are split into:
  - candidates inside `real_labels > 0` → "real_artefact_or_dia_source"
    (informational; **not** counted against the NN).
  - candidates outside both truth and `real_labels` → "spurious".
- Headline numbers in `summary.json`:
  - `N_lsst_only`: injections with `stack_detection=True`.
  - `N_nn_only` (per score grid): injections with `nn_recovered=True`.
  - `N_union` (per score grid): set union of the two recovery sets.
  - `N_lsst_missed_nn_recovered` (per score grid).
  - `N_spurious_candidates_per_panel` (per score grid).
- Plots:
  - FROC (per band, all-band): x = spurious candidates per panel,
    y = NN-recovered injections.
  - Completeness vs SNR for {LSST, NN, NN ∪ LSST, NN among LSST-missed}.
  - Completeness vs trail length.
  - Completeness vs β.
- Visual panels (≤ 20): TP, FP, LSST-missed+NN-recovered, NN-missed cases.

### 1.5 `slurm_pilot.sh`

- 1 ada-partition node, 1 GPU, ~6 h walltime ceiling.
- `srun python train_pilot.py …` → `srun python evaluate.py …` chained.
- Logs to `experiments/diffim_runs/<run_name>/`.

---

## 2. Pilot success criteria (must hit at least one)

These are the criteria I'll write into `summary.json` as a `pass: bool` and
treat as the gating decision for whether to scale up.

1. **At a candidate budget of ≤ 5 spurious candidates per panel**, NN
   recovers **≥ 30 %** of injections missed by `stack_detection`-5σ.
2. **Union (NN ∪ LSST-5σ) completeness** is at least **+10 percentage
   points** over `stack_detection`-5σ alone, at the same candidate budget.
3. The candidate budget that brings the NN to the recall of the **3σ stack
   threshold** is **smaller than the candidate budget the 3σ stack
   threshold itself costs** (we can compute the latter from per-panel
   `dia_source` counts on the `test_3sigma` build; if those aren't saved,
   approximate by counting `stack_detection=True` from 3σ minus 5σ as a
   proxy for "extra candidates at 3σ").

If none of these hits, the pilot is a clean negative result and we report
that — same posture as for the direct-image phase, and a real signal that
diffim alone is not the magic ingredient.

---

## 3. Open questions to confirm before implementing

1. **Pilot subset size** — full 800 train panels (~88 GB read, takes
   hours/epoch on 1 GPU) vs ~150 panels (~16 GB, easier to iterate). I
   recommend 150 panels for the pilot — the dataset has 6 bands and lots
   of repeated visits, so 150 random panels still has good coverage.
2. **`real_labels` in the loss**: do we **(a) ignore** those pixels in the
   loss, **(b) treat as soft-negative**, or **(c) ignore in loss but mark
   matched candidates as "informational" in eval (not FPs)**? My default
   is (c) — the cleanest scientific story. The training-time choice does
   not affect the eval-time bookkeeping.
3. **Variance plane**: not stored, so the variance-aware normalization
   `normalize_diffim_variance` is unavailable in the pilot. Recommend
   keeping it that way (plain MAD) and re-running the experiment with
   variance once we've regenerated the dataset with `driver.py` — that
   becomes Experiment 4 (channel ablation).
4. **Test split visit overlap (8 visits)**: noted in the inspection
   report; for the pilot result we'll *also* compute a "strict-visit"
   subset (test panels whose visit is not in train) and report both
   numbers in `summary.json`. No code change needed beyond an extra
   pandas filter.
5. **DDP later**: I am not setting up DDP for the pilot. If the pilot
   works, the next milestone is graduating to `ADCNN/main.py` with a
   diffim-aware dataset class — that's a separate, tracked task.

---

## 4. Compute footprint of the pilot

| Stage | Resource | Wall time (estimate) |
|---|---|---|
| Dataset class + smoke run on CPU (1 panel, batch=2) | login node | 5 min |
| Train pilot (150 panels, 20 epochs, 1 GPU) | 1 GPU, ada | ~3 h |
| Inference on 50 test panels × 3 thresholds | 1 GPU | ~10 min |
| Candidate extraction + metrics + plots | 1 CPU | ~10 min |

Total: ~4 wall-hours on one ada GPU, plus negligible CPU.

---

## 5. What this proposal explicitly does NOT do

- Does not regenerate the dataset.
- Does not modify `ADCNN/main.py`.
- Does not touch the direct-image code path.
- Does not run DDP, multi-GPU, multi-node.
- Does not optimize losses, sampler, or model beyond defaults.
- Does not commit to any architectural changes (UNetResSEASPP, attention
  layers, line-coherence heads, etc.).

Those come later, and only if the pilot demonstrates that the diffim NN
has real signal to add over the LSST detector.
