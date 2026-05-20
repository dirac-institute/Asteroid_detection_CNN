# Real-negative-background training for v7 — DESIGN + EVIDENCE + GO/NO-GO

Scope: design, CPU prototype and cost/benefit only. No GPU, no SLURM, no
tracked-file edits. Everything here lives under `experiments/explore_realneg_train/`
(gitignored). The production injector and training scripts are described, not run.

Artifacts produced:
- `step2_separability.py` / `step2_separability.txt` — real-FP vs synthetic-positive feature-space analysis
- `step3_prototype.py` — CPU hybrid-panel builder
- `proto/hybrid_panel_*.png`, `proto/proto_stats.csv` — 5 prototype hybrid panels + stats

---

## 0. The idea, in one line

v7 was trained on faint synthetic trails injected on real-background diffims,
but it **never saw a real subtraction residual *as a negative*** with
faint-trail-grade probability targets. The idea: build training panels =
**real empty-CCD difference image + a painted faint synthetic streak
(SNR≈3–10, realistic length)**, so the network learns
`faint trail` vs `real artifact` discrimination on the actual residual
distribution it fails on, rather than on clean synthetic-only negatives.

---

## 1. What machinery already exists vs. what is missing

### 1.1 The two existing builders (read in full)

`ADCNN/data/dataset_creation/simulate_inject_diffim.py` — the **synthetic
injector** v7 was trained on. Per (visit, detector):

1. fetch PVI + `single_visit_star_footprints` + overlapping `template_coadd`
2. AlardLupton subtract the **clean** PVI → clean diffim
3. `DetectAndMeasure` on the clean diffim → `pre_injection_Src` (= the real
   residuals: variable stars, DCR dipoles, kernel mismatch — *exactly the
   stuff that becomes FP*)
4. `forbidden = PVI mask planes ∪ clean-diffim residual footprints`
5. `generate_one_line` → injection catalog, flux from `snr_to_mag(...,
   snr_definition="detection")` using PVI photoCalib + PSF + summary `magLim`
6. inject into a PVI **clone**, subtract again with the **same kernel
   candidates** → injected diffim
7. `DetectAndMeasure` on injected diffim → `post_injection_Src`
8. drawn-line truth mask via `draw_one_line` (thickness = ½ local PSF width)
9. crossmatch pre/post, footprint-overlap recovery
10. **write**: `images` = injected diffim float32; `masks` = drawn-line truth;
    `real_labels` = `footprints_to_label_mask(pre_injection_Src)` (the real
    residual footprints → channel-3 input).

> Key fact (corrects a common misreading): **backgrounds were already real.**
> simulate_inject subtracts a real template from a real PVI; the diffim
> carries the genuine residual field. The domain gap is therefore **not
> "synthetic backgrounds"** — it is (a) the injected-SNR/length *distribution*
> and (b) the **absence of real residuals as explicit faint-grade negatives**:
> the training run used `--mag-mode snr --mag-min 2 --mag-max 8`, trail
> 6–60 px, `--number 20` injections per panel and essentially **no
> zero-injection panels**, so every panel's loss is dominated by "find the
> faint trail", never "this whole panel is real junk, output ~0".

`ADCNN/data/dataset_creation/build_test_real.py` — the **non-injection real
builder**. Already has *exactly* the empty-panel machinery we need:
`scan` writes a `manifest.csv` of asteroid panels **plus N_EMPTY
no-asteroid `role=empty` panels** (random science CCDs on asteroid visits,
excluding any real-asteroid pair); `build` runs subtract + DetectAndMeasure
and writes `test.h5/test.csv/panels.csv` byte-compatible with the synthetic
sets. `real_labels` = all diaSource footprints **except** the
asteroid-matched one.

`ADCNN/data/diffim_dataset.py` — `build_3channel`: `ch0 = clip(diffim/σ,±5)`
(σ = per-panel MAD), `ch1 = log1p-compressed local-std`, `ch2 = real_labels>0`.
`DiffimRandomCropDataset3ch` samples positive anchors from `csv` injections
and negative anchors *uniformly at random over panels* (`n_neg_anchors_per_epoch`).

`ADCNN/training/diffim_train.py` — AFTL + small BCE-anchor + masked-orient
loss; `--init-from <ckpt>` fine-tune mode (loads model+EMA, fresh
optimizer/scheduler, low `--lr`, few `--epochs`). Negatives contribute via
the BCE-anchor term and the AFTL `(1-tv)^γ` on empty tiles.

`ADCNN/utils/helpers.draw_one_line` / `common.draw_one_line` — thick-line
rasterizer (OpenCV), the truth-mask primitive. `common.snr_to_mag` /
`mag_to_snr` — the stack flux↔SNR model (detection mode: `σ_F = F(magLim)/5`).

### 1.2 What is therefore already there

| Need | Status | Where |
|---|---|---|
| Real empty-CCD diffim panels | **EXISTS** | `build_test_real.py scan` (`role=empty`) → `build` |
| Real-residual `real_labels` channel on empty panels | **EXISTS** | `build_test_real._labels_excluding` |
| Faint synthetic trail with PSF, flux↔SNR model, truth mask | **EXISTS** | `simulate_inject_diffim.generate_one_line` + `snr_to_mag(detection)` + `draw_one_line` |
| `--mag-mode snr` low-SNR sampling | **EXISTS** | `generate_one_line` snr branch |
| Hard-negative (zero-injection) panel path | **PARTIAL** | `--number 0` works mechanically but is untested as a deliberate fraction; no `--empty-fraction` knob |
| 3-channel + fine-tune from precision-tilt ckpt | **EXISTS** | `diffim_dataset` + `diffim_train --init-from` |

### 1.3 What is missing (the actual delta to build)

1. **An `--empty-fraction` knob in `simulate_inject_diffim`** (or a thin
   wrapper) so a controllable fraction of selected (visit,detector) panels
   are written with **zero injections** (real diffim + empty mask + real
   residual `real_labels`). Mechanically this is the `--number 0` path; what
   is missing is *selecting a deliberate fraction of panels for it* and
   making the dataset writer allocate/track them. ~30–80 lines.
2. **Low-SNR-focused, wide-length sampling**: change the `snr` draw in
   `generate_one_line` from `rng.uniform(2,8)` to a **log/Beta draw bulked at
   3–7, ceiling ≈10** and run with `--trail-length-min 4 --trail-length-max
   200`. ~10 lines + CLI flags. (The plan in
   `experiments/diffim_runs/test_real/RETRAIN_PLAN.md §A` already specifies
   this.)
3. **Negative-anchor mining biased to empty panels** in
   `DiffimRandomCropDataset3ch`: currently negative anchors are uniform over
   *all* panels; we want them concentrated on the zero-injection real panels
   (and on `real_labels>0` footprints) so the loss explicitly penalizes
   firing on real residuals. Either oversample empty `panel_ids` in
   `regenerate_anchors`, or pass empty panels with a higher neg weight.
   ~20 lines, fine-tune-only (no architecture change).
4. **A held-out real-empty supply that does NOT leak into `test_real`.**
   `DATA_DIFFIM/test_real` already consumed 150 `role=empty` CCDs (only 119
   scored). New training empties must be scanned from **different
   (visit,detector) pairs / a different day_obs window**. The scan tooling
   exists; the constraint is a *manifest hygiene* requirement, not new code.

Net: the heavy stack machinery (subtract, detect, inject, flux model, label
mask, 3-channel, fine-tune) is **100% already built and validated**. The work
is ~3 small, reversible edits (empty-fraction, SNR draw, neg-anchor bias) +
a clean empty-panel scan. This is the same surface the existing
`slurm_v7_finetune.sh` already exercised.

---

## 2. Evidence: the separability gap that on-real-bg training targets

`step2_separability.py` over the on-disk dumps
(`results/parts/emp_*.csv` 131,494 raw v7 real-empty candidates;
`empft_*.csv` 121,844 v7_ft real-empty candidates; `syn5_ft.pkl` 877
synthetic TRUE trails / 35,150 synthetic FALSE; `per_panel_fp.csv`).

**FP magnitude.** On 119 scored real empty CCDs, v7+V2-RF still fires
**mean 81.7 / median 67 / max 292 false "asteroids" per empty CCD**;
**0 / 119 panels are FP-free**. v7 raw emits ~900 raw candidates/panel
before the RF. This is the entire problem.

**Where the real FP live in feature space vs. genuine faint trails**
(median [p25,p75]):

| feature | real-FP (v7_ft) | synthetic TRUE | reads as |
|---|---|---|---|
| `mf_snr` | 1.10 [0.0, 2.26] | **4.04 [2.72, 5.77]** | real FP cluster at the noise floor; true trails are real signal |
| `mf_length` | 29 [12, 43] | **55 [41, 68]** | true trails are longer |
| `mf_flux` | 284 [0, 863] | **3120 [1714, 5115]** | true trails carry ~10× the integrated flux |
| `or_agg_max` | (low) | **0.91 [0.85, 1.0]** | orientation-coherence; AUC 0.91 alone |
| `max_p` | 0.87 [0.67, 0.98] | **1.00 [1.00, 1.00]** | v7's own confidence |
| `loc_dipole` | 0.20-ish | similar | **does NOT separate** (76% overlap) |
| `aspect` | 1.x | 1.x | **does NOT separate** (86% overlap) |

**Overlap test** — fraction of real-FP mass inside the synthetic-TRUE
[p05,p95] band (high ⇒ feature can't separate ⇒ network must learn it
directly): `loc_dipole` 76%, `aspect` 86%, `loc_skew` 72%, `mf_length`
61–65%, `mf_snr` 53–58%. Several "artifact" features (dipole, aspect, skew)
**do not separate at all** — i.e. the simple hand-features the V2 RF leans on
can't reject the residual FP; the discrimination has to come from the
*image* representation, which is precisely what retraining the CNN on real
residuals-as-negatives provides.

**Discriminability ceiling.** A 12-feature RF separates real-FP(ft) from
synthetic-TRUE at **AUC 0.95** (held-out). So the two populations *are*
separable in principle — but the RF that achieves it is itself
synthetic-trained, and the FT-RF hard-neg attempt already
**plateaued at ~43 FP/CCD** (`RETRAIN_PLAN.md` Step ① / `fp_fix.txt`):
RF reranking halved FP but the bottleneck is now **v7's candidate
generation**, not the reranker. The 0.95 AUC says the signal exists; the
plateau says it has to be injected *into the CNN's training distribution*,
not bolted on downstream. **This is the quantitative case for on-real-bg
retraining.**

> Caveat on the AUC: the comparison is real-empty-FP vs *synthetic* TRUE
> trails (no real faint-trail-on-real-empty population exists yet — that is
> the very thing this dataset would create). It bounds separability, it does
> not prove the CNN will hit it. See risks.

---

## 3. CPU prototype (no stack, no GPU, no training)

`step3_prototype.py` takes 5 genuine `role=empty` real-diffim panels from
`DATA_DIFFIM/test_real/test.h5` (panels 2578/2657/2575/2671/2642, bands
u/u/z/g/i) — real AlardLupton residual fields — and paints a faint
Gaussian-PSF-convolved streak whose **integrated signal is calibrated to a
target matched-filter SNR** using the *exact* downstream estimator
(`experiments/diffim_pilot/matched_filter.matched_filter_from_coords`:
`SNR = Σ diffim along line / (σ_MAD·√n_line)`), with one closed-loop gain
correction (the CPU stand-in for what `snr_to_mag`+photoCalib do in the real
stack injector). Real residual footprints (`real_labels>0`) in the same
panel are used as the contrast object.

Result (`proto/proto_stats.csv`, `proto/hybrid_panel_*.png`):

| panel | band | σ_MAD | target SNR | recovered SNR | bg-only SNR | trail peak/σ | real-resid \|z\|max |
|---|---|---|---|---|---|---|---|
| 2578 | u | 2.0 | 3 | 3.16 | 0.16 | 0.44 | 266.6 |
| 2657 | u | 1.6 | 5 | 4.93 | -0.07 | 0.56 | (none in crop) |
| 2575 | z | 69.8 | 7 | 6.47 | -0.53 | 0.66 | 3.30 |
| 2671 | g | 21.2 | 10 | 11.25 | 1.25 | 0.78 | 5.03 |
| 2642 | i | 81.7 | 4 | 5.73 | 1.73 | 0.48 | 12.3 |

Reads as **construction is sane**:
- Recovered matched-filter SNR tracks the target (3→3.2, 5→4.9, 7→6.5,
  10→11.2); residual scatter is panel-noise + footprint-PCA stochasticity,
  the same the stack injector has. Background-only SNR ≈ 0 — the trail is
  what creates the signal.
- The injected trail sits at **peak 0.4–0.8 σ, median |z|≈0.7–0.9** — buried
  in the noise exactly like a real faint asteroid — while the **real
  residuals in the same panels reach |z| = 5…266 σ**. That contrast
  (faint oriented line vs bright/extended real artifact, both on the same
  real residual field) is precisely the discrimination signal on-real-bg
  training would teach.
- Per-panel σ_MAD spans **1.6 → 82** ⇒ the diffim is *not* normalized at
  storage; `build_3channel`'s per-panel `clip(±5σ)` is what the network
  consumes. The prototype scales the trail by each panel's own σ, matching
  the production preprocessing. Bands u/g/i/z exercised (multi-band sane).
- Panel 2642's bg-only SNR +1.7 is the matched filter catching a real
  residual lying along the random injection line — realistic, not a bug; in
  production `generate_one_line`'s `forbidden` mask (PVI mask planes ∪
  clean-diffim residual footprints) prevents injecting on top of residuals,
  which the prototype deliberately omits to keep it stack-free.

The prototype is a sanity check of the *observable*, not the production
injector. Production must keep the stack path (real subtract + photoCalib
flux + `forbidden` mask), which already exists and is validated.

---

## 4. Dataset recipe (production, GPU-side — design only)

**4.1 Empty-panel supply & leakage protocol.**
- `test_real` already burned 150 `role=empty` CCDs (day_obs window of the
  fast-mover catalog). Training empties **must** be scanned from a
  **disjoint (visit,detector) set** — easiest: a **different day_obs
  window** in `--where` than both the v7 training set and `test_real`.
  Record the exact (visit,detector) lists; assert empty-set ∩ test_real-empty
  = ∅ and ∩ v7-train = ∅ before building.
- Supply needed: the addressable prize is small (≈30–40 stack-missed
  sightings at SNR 3–7). FP suppression — not recall — is the lever, and FP
  is learned from *empty* panels. Target **~400–600 real CCDs total**,
  **~35% zero-injection empties** (~150–200 empty panels) + ~65% faint-trail
  panels (~250–400). That is ≈ the size of the existing `train.h5`
  (800 panels) and well within one scan/build.

**4.2 Injection sampling** (in `simulate_inject_diffim`, gated by new flags):
- `--mag-mode snr`, SNR **log/Beta-sampled, bulk 3–7, hard ceiling ≈10**
  (the stack already owns SNR≥5 at ≥92–98% completeness per
  `eval_snr_gain.py`; spending capacity above 10 is wasted — see
  `RETRAIN_PLAN.md §"Where the gains actually are"`).
- `--trail-length-min 4 --trail-length-max 200` (real trails reach 150+ px;
  v7 trained 6–60 → length OOD).
- `--number` ≈ 15–25 on trail panels; **`--empty-fraction ≈ 0.35`**
  (new knob) → those panels get `--number 0`.
- Keep the existing `forbidden` mask so trails are not painted onto real
  residuals (preserves clean truth labels).

**4.3 Negative-anchor mining** (`DiffimRandomCropDataset3ch`, fine-tune
config only): bias `regenerate_anchors` negative draws toward empty
`panel_ids` and `real_labels>0` footprints (oversample factor ~2–3×) so the
loss is dominated by "real residual → output ≈ 0".

---

## 5. Fine-tune recipe (reuse, no new infra)

Reuse `experiments/diffim_runs/test_real/slurm_v7_finetune.sh` verbatim,
swapping the data and bumping negatives:

```
python -m experiments.diffim_pilot.v5_train \
  --run-name v7_ft_realneg \
  --init-from .../diffim_runs/pilot_v7/ckpts/best.pt   # or the precision-tilt best.pt
  --data-h5 <new realneg train.h5> --data-csv <...> \
  --n-train-panels ~500 --n-val-panels ~40 \
  --epochs 12 --batch-size 24 --lr 5e-5 \
  --n-pos-anchors-per-epoch 3000 --n-neg-anchors-per-epoch 6000  # ↑ negatives \
  --stk-balance 0.3 --aftl-alpha 0.6 --aftl-beta 0.4 --aftl-bce-anchor 0.2 \
  --widths 24 48 96 192 384 --kernel-lens 11 21 41 --n-angles 12 \
  --ema-decay 0.999 --ema-exclude agg_alpha
```

Architecture flags must match the v7 checkpoint (they do above). Then export
TorchScript + retrain the V2 RF including the new empty-panel candidates as
negatives (RETRAIN_PLAN §D) — cheap, high-leverage, no GPU.

---

## 6. GPU cost estimate (measured, not guessed)

**Data build (CPU/SLURM, no GPU):** the existing diffim builders run ~90-way
on `roma`. `test_real` build (2,678 panels) completed in one slurm job;
~400–600 panels is a fraction of that — **est. 4–10 h wall on the existing
~40–90-way CPU array** (subtract+detect dominated). No GPU.

**Fine-tune (1 × A100/ampere):** the *measured* precision-tilt fine-tune
(`experiments/diffim_runs/v7_ft_hn/train.log`) ran **12 epochs × 740 panels ×
7,000 anchors/epoch at ~250 s/epoch ≈ 51 min total** on one ampere GPU
(`slurm_v7_finetune.sh` requests 4 h, used <1 h). The realneg fine-tune is
**smaller** (~500 panels) with **more negative anchors** (~9,000/epoch vs
7,000) → est. **~280–320 s/epoch × 12 ≈ 1.0–1.1 GPU-hours**. Budget **1
ampere-GPU job, ≤2 h** (well inside the existing 4 h request). TorchScript
export + RF retrain: minutes, CPU.

**Eval (reuse, GPU optional):** `slurm_score_array.sh` + `merge_results.py`
on `test_real`; `sweep_threshold.py`/`sweep_curve.py` for the FP–recall
curve; synthetic `test_5sigma` regression via the existing BAR
(`bar_ft.txt`/`bar_shipped.txt`). ~1–2 GPU-h scoring, all infra exists.

**Total marginal cost: ≈ 1 GPU-hour of training + ≤2 GPU-h eval + ≤10 CPU-h
data build.** This is *small* — comparable to one existing fine-tune cycle.

---

## 7. Expected FP-reduction mechanism

v7 currently emits ~900 raw candidates / empty CCD because, on tiles
containing real residuals it never saw labeled as negatives, its AFTL/Hough
head has no gradient pushing those to 0 — the synthetic training panels'
residuals were never *anchored* as negatives (negatives were uniform-random
tiles, mostly flat sky). Adding **~150–200 zero-injection real empty panels
with heavily-mined negative anchors on the residual footprints** gives the
network direct "real residual → 0" gradient on the exact distribution
(`step2`: dipole/aspect/skew don't separate ⇒ must be learned in-image). The
realistic mechanism: **raw candidates/empty CCD drop from ~900 toward
≤100–200**, and post-RF FP from ~80 toward an order-of-magnitude lower —
*if* the residual distribution is learnable by the CNN (the AUC-0.95 result
says the information is present). Recall in the SNR 3–7 band should be
preserved or improved (we now train at exactly that SNR with realistic
lengths, vs the OOD 6–60 px / SNR 2–8 v7 saw).

---

## 8. Leakage / eval protocol

1. **No-leak scan**: training empties scanned from a day_obs window disjoint
   from both v7-train and `test_real`; assert (visit,detector) disjointness
   before build; persist the lists in the run dir.
2. **Held-out real empties**: reserve a fresh real-empty scan (different
   pairs again) as the FP benchmark — do NOT score on training empties.
3. **Synthetic regression bar**: `test_5sigma` (1,000 objects) must not lose
   true-trail recall (compare to `bar_shipped.txt`/`bar_ft.txt`:
   posR must stay ≈1.0; the realneg RF hard-neg run already showed
   posR 1.0→1.0).
4. **test_real benchmark**: per-object / per-sighting recall + FP/empty-CCD
   vs the documented baseline (29.9% obj recall, +7 NN-only objects, 81.7
   FP/CCD). Success ⇒ FP/CCD → ≤~1–2 (2nd-stage-usable) **and** NN-only
   objects ≥ baseline.
5. Curves via `sweep_threshold.py`/`sweep_curve.py` (FP vs recall at matched
   operating points), not single-threshold counts.

---

## 9. Risks

- **Could hurt synthetic recall.** Heavier negative weighting + low-SNR-only
  training can push the model conservative; the SNR-3–7 trails are
  *intentionally* near the noise floor. Mitigate: keep `stk-balance` modest,
  monitor `test_5sigma` posR each fine-tune, early-stop on recall drop.
- **Selection bias in empty panels.** "Empty" = random science CCD on an
  asteroid visit with no catalogued fast mover; it can still contain an
  *uncatalogued* real moving object → mislabeled negative. Low rate, but it
  caps achievable FP and biases the negative set. Mitigate: large empty pool,
  accept a small label-noise floor.
- **Real-empty supply / diversity.** test_real already consumed 150 empties
  from one day_obs window; training + held-out eval need *more, disjoint*
  pairs. Supply is finite per window; may need to widen `--where`. Diversity
  of residual morphology (DCR/CR/ghost/edge) must approximate test_real or
  the FP gain won't transfer.
- **The AUC-0.95 is an upper bound, not a guarantee.** It is measured against
  *synthetic* TRUE trails; the CNN may not reach it, and the real
  faint-trail-on-real-empty population is unobserved (it is what we'd be
  creating). The honest expectation is "large FP reduction", not "FP→0".
- **Plateau risk repeats.** The RF hard-neg attempt plateaued at ~43 FP/CCD.
  If the CNN's *candidate generation* is fundamentally limited by the ±5σ
  clip / architecture rather than the training distribution, on-real-bg
  retraining could also plateau — though at the CNN level there is far more
  capacity to reshape than at the RF.

---

## 10. GO / NO-GO

**Recommendation: conditional GO — proceed, but gated.**

Rationale from the evidence:

- **GO factors.** (i) Cost is *small and measured*: ~1 GPU-hour fine-tune +
  ≤10 CPU-h data build, all infra/scripts already exist and were exercised
  (`slurm_v7_finetune.sh`, `v7_ft_hn/train.log`). (ii) The mechanism is
  sound and *quantified*: real FP overlap genuine trails on the
  hand-features the RF relies on (dipole 76%, aspect 86%, skew 72% overlap),
  so downstream reranking provably plateaued (~43 FP/CCD) — the fix must be
  in the CNN's training distribution, and the separating information
  demonstrably exists (12-feature AUC 0.95). (iii) The prototype shows the
  hybrid construction is sane and the production machinery to do it for real
  is ~3 small reversible edits on top of fully-built, validated stack code.
  (iv) The dominant lever is FP suppression, which is learned from *empty*
  panels — the cheapest, lowest-risk part of the recipe.

- **The gate (do the cheap thing first).** Per `RETRAIN_PLAN.md`, Step D
  (retrain the V2 RF with real empty-panel hard negatives) is even cheaper
  and *already done*: it **halved** real-empty FP (85→43 @thr0.10) with
  **synthetic recall fully preserved**, then **plateaued**. That plateau is
  the green light — it isolates the bottleneck to v7's candidate generation,
  which is exactly what on-real-bg retraining addresses. So: **GO to the
  on-real-bg fine-tune**, with the explicit kill-criterion below.

- **NO-GO / kill criteria** (decide after the *first* 1-GPU-hour fine-tune +
  reuse-eval): abandon if **either** (a) `test_5sigma` true-trail recall
  drops > ~5% (posR < ~0.95) and cannot be recovered by rebalancing, **or**
  (b) held-out real-empty FP/CCD does not improve **meaningfully beyond the
  RF-hard-neg plateau (~43/CCD)** — i.e. < ~2× further reduction. If FP
  cannot be driven toward a 2nd-stage-usable rate (order ≤1–2/CCD)
  *regardless of recall*, v7-as-second-stage is not worth pursuing and this
  line should stop. The small addressable prize (~30–40 stack-missed
  sightings at SNR 3–7) does not justify more than this one bounded,
  reuse-only experiment.

**Bottom line:** one bounded, ~1-GPU-hour experiment on top of existing
infra, with a sharp kill-switch. The evidence (FP overlap on RF features +
RF-hard-neg plateau + AUC-0.95 separability ceiling + sane prototype) makes
it the logical, low-cost next step — but it is a *gated* GO, not an
open-ended program.
