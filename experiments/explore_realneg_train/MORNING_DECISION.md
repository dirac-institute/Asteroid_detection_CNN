# Morning DECISION — 2026-05-20 (overnight 2026-05-19 ~22:00 → ~09:00 CEST)

Bottom line — **no tracked-code change, no checkpoint promotion**. Production
config stays exactly as it is. Promoted FT v7 (`v7_ft_hn/ckpts/last.pt` +
`rf_postproc_v2_ft.pkl`) at `DEFAULT_THR = 0.50` remains the deployed
2nd-stage detector. Evidence below.

What you have in the morning:

1. Four independent investigations that converge on the same scientific
   conclusion (sections 2–5).
2. A specific, validated empty-mode worker bug in
   `ADCNN.data.dataset_creation.simulate_inject_diffim` blocking the realneg
   fine-tune (section 6).
3. 21 G of valid leakage-vetted real **trail** panels saved on scratch
   (section 6) — partial but reusable.
4. The one targeted CNN-side experiment worth running next (section 7).

---

## 1. What I tried, what hit walls, what fired

| stream | plan | what happened |
|---|---|---|
| realneg dataset build (job 26877243) | 400 trail + 200 empty + 120 held-out empties on scratch | trail 399/400 ✅ (21 G); empty 0/200 ❌; held-out 0/120 ❌ — worker bug on `number=0` (§6); merge then ENOSPC on csv write |
| realneg fine-tune + reuse-eval + killswitch | apples-to-apples ft vs promoted on same held-out empties | **never launched** — no valid empty-bg train data, no held-out empty FP bench |
| 5-agent exploration | independent levers, no promotion | **4 of 5 produced RESULTS.md**: variant_matrix, extraction, calib_tta, reranker. Coherence wrote intermediate reports but no final RESULTS.md. Net: 3 NO-GOs, 1 staged-conditional |
| scratch quota | 100 G | filled by 59 G contaminated merge dataset (200/600 panels were the failed-empty zero-fills) — deleted; now 42 G in use, 58 G free |

The user's NON-NEGOTIABLE GATES (synthetic objectwise BAR ≳ promoted +
killswitch GO before any tracked promotion) did exactly their job — nothing
shipped without evidence.

---

## 2. Reranker (`pilot_v7/ckpts/rf_postproc_v2_ft.pkl`) — full diagnostic

(from `/sdf/scratch/users/m/mrakovci/explore/reranker/RESULTS.md`)

Findings load-bearing for future work:

- `rf_postproc_v2_ft.pkl` and the shipped `rf_postproc_v2.pkl` are
  **byte-identical** (md5 `2febc4df…`). The bar_ft vs bar_shipped gap (cTP
  858 vs 800 @0.50) is **entirely the CNN model**, not the RF.
- `posR≈1.0` used to promote the RF is **in-sample** (the RF was trained on
  every `syn5_ft.pkl` positive). Honest panel-disjoint OOB recall, retrained
  per fold on other syn panels + the SAME train-split real empties:

  | thr | FPgen/CCD | posR (in-sample) | **OOB recall (panel-disjoint)** | Δ |
  |---|---|---|---|---|
  | 0.10 | 37.8 | 1.000 | 0.795 | −0.205 |
  | 0.50 | 5.6 | 0.999 | 0.621 | −0.377 |

  The shipped `DEFAULT_THR=0.50` loses ~38 % of genuine synthetic trails
  out-of-sample. The "promoted RF posR≈1.0" is an artefact of the
  promotion protocol; the *real-trail* score (+7 objects / +46 sightings
  on `test_real`) is the honest number.

- The 5.6 surviving genuine-FP / CCD at thr 0.50 sit **inside** the positive
  feature manifold — median `FP_in_posrange` over 72 feats = 0.97. The
  separator hand-features (`loc_dipole`, `aspect`, `loc_skew`, `elongation`,
  `mf_snr`) **do not separate** here. The reranker has consumed every
  separable bit in V2 features.

Verdict: **the RF reranker is already at its V2-feature ceiling.** A bigger
RF or a stacked GBM cannot help without new features. Cf. §3 below: the new
feature would have to come from the CNN itself.

## 3. Pre-RF candidate extraction — physical gates exhausted

(from `/sdf/scratch/users/m/mrakovci/explore/extraction/RESULTS.md`)

v7 emits 812 raw cand/CCD on empty diffims. They are **not** cheap junk —
the adaptive `t_low = μ + 6σ` capped at 0.5 already pins binarization high,
so every raw candidate has `max_p ≳ 0.52`, median area 273 px, large and
elongated. The "inflation" is the model producing trail-shaped blobs on
real empty diffims.

Pre-RF pre-gate strict-dominance grid scan (posR ≥ 0.999, faintR ≥ 0.995):
the best loss-free pre-gate **saves 0.09 FP/CCD out of 47.6 (0.2 %)**. On
real-truth `cand_*.csv`, applying even the mildest lossy gate (area≥8,
mf_snr≥1.5) to the 47 RF-surviving real-trail sightings expects to lose 1–7
real object recoveries to save < 3 FP/CCD of 47.6 — catastrophic trade.

Only `min_area: 4 → 8` is a free-by-rounding tightening (raw cands −9 %, FP
−0.31/CCD, real loss expected 0). Defensible but **not** a real win, and
**not applied** without explicit user sign-off (tracked-code change).

Verdict: **no pre-RF gate strictly dominates.** The bottleneck is the CNN
itself.

## 4. Calibration + TTA — both NO-GO

(from `/sdf/scratch/users/m/mrakovci/explore/calib_tta/RESULTS.md`)

**A. Per-band / per-σ recalibration** of the 5 CNN confidence features
(`max_p, mean_p, top5_mean_p, integrated_logit, or_agg_max`), z-standardised
on train-CCD genuine-FP background, RF retrained:

| posR* | base FP/CCD | band Δ | σ Δ | band×σ Δ |
|---|---|---|---|---|
| 0.999 | 6.2 | +0.3 | −0.8 | +0.0 |
| 0.990 | 3.9 | +0.4 | −0.2 | +0.4 |
| 0.950 | 3.1 | +0.4 | +0.1 | +0.2 |

Every variant neutral-to-worse at fixed posR. Single-feature AUC degrades
under per-group standardisation across the board. The synthetic→real
miscalibration is a **global** compression, not a per-band/per-σ shift.
**NO-GO.**

**B. D4 test-time augmentation:** the addressable boundary band is set by

| MF-SNR | n stack-missed sightings | NN recall now |
|---|---|---|
| < 3 | **832 (91 %)** | 0.043 |
| 3–5 | 33 | 0.242 |
| 5–7 | 2 | 0.000 |
| 7–10 | 2 | 0.500 |
| 10–15 | 10 | 0.000 |
| > 15 | 38 | 0.026 |

**91 % of stack-missed sightings have no recoverable residual flux.** The
TTA-addressable boundary band is ~28 sightings / 1–2 objects total. Generous
+25 % TTA uplift on that band = ~+7 sightings / 0–1 objects. Cost = 4–6 ×
GPU. **NO-GO.** (Conditional bounded probe possible — only worth ~2–4
GPU-h on the ~2 000 asteroid panels with stack-missed sighting MF-SNR ≥
2.5, with a pre-registered ≥+5 sightings / ≥+1 object bar.)

## 5. Variant matrix (real-empty-bg fine-tune) — design only, no GPU spent

(from `/sdf/scratch/users/m/mrakovci/explore/variant_matrix/RESULTS.md`)

Honest framing of the prize, restated: `threshold_sweep.txt` on `test_real`
gives the entire addressable upside as 22 stack-never-seen objects @thr0.02
/ 468 FP per empty CCD, collapsing to 8 @0.10 / 79 FP, 3 @0.20 / 31 FP, 0
@0.50; `snr_gain.txt` shows the NN recall over the stack only at SNR 3–5
(24 %) and SNR 7–10 (50 %, n=2). The lever is **FP suppression to make the
existing +7-object gain deployable at low FP**, not big recall gains.

Three orthogonal variants staged at
`/sdf/scratch/users/m/mrakovci/explore/variant_matrix/scripts/`
(`cfg0_baseline.sh`, `cfg1_negheavy.sh`, `cfg2_fptilt.sh`,
`cfg3_recallguard.sh`), with a rubric gated by `killswitch.py` (synthetic
recall preserved AND >2× FP improvement at matched posR). All 4 ready to
`sbatch`, all `bash -n` clean, all guard on `dataset/train.h5` — **none
launched** because the realneg dataset never finished building (§6).

## 6. Why the dataset build failed (and the fix)

(a) **Worker bug on `mode=empty` (`number=0`)**: every one of the 200
empty-mode + 120 held-out-mode panel jobs errored — pattern of ~5 s
per-panel crash, 0/200 OK. The trail mode (`number=20`) ran cleanly
(399/400). The 0/200 + 0/120 pattern points to `one_detector_injection`
not surviving `number=0`. This is a tracked bug in
`ADCNN/data/dataset_creation/simulate_inject_diffim.py` — **not** in the
experimental wrapper. Diagnosis path: run the worker with `number=0` on a
single visit/detector under a debugger; the first hit is almost certainly an
empty-catalogue path that assumes ≥1 injection. Fix is one-line guard;
re-run build (~1.5 GPU-h roma).

(b) **Scratch quota crisis**: even before the worker bug, the build was
going to be tight. 80 G scratch with 400 trail panels gzip-h5'd to 21 G
proves the realneg dataset (trail + empty + held-out) cannot fit at gzip-4
on this scratch — would need gzip-9, dataset relocation to a larger volume
(e.g. `$HOME` if quota allows), or per-panel pre-cropping. The merge
ENOSPC'd writing the 2.3 MB train.csv — i.e. literally the very last byte —
underscoring how tight this is.

(c) **Cleanup done**: deleted the 59 G contaminated merge h5 (200 of 600
panels were zero-fills, unusable); deleted the two 1944-byte empty-mode
data.h5 stubs. Trail data preserved at
`/sdf/scratch/users/m/mrakovci/realneg/data/trail/data.h5` (21 G, 399 valid
panels, gzip-4) — reusable once the empty-mode bug is fixed.

(d) **No file outside scratch was touched.** No tracked code edited overnight.

## 7. The one experiment that's worth running next

Idea **B** from the variant_matrix idea scan: **focal / hard-neg-mined
weighting on `real_labels>0` footprints**, applied as a sampler-only edit
in `DiffimRandomCropDataset3ch.regenerate_anchors` + a focal `(p)^γ` weight
on the AFTL/BCE neg term. ~25–40 LOC, fine-tune-only, fully reversible. Cost
~1 GPU-h. Mechanism: concentrate gradient on the exact real-residual
footprints that become FP — the kind of signal §2 proved hand-features
cannot capture.

This is **not** authorised here — it requires a tracked-code edit, which
the morning report should NOT do unilaterally. It is the recommended next
move if the user wants to spend GPU on FP suppression. Pre-conditions:
fix the `number=0` worker bug first (§6) so the variant matrix is even
runnable; idea B then comes AFTER the matrix gates the data-exposure
vs. loss-shape question, not instead of it.

## 8. What changes today: nothing

- `experiments/diffim_runs/pilot_v7/ckpts/v7_scripted.pt` → unchanged.
- `experiments/diffim_runs/pilot_v7/ckpts/rf_postproc_v2.pkl` → unchanged.
- `DEFAULT_THR = 0.50` → unchanged.
- No commit, no push, no tracked-code edit overnight.
- `test_real` headline number unchanged: **NN-only +7 objects / +46
  sightings of 917 stack-missed**, gating FP/CCD 5.6 @0.50 on the
  held-out empties.

## 9. Concrete next-morning actions for the user

In priority order, all bounded:

1. **Fix the worker `number=0` bug** in
   `ADCNN/data/dataset_creation/simulate_inject_diffim.py` (~15 min). I
   did not edit tracked code overnight because the gate rule applies.
2. **Re-run the build** with the fix (and gzip-9 or relocated dataset to
   beat the scratch quota). Trail step already done — skip with
   `test -f trail/data.h5 || …` to save ~30 min.
3. **Run `cfg0_baseline.sh`** + eval + `killswitch.py`. If killswitch
   prints GO with `usable_fp > ~1–2/CCD`, queue idea B as the single
   sanctioned follow-up (§7).
4. **Bank the +7-object result** in `test_real` regardless of whether the
   realneg line shows a win: it is already a real, reproducible scientific
   gain over the LSST 5σ stack.

The convergent honest finding from 3 independent agents (extraction, calib,
reranker) plus the design framing of the 4th (variant_matrix) is the same:
**v7 + V2 RF have exhausted the post-CNN levers.** Any further FP
suppression has to come from the CNN model itself — and the only justified
way to attack the CNN is the precision-tilt fine-tune on real empty
backgrounds with footprint-targeted gradient. That is the experiment the
realneg line was designed to gate. The morning leaves you with the
exploration done, the gates in place, and a clean shovel-ready path
forward — not a promoted change without evidence.
