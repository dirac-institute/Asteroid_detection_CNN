# explore_oppoint — best science operating point for the v7 CNN 2nd stage

Scope: CPU-only, read-only on `experiments/diffim_runs/test_real/results/`.
All scripts + raw reports in this dir. Nothing tracked was modified.

Scripts:
- `pareto_and_thresholds.py` -> `flat_pareto.csv`, `_pareto_report.txt`,
  `per_snr_frontier.csv`, `per_band_hits.csv`, `per_length_hits.csv`
- `physical_gates.py` -> `_gates_report.txt`
- `twod_and_band.py` -> `_twod_band_report.txt`

## Anchor (reproduced threshold_sweep.txt exactly)

99 stack-NEVER-detected objects; 917 stack-missed sightings; 150 empty CCDs.
- new objects = unique ObjID **in the 99-never-detected set** with a
  on_truth=1 candidate at score>=t
- new sightings = unique asteroid image_id with on_truth=1 at score>=t
- FP/empty-CCD = #empty-role candidates score>=t / 150

My reproduction matches `threshold_sweep.txt` to the last digit
(NEW_obj 22/14/8/4/3/2/2/0 at thr 0.02..0.50; FP/CCD 468/201/79/45/31/19/12/8;
missSight within 1-3 of file rounding). Methodology validated.

**Selection-bias flag (important):** cand_*.csv contains 78 unique on_truth
ObjIDs but only **22** are genuinely stack-never-detected. The other 56 are
"free" stack-missed *sightings* of objects the stack already catches in
another visit — they inflate the sighting count but add **zero** new
objects. The real prize is tiny: at the most generous threshold (0.02) the
absolute ceiling is **22 new objects** for **468 FP per empty CCD**.

## 1. The real science Pareto frontier

Dense flat-threshold curve (`flat_pareto.csv`), objects frontier:

| thr  | new_obj | new_sight | FP/empty-CCD |
|------|---------|-----------|--------------|
| 0.02 | 22      | 136       | 468 |
| 0.04 | 16      | 100       | 277 |
| 0.06 | 14      | 71        | 160 |
| 0.10 | **8**   | **47**    | **79** (promoted) |
| 0.12 | 5       | 40        | 60 |
| 0.17 | 4       | 22        | 38 |
| 0.28 | 3       | 16        | 20 |
| 0.46 | 2       | 5         | 10 |
| 0.50 | **0**   | **3**     | **8** (promoted) |

There is **no knee**. new_obj decays roughly linearly in log(FP):
every halving of FP/CCD costs ~3-4 new objects. The objects/FP ratio is
*best at the top of the curve* (22 obj / 468 FP = 1 object per 21 FP/CCD)
and only gets worse going down. There is no flat-threshold point that
yields meaningful objects at a tolerable FP rate. The promoted thr 0.10
(8 obj, 79 FP/CCD) is on the frontier but not special; thr 0.50 yields
**0** new objects.

Cost-optimal flat point: depends entirely on the FP price. If a real
detection is worth ~20-30 empty-CCD false alarms, the whole curve is
break-even-or-worse; no flat point clears a realistic bar.

## 2. Per-SNR / per-band / per-length / 2-D cuts

**Per-trail-length** (`per_length_hits.csv`): new objects are spread
across all length bins (8 at <8px, 13 at 8-12, 5 at 12-20, 7 at >20).
A length gate cannot isolate the prize — consistent with snr_gain.txt's
finding that <8px NN recall is only 0.023; the few <8px wins are not
separable by length alone.

**Per-band** (`per_band_hits.csv`, `_twod_band_report.txt` (b)): new
objects are concentrated in **r/i/z (9+9+9)**; **u and y give 1 each**
despite u having the *most* on_truth hits (35) — u/y hits are
overwhelmingly objects the stack already owns. Dropping band u costs
**0 new objects** at every threshold (13/7/2 unchanged) while removing
~25-30% of asteroid candidate volume; in production the pipeline runs
per-CCD/band, so dropping u (and y) also removes those bands' empty-CCD
FP entirely. **riz-only** loses ~4 objects (13->9 at thr 0.05) — too
aggressive. **drop-u (keep grizy) is a clean, free, modest win.**

**Per-SNR-bin flat thresholds**: do NOT help on their own
(`per_snr_frontier.csv`) — empties in cand_*.csv carry no SNR, so a
bin-wise asteroid threshold has no matching FP control and the worst-case
binds. But the right formulation — a **2-D score×SNR cut** with FP
counted fairly on empft (which has mf_snr) — **does move the frontier**
(`_twod_band_report.txt` (a)):

Rule: keep if `score>=0.20` OR `(3<=mf_snr<12 AND score>=0.10)`

| metric | 2-D cut | flat (matched, FP via empft) |
|--------|---------|------------------------------|
| new_obj | 3 | 3 needs thr~0.20 |
| new_sight | 26 | thr 0.20 gives only 20; 26 sightings -> thr~0.06-0.08 |
| FP/CCD | **32** | thr 0.20: ~10 FP but 20 sight; for 26 sight + 3 obj: **78-133** |

At fixed yield (3 objects, 26 sightings) the 2-D cut delivers
**~32 FP/CCD vs ~78-133 FP/CCD for the flat threshold — a 2-4x FP
reduction at matched science.** This is the *only* lever that genuinely
improves the frontier. It works because the addressable mf_snr 3-12
sightings tolerate a lower score gate while the FP-dominated low-SNR bulk
is held to a high score.

## 3. Physical FP gates

Built on the only feature-bearing files: `syn5_ft.pkl` (877 synthetic
positives, label_v2==1) and `empft_0.csv` (121,844 real-residual FP, 150
CCDs). **Structural caveat:** the real on_truth asteroid recoveries in
cand_*.csv have **no feature rows anywhere**, so a gate's effect on the
*real* science recoveries cannot be measured directly — we use the
project-standard synthetic-positive recall (posR) as the
trail-preservation proxy (same proxy as fp_fix.txt).

Result (`_gates_report.txt`): at **posR>=0.99 every cheap physical cut
removes only 3-5% of raw empft FP.** Best single gates:
- `or_agg_mean_loose >= 0.169`: posR 0.990, **14% raw FP cut** (strongest)
- `area >= 6`: posR 0.993, 5% FP cut
- `mf_length >= 7`: posR 0.997, 4% FP cut

Two-feature AND combos do no better (3-5%). The synthetic positives' low
tail overlaps the real-residual FP too heavily (positive p5
`or_agg_mean_loose`=0.355 sits right at FP median 0.378).

**Decisive negative — projection onto the deployable frontier**
(`_gates_report.txt` bottom): after the RF score cut (the operating
regime), the surviving empties are *already* the trail-like ones, so the
physical gate's FP-survival is 0.976-1.00:

| thr | FP/CCD flat | FP/CCD + best gate | gate keeps |
|-----|-------------|--------------------|-----------|
| 0.05 | 200.9 | 196.2 | 0.976 |
| 0.10 | 79.2 | 78.8 | 0.995 |
| 0.20 | 31.4 | 31.4 | 1.000 |

The RF has already internalized all the cheap physical signal
(`or_agg_*`, `mf_length`, `elongation`, `area` are RF inputs). A standalone
physical gate adds essentially **nothing** on top of the RF score
(<=0.5% FP reduction at the operating point) while adding deployment risk.

## 4. Honest verdict & recommendation

**Is there a better deployable operating point than the promoted one?**

- **Physical gates: NO.** Redundant with the RF; <=0.5% FP at the
  operating point at best. Do not deploy.
- **Per-SNR/length flat thresholds: NO.** Prize not separable that way.
- **2-D score×SNR cut: YES, modestly.** `score>=0.20 OR (3<=mf_snr<12
  AND score>=0.10)` recovers **3 objects + 26 sightings at ~32 FP/CCD**.
  The flat curve needs ~78-133 FP/CCD for the same — a real **2-4x FP
  reduction at matched yield**, the single defensible improvement.
- **drop band u (keep grizy): free.** 0 object loss, removes ~25-30%
  candidate volume and u-band empty FP in production. Stack with the 2-D
  cut.

**But the honest bottom line: the prize is genuinely too small and
selection-biased.** Even the best operating point recovers only **3 new
objects** (of 99 the stack never sees) at ~32 FP/empty-CCD — i.e. ~1 new
object per ~10 empty-CCD false alarms, every empty CCD also carrying ~30
false candidates. To get into double-digit new objects (>=14) you must
accept >=130 FP/CCD on either curve. This confirms snr_gain.txt: the
stack owns MF-SNR>=5 (recall 0.92-0.98); the only addressable band is
MF-SNR~3-7 where the 2nd stage's own recall is weak, and most cand
on_truth "wins" are stack-already-detected objects (56 of 78).

**Recommendation:**
1. Do **not** add physical post-gates — pure complexity, no gain.
2. If a 2nd stage is shipped, replace the flat RF threshold with the
   **2-D score×SNR rule** (`score>=0.20 OR (3<=mf_snr<12 AND
   score>=0.10)`) and **restrict to grizy** (drop u). Expected:
   ~3 new objects, ~26 new sightings, ~32 FP/empty-CCD — strictly
   better than the promoted thr 0.10 (8 obj/47 sight but **79 FP/CCD**)
   on the FP axis, though fewer raw objects; and far better than thr 0.50
   (0 objects). Pick between thr-0.10-flat and the 2-D rule by the
   operational FP budget: 2-D if FP/CCD must stay <~40.
3. **Strategic:** the real ceiling here is ~22 new objects at absurd FP.
   The deployable, defensible gain is ~3 objects. This is too small to
   justify a 2nd-stage pipeline on its own merits. The leverage is not in
   the operating point — it is in retraining the CNN to actually detect
   the MF-SNR 3-7 / short-trail regime (current NN recall there 0.24 /
   0.023). Without that, no operating point makes this worthwhile.
