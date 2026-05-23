# Multi-visit tracklet linking as an FP-suppression lever for the v7 CNN 2nd stage

Scope: CPU-only re-analysis of existing dumps under
`experiments/diffim_runs/test_real/results/`. No tracked files modified, no
GPU/SLURM. Scripts: `linking_analysis.py`, `fp_tracklet_model.py`. Raw
console output captured in `_numbers.txt`, `_fp_numbers.txt`.

## TL;DR

**Multi-visit linking is NOT the lever that rescues this operating point —
it is killed on the true-positive side before the FP benefit can be
collected.** All **7** of the NN-only objects (the actual scientific prize:
asteroids the 5σ stack never caught) are detected by the v7 NN in **exactly
one visit each**. Any rule of the form "require ≥2 NN detections that link
into a tracklet" removes **7/7** of them. The combined per-object recall
gain over the stack collapses from **+7 → +0**. The 46 stack-missed
sightings fare better (16–33 retained depending on the linking model) but
those are *sightings of objects the stack mostly also sees*, so they do not
add new objects. The FP side of linking does work in principle (≈30×–340×
suppression, residual ≪1 false multi-night track per field) but it requires
**3-epoch cross-night orbit attribution**, not a cheap 2-of-2 positional
gate — and it is moot here because the prize does not survive.

Bottom line: with the v7 NN as it is, linking cannot be the dominant lever
for shrinking the FP price *without first making the NN fire on the same
object in ≥2 visits*. The recommendation is to attack the FP price directly
(reranker / threshold) and treat linking as a downstream confirmation step
that only pays off once per-object multiplicity exists.

---

## Data and joins (verified)

- `per_sighting.csv`: 2230 sightings, 288 unique objects. `image_id` is a
  perfect 1:1 key into `DATA_DIFFIM/test_real/test.csv` (ObjID and visit
  match 100% on the join), which supplies per-sighting `ra,dec`.
- `visit` in `per_sighting` == `FieldID` in
  `DATA/sv_fast_movers_for_karlo_fast_with_pixels_rerun.csv` (2026 of the
  visits join; `fieldMJD_TAI` recovered for 100% of sightings via this
  key). Each `visit` is a distinct exposure/pointing at a distinct time;
  the same `ObjID` recurring across visits is, by construction, an
  ephemeris-consistent track.
- `per_panel_fp.csv`: 119 empty panels, NN FP/panel mean **81.7**, median
  **67**, max 292, total **9718**, **0** clean panels. The 119 empties are
  **115 distinct visits / 119 distinct (visit,detector) / 87 distinct
  detectors / 36 dates**; only 4 visits contain >1 empty panel and never on
  the same detector. There is **no revisited empty pointing** in this dump.
- CCD geometry: ≈4096×4004 px @ 0.20″/px ≈ **0.228°×0.222°**, area
  **0.0506 deg²**.
- Real fast-mover sky rates (truth ephemeris): min 1.0, median 1.5, 95th
  4.2, max **47.0** deg/day; exposure 30 s. Successive-visit time gaps
  along real tracks: median **16 min**, 90th **1.08 d**, max 22 d.

---

## Part 1 — Multiplicity of the real prize

### 1a. The 7 NN-only objects (stack NEVER detected — the new objects)

| ObjID | total sightings | NN-detected distinct visits |
|---|---:|---:|
| K11UJ2Y | 10 | **1** |
| K14QT5O | 4 | **1** |
| K20B01Y | 9 | **1** |
| K20P02Y | 17 | **1** |
| K21R03Y | 8 | **1** |
| K21T13V | 13 | **1** |
| K25M89D | 2 | **1** |

**Every one of the 7 is a single-NN-detection object.** It does not matter
which linking definition we use (strict same-class, or "link using *any*
NN detection on the track"): **0 of 7 survive a ≥2-detection requirement.**
The +7 object gain over the stack is entirely carried by single-shot NN
detections. This is the decisive result.

### 1b. The 46 stack-missed NN-recovered sightings (pure 2nd-stage gain)

The 46 sightings span **35 distinct objects**. Retention under two models:

| Linking model | objects kept | sightings kept |
|---|---:|---:|
| **A** — ≥2 NN-recovered *stack-missed* dets, distinct visits | 5 / 35 | **16 / 46** |
| **B** — ≥2 NN dets of *any* kind on the same track (NN run on every visit) | 22 / 35 | **33 / 46** |

Model B is the physically fair one if the NN is scored on every visit (a
stack-also-detected NN hit is still a legitimate tracklet point). Even so,
13 sightings / 13 objects are lost, and crucially **the 35 objects here
overlap heavily with objects the stack already detects in other visits** —
they add few/no *new objects*, so this gain is sightings-level, not the
object-level prize.

### 1c. Object-level recall after linking

| | single-visit (current) | ≥2-NN linked (model B) |
|---|---:|---:|
| stack objects | 189 | 189 |
| + NN-only objects | **+7** | **+0** |
| combined | **196** | **189** |

Across all 86 NN-recovered objects, only 43 have ≥2 NN-detected distinct
visits and 26 have ≥3; the per-object NN-detection multiplicity
distribution is dominated by the "1" bin (43 of 86 objects). The v7 NN is a
*sporadic* detector per track — exactly the regime in which a
multi-detection requirement is most destructive.

**Selection-bias / blind-discovery caveat (honest statement):** these 288
objects are *known, already-discovered* fast movers; "ephemeris-consistent
across visits" is true *by construction* for any two sightings of the same
ObjID, which is why model B looks as good as it does. A real blind linker
does **not** have ObjID labels — it must *infer* the track from the
detections alone. With only ~1 NN detection per object and ~67–82 FP per
CCD drowning each real point, a blind linker would in practice fail to
assemble these tracks at all. The numbers above are therefore an
**optimistic upper bound** on TP retention; the real blind-discovery
retention of the +7 prize is **0** (one point cannot make a track) and of
the 46 is well below the model-B 33.

---

## Part 2 — FP side: can isolated residuals form tracklets?

### 2a. Structural argument

Empty-panel FP are isolated per (visit,detector): each empty panel is a
distinct pointing at a distinct time with **no shared ObjID/track**. Two FP
from two different empty panels sit at unrelated sky positions and times;
there is no single ephemeris both must satisfy. In *this dump* the empties
are 115 distinct visits with no revisited pointing, so the **realized**
number of FP→FP link candidates is **0**. The quantitative model below is
the survey-realistic upper bound (what you'd get if the same empty field
were revisited as a real cadence would do).

### 2b. A 2-point positional gate does NOT suppress FP

With the *measured* density (median 67 FP/CCD) and the *plausible* rate
band (1–47 deg/day), the intra-night search corridor swept by the allowed
motion in any realistic revisit interval (15 min – 4 h) covers an
**order-1 fraction of the CCD** (corridor fraction → 1.0). Consequently the
expected number of spurious intra-night "tracklets" per revisited field is
**huge (≈k² → thousands)**. A 2-of-2 positional association is a *weak*
filter at this FP density — this is the key methodological correction: the
naïve "require 2 detections in a window" intuition fails here.

### 2c. The real lever: 3-epoch cross-night attribution

The decisive suppression comes from requiring the tracklet to **attribute
to a kinematically consistent orbit across ≥3 epochs**: a spurious tracklet
has a *random* implied rate **and** direction and must, by chance, find a
later-night spurious tracklet that continues the **same great-circle,
constant-rate** motion to arcsec position tolerance and ~percent rate
tolerance. Modeled with the measured FP density, CCD geometry and visit
cadence:

| astrometric tol σ_pos | residual FALSE 3-epoch tracks / field | FP suppression vs single-visit (67/CCD) |
|---|---:|---:|
| 0.3″ | ≈ 0.20 | **≈ 340×** |
| 0.5″ | ≈ 0.54 | **≈ 123×** |
| 1.0″ | ≈ 2.2 | **≈ 31×** |

So if the prize survived, 3-epoch linking would crush the FP price by
**~30×–340×** (order of magnitude: **~10²**), bringing residual false
multi-night tracks to **≲1 per field** from the current ~67–82 FP/empty
CCD. The FP physics is favourable; the linking lever is real **on the FP
axis**.

---

## Part 3 — Net science verdict

| metric | single-visit (current) | ≥2-NN linked |
|---|---|---|
| new objects vs stack | **+7** | **+0** (7/7 lost) |
| 2nd-stage sightings (of 46) | 46 | 16 (model A) / 33 (model B, optimistic) |
| FP / empty CCD | ~67–82 | ≲1 false 3-epoch track/field (≈30–340× down) |

**Is multi-visit linking the dominant lever to shrink the FP price here?
No.** It would be a spectacular FP lever (~10² suppression) *if there were
multi-visit NN detections to link*, but the v7 NN produces the +7 prize
objects from **single** detections, so the linking requirement zeroes the
unique scientific gain before the FP benefit is realized. Linking does not
shrink the FP price *for free*; it trades essentially the entire object-
level prize for it. The binding constraint is **NN per-object detection
multiplicity / per-sighting recall**, not FP geometry.

### Recommendation

1. **Do not gate the current v7 output on multi-visit linking.** It
   eliminates 7/7 of the new objects. The FP price must be attacked
   directly first (the V2 RF reranker / threshold work already in flight),
   keeping the operating point single-visit until per-object recall rises.
2. **Linking is the right *downstream confirmation* step, but only after**
   the NN reliably fires on the same object in ≥2 (ideally ≥3) visits.
   Concretely: push NN per-sighting recall on stack-missed sightings (the
   detector is at 5.0% there) so that real objects accumulate ≥2–3 hits
   along their tracks; *then* a 3-epoch HelioLinC-style attribution gives
   the ~10² FP suppression essentially free of TP loss.
3. **The FP physics is on our side** — 3-epoch attribution, not 2-of-2,
   is what matters. Any future linking design should be specced as
   ≥3-epoch orbit attribution with arcsec position + ~percent rate
   tolerance, not a 2-point window (which is a weak filter at 67 FP/CCD).

### What a rigorous follow-up needs (additional dump)

The current dumps give per-sighting truth-joined positions but **no NN
*candidate* sky coordinates** (the (RA,Dec) of every NN detection,
including FP, per visit/detector). A real linking study requires:

- **`nn_candidates.csv`**: for every NN detection on every panel (asteroid
  *and* empty), the predicted/centroided **RA, Dec, visit, detector,
  MJD, score**, with NO ObjID label (blind). This lets one run an actual
  tracklet/attribution linker over the real FP cloud + real asteroid hits
  and measure realized TP retention and FP survival end-to-end, instead of
  the analytic upper/lower bounds used here.
- Ideally a **revisited-empty-field** set (same pointing across ≥3 visits)
  so the FP-attribution rate is *measured*, not modeled — the present
  empties have no revisited pointing.
