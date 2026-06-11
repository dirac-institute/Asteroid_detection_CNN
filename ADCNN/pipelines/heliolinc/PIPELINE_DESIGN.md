# Canonical same-night NEO-discovery pipeline

Detect asteroid trails in LSST difference images with ADCNN, then link them **within a single
night** into moving-object tracks and crossmatch to the known catalogue — so an uncatalogued track
is a NEW near-Earth-object candidate. This is the canonical pipeline; run it the same way every time.

## Why single-night trail-state linking (not heliolinc)

heliolinc links by scanning a grid of heliocentric `(r, ṙ)` hypotheses, because a classical detection
gives only a *position* — the grid exists to *infer* each tracklet's velocity. On ADCNN's dense
output that grid search enumerates **millions** of impure candidate clusters that then each need a
Method-of-Herget orbit fit — verified intractable (~months; ~2.4 h even at 110× parallel), and it
*missed* the known NEO 2025 NY2.

**Key insight:** an ADCNN *trail* measures the velocity directly (its two endpoints over the
exposure). So the hypothesis grid is unnecessary. We propagate each detection to a reference time
using its **own trail velocity** and cluster once in 4-D `(RA@tref, Dec@tref, vRA, vDec)`. Same-object
detections collapse to one point; false positives scatter. **O(N log N), seconds, no explosion** —
and it recovers 2025 NY2 cleanly. A physical check (linear motion + each trail's PA/speed matching
the inter-epoch motion) rejects false links; a randomized-trail null test confirms ~0 false rate.
Tradeoff: this targets *trailed* fast movers (the NEOs we want) and yields short-arc same-night
*candidates* (for follow-up), not determined orbits.

## Stages (run via `sn_run.slurm`)

```
build_manifest.py     tracts + day_obs window      -> manifest.csv     (registry only; LSST stack env)
sn_detect.slurm       discover_stream.py (ADCNN)    -> adcnn_dets.csv   (ampere GPU; resumable; op-point
                                                                          = val2-calibrated cnn_postproc.json)
build_known_catalog   preloaded_ss_object_visit     -> known.csv        (LSST stack env; mpcorb crossmatch ref)
mask_flags.py         LSST diffim mask planes       -> *_masked.csv     (TP-safe artifact cut, ~15-20% FP)
trail_state_link.py   (pos, trail-velocity) cluster -> tracks.csv       (single-night; physical_check;
                                                                          2-visit bound-orbit check; crossmatch)
```

`tracks.csv` columns: night, ndet, nvisit, **n_epochs, tier** (`3+visit`/`2visit`), arc_hr,
rms_arcsec (linear fit), speed_degday, ra, dec, check, match_obj, match_frac, status
(CONFIRMED = known recovery / NEW = uncatalogued candidate).

## How to run (morning run — one command)

`sn_run.slurm` is night-parameterized: give it the night (`DAY` = day_obs) and the field (`TRACTS`);
it builds the manifest, runs ADCNN through ampere preemption, then known -> mask -> link.

```bash
RUN=$PWD/ADCNN/pipelines/heliolinc/run_<night> DAY=20250706 TRACTS=8731 \
    sbatch --export=ALL,DAY,TRACTS,RUN ADCNN/pipelines/heliolinc/sn_run.slurm
# -> RUN/tracks.csv   (TRACTS accepts lists/ranges, e.g. 8487-8493,8729-8735)
```

## Two confidence tiers (and the open FP-linkage problem)

`physical_check` emits two tiers, distinguished by `n_epochs`:

- **`3+visit`** (>=3 distinct epochs): full linear-motion residual test. **Defensible today** — recovers
  the real NEO 2025 NY2 cleanly (linear RMS 0.67", null 0/50). This is the shippable result.
- **`2visit`** (exactly 2 epochs, the Heinze minimum): the linear test is degenerate for 2 points, so
  the tier adds a tighter trail-PA tolerance, a trail-vs-trail velocity agreement, and a **per-candidate
  bound-orbit test** (`orbit_check.py`: Method of Herget + Lambert, using the trail velocities). The
  orbit code is correct (recovers NY2's a~1.09 AU, e~0.07 orbit) **but on raw ADCNN output 2-visit
  false-linkage ~= signal (~375/night)** — the FP density is too high. Thinning it makes 2-visit
  defensible (ADCNN score>=0.9 -> null 0.3/night, NY2 kept); the stack real/bogus path does NOT help
  (it misses fast trails / flags them bogus). See memory `two-visit-not-defensible`.

## FPP-calibrated operating point (the ADCNN score floor as a function of significance)

The ADCNN score floor is set from a **false-alarm-rate budget**, not by hand. `calibrate_link_fpp.py`
Monte-Carlos the false-linkage rate `lambda_FP` (false tracks/night) vs the score floor, using a
**cross-epoch-permutation null** (each visit gets an independent random rigid sky offset -> destroys real
cross-visit objects, preserves the real FP structure), then extrapolates `lambda_FP(S)` to the requested
false-alarm rate. One-sided **3 sigma = lambda_FP <= 1.35e-3/night**.

Result (calibrated, pooled over 11 night-fields; persisted in `link_fpp.json`):

| tier | `lambda_FP ~` | S* for 3 sigma | real NEO at S* |
|---|---|---|---|
| 2-visit | `rho^2.0` | ~0.97 | **lost** (caps ~2 sigma) |
| **3-visit** | `rho^3.5` | **~0.80** | **NY2 recovered** |

**=> The defensible 3-sigma same-night discovery operating point is the 3+visit tier at ADCNN
`--score-min 0.80`** (wired into `sn_run.slurm`; override with `SCORE_MIN=`). 2-visit cannot reach 3 sigma
at this FP density, so 2visit tracks are candidate-grade only. **Cost = cadence:** 3-visit needs fields
with >=3 same-night revisits (dense-cadence fields like NY2's; the spread-out band recovers 0).

## Validating a NEW candidate (before calling it real)

A `status==NEW` row is a candidate only after: it passes `physical_check` (already enforced), survives
the randomized-trail-angle null test for its night (`validate_candidate.py`; false-link rate ~0), its
detections are real (good CNN score, clean trails, not on artifact masks), and it matches no `known.csv`
object. A short single-night arc is a candidate for follow-up, **not** a determined orbit.

## Components

| file | role |
|---|---|
| `build_manifest.py`     | tracts + window -> diffim manifest (leakage-excluded) |
| `discover_stream.py`    | Butler diffim -> ADCNN seg + cutout CNN -> sky detections + trail endpoints (op-point from val2 sidecar) |
| `build_known_catalog.py`| `preloaded_ss_object_visit` -> known-object ephemerides for crossmatch |
| `mask_flags.py`         | LSST diffim mask-plane FP filter (SPIKE/SAT/CR/STREAK/DETECTED_NEGATIVE/...) |
| `trail_state_link.py`   | **canonical linker** — single-night (pos, trail-velocity) clustering + physical_check + crossmatch |
| `orbit_check.py`        | per-candidate bound-orbit test (Method of Herget + Lambert) for the 2-visit tier |
| `validate_candidate.py` | randomized-trail null test + per-object recovery report for a night |
| `calibrate_link_fpp.py` | MC calibration of the ADCNN score floor to a 3-sigma linkage FPP budget -> `link_fpp.json` |
| `sn_detect.slurm`       | Stage-1 ADCNN SLURM job (ampere, resumable) |
| `sn_run.slurm`          | **canonical orchestrator** — manifest -> detect (resubmit through preemption) -> known -> mask -> link |
| `butler_diasource_catalog.py` | (auxiliary) stack diaSources + real/bogus reliability, for the deferred FP-density work |

## Notes

- The heliolinc hypothesis-grid path (`samenight_realdata.py`, `neo_pipeline/`, the `external/heliolinx/`
  binaries) was **removed** — it hit the candidate-explosion wall on ADCNN's dense unculled input and is
  superseded by `trail_state_link.py`. History: memory `exact-bottleneck-verified`.
- The ADCNN operating point is **not** hardcoded: `discover_stream.py` reads the val2-calibrated
  `threshold` from the `models/cnn_postproc.json` sidecar (override with `--cnn-thr`).
```

## Code boundary — what is ours vs Ari Heinze's (heliolinx)

Keep these strictly separate; do not blur them.

- **Ari Heinze's code (NOT ours):** everything under `external/heliolinx/` (binaries + C++ source) and
  `external/heliolinx-aux/` (test data, hypothesis grids, Earth ephemeris, ObsCodes). It is **vendored
  read-only, gitignored** (`.gitignore` line `external/`; 0 files tracked), and **never edited by us**.
  Treat it as an upstream dependency: we only *invoke* the binaries and *consume* his shipped aux files.
- **Our code (ours):** everything under `ADCNN/`. The only bridge to his binaries is the thin wrapper
  `run_heliolinx.py`, which calls them by absolute path (`make_tracklets → heliolinc → link_purify`; use
  plain `heliolinc`, not `heliolinc_omp` — sum-format skew) for the ADCNN-vs-stack head-to-head
  (`h2h-heliolinx-campaign`). It reimplements none of his algorithm.
- Rule: never fork his source into `ADCNN/`, never commit anything from `external/`. Linker changes go in
  our wrapper (params/inputs) or upstream in his repo separately.
