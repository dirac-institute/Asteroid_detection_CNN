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
build_manifest.py     tracts + day_obs window      -> manifest.csv     (registry only)
sn_detect.slurm       discover_stream.py (ADCNN)    -> adcnn_dets.csv   (ampere GPU; resumable)
build_known_catalog   preloaded_ss_object_visit     -> known.csv        (LSST stack env; mpcorb crossmatch ref)
mask_flags.py         LSST diffim mask planes       -> *_masked.csv     (TP-safe artifact cut, ~15-20% FP)
trail_state_link.py   (pos, trail-velocity) cluster -> tracks.csv       (single-night; physical_check; crossmatch)
```

`tracks.csv` columns: night, ndet, nvisit, arc_hr, rms_arcsec (linear fit), speed_degday, ra, dec,
check, match_obj, match_frac, status (CONFIRMED = known recovery / NEW = uncatalogued candidate).

## How to run (every time)

```bash
# 1. manifest for the discovery field (LSST stack env). The canonical field is the ecliptic NEO band:
python -m ADCNN.pipelines.heliolinc.build_manifest \
    --tracts 8487-8493,8729-8735 --day-start 20250709 --day-end 20250723 \
    --out ADCNN/pipelines/heliolinc/run_band/manifest.csv

# 2. run the whole pipeline (one orchestrator; resubmits ADCNN through ampere preemption automatically):
RUN=$PWD/ADCNN/pipelines/heliolinc/run_band \
    sbatch --export=ALL,RUN ADCNN/pipelines/heliolinc/sn_run.slurm
# -> RUN/tracks.csv
```

## Validating a NEW candidate (before calling it real)

A `status==NEW` row is a candidate only after: it passes `physical_check` (already enforced), survives
a randomized-trail-angle null test for its night (false-link rate ~0), its detections are real (good
CNN score, clean trails, not on artifact masks), and it matches no `known.csv` object. Short single-
night arc -> report as a candidate for follow-up, not a determined orbit.

## Components

| file | role |
|---|---|
| `build_manifest.py`     | tracts + window -> diffim manifest (leakage-excluded) |
| `discover_stream.py`    | Butler diffim -> ADCNN seg + cutout CNN -> sky detections + trail endpoints |
| `build_known_catalog.py`| `preloaded_ss_object_visit` -> known-object ephemerides for crossmatch |
| `mask_flags.py`         | LSST diffim mask-plane FP filter (SPIKE/SAT/CR/STREAK/DETECTED_NEGATIVE/...) |
| `trail_state_link.py`   | **canonical linker** — single-night (pos, trail-velocity) clustering + physical_check + crossmatch |
| `sn_detect.slurm`       | Stage-1 ADCNN SLURM job (ampere, resumable) |
| `sn_run.slurm`          | **canonical orchestrator** — detect (resubmit through preemption) -> known -> mask -> link |

## Superseded (kept for reference / multi-night only)

`samenight_realdata.py` (heliolinc same-night with the hypothesis grid + parallel link_purify),
`samenight_link.py`, the heliolinx binaries under `external/heliolinx/`, `neo_pipeline/`. These hit
the candidate-explosion wall on ADCNN's dense unculled input; `trail_state_link.py` replaces them for
single-night discovery. See memory `samenight-trailstate-pipeline` / `exact-bottleneck-verified`.
```
