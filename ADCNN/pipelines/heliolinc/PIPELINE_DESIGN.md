# End-to-end ADCNN -> HelioLinC discovery pipeline

Stream real diffims directly from the Butler, run the deployed two-stage ADCNN detector
(segmentation model + focal cutout CNN), produce a detection catalog, link with HelioLinC,
and crossmatch to known objects -> CONFIRMED (known) + NEW (unmatched) asteroids. No diffim
panels are written to disk; everything flows through an in-memory rolling buffer.

## Diffim source

Official DP2 diffims are stored under dataset type `difference_image` in the stage4
collection `LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4` (dims = visit / detector / band;
shape 4000x4072; WCS + VisitInfo / MJD attached). Fetch them directly; ~2 s/panel
(I/O-bound), parallelisable across many CPU workers. A region without a prebuilt diffim
can fall back to on-the-fly AlardLupton subtract from PVI (stage2) + template (stage3) via
`ADCNN.data.dataset_creation.butler_tasks.run_subtract`.

## Architecture (producer / consumer)

The LSST stack and torch envs do not coexist, so the pipeline splits into two coupled
processes sharing a bounded rolling buffer:

```
Butler query (visit, detector list)
   |
PRODUCER (lsst_distrib, N CPU workers):  butler.get("difference_image", ...)
   |   diffim + WCS + MJD  ->  bounded rolling buffer
CONSUMER (asteroid_cnn, GPU workers):  segmentation NN -> candidate extraction ->
   |                                    matched-filter measurement -> cutout CNN
   |   kept detections  ->  wcs.pixelToSky  ->  (RA, Dec, MJD, mag, band)
Detection catalog (CSV / parquet -- the ONLY persistent output)
   |
HelioLinC:  make_tracklets  ->  heliolinc (NEO heliohypo grid)  ->  link_refine
   |   per ~2-week discovery window
Crossmatch linked tracks to known SSObject / MPC ephemerides
   ->  CONFIRMED (matches known) + NEW (no match) asteroid candidates
```

## Components

| file | role |
|---|---|
| `discover_stream.py`         | producer/consumer streamer (Butler -> ADCNN -> detection catalog) |
| `adcnn_wcs.py`               | alternate Stage 1 bridge: takes a disk catalog from `ADCNN.inference.catalog` and attaches RA/Dec via the Butler WCS |
| `veres_measure_catalog.py`   | per-panel Veres trailed-PSF fit (Stage 2 measurement) |
| `butler_diasource_catalog.py`| fetch DRP `diaSource` reliability + trailLength (Stage 2 input) |
| `butler_manifest.py`         | build visit/detector manifests from the Butler |
| `trail_tracklets.py`         | trailed detections -> tracklets (Stage 3 input) |
| `crossmatch.py`              | linked tracks -> confirmed (known SSObject) / new (unmatched) |
| `build_catalog.py`           | sanity-test inputs from forced-photometry truth |
| `build_known_catalog.py`     | known-object crossmatch reference catalog |
| `hunt_parallel.sh`           | grid-sharded HelioLinC discovery run |
| `hunt_standard.sh`           | full-grid sanity run seeded by `make_tracklets` |
| `link_and_match.sh`          | end-to-end link + crossmatch |
| `neo_pipeline/`              | canonical 3-stage SLURM DAG (see `neo_pipeline/README.md`) |
| `heliolinc2/`                | vendored Heliolinc3D source + Python bindings (build artifact, gitignored) |

## Canonical entry point

```bash
bash ADCNN/pipelines/heliolinc/neo_pipeline/run.sh
```

The DAG is `01_detect.slurm` -> `02_measure.slurm` -> `03_link.slurm`, chained with
`--dependency=afterok`. Knobs (paths, models, thresholds, worker counts, HelioLinC grid)
live in `neo_pipeline/config.sh`; see `neo_pipeline/README.md` for the full reference.

## Throughput

Measured on 4 x A100 with parallel CPU prep + 6 Butler workers per shard:

- Butler diffim fetch: ~3 s/panel (collection I/O bound)
- segmentation NN: ~4 s/panel/GPU
- candidate extraction + cutout CNN: ~3 s/panel/GPU
- Aggregate (4 GPUs): ~0.6 panel/s -> ~324 s/visit (189-detector visit)

LSST cadence (30 s/visit) requires ~12 A100 GPUs (2 ampere nodes) with the current code,
or per-GPU acceleration (TRT-compiled seg net + C++ candidate extract).
