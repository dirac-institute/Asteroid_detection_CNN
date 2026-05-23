# End-to-end ADCNN -> HelioLinC discovery pipeline (Butler-streaming, 4-GPU, no disk)

## Goal
Stream real diffims directly from the Butler, run the trained ADCNN prototype
(reg2 v7 + reg2 neg5 RF), produce a detection catalog, link with HelioLinC, and
crossmatch to known objects -> CONFIRMED (known) + NEW (unmatched) asteroids.
Never persist the heavy 117 MB/panel images; process in-memory; go as fast as possible.

## Diffim source (RESOLVED)
- Official DP2 diffims ARE stored: dataset type `difference_image` in the **stage4**
  collection `LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4`
  (dims = visit,detector,band; shape 4000x4072; carries WCS + VisitInfo/MJD).
- => FETCH them (no AlardLupton subtraction needed). ~19.5s/fetch (I/O-bound, first
  fetch incl. butler overhead). Parallelize fetches across many CPU workers.
- Fallback (if a region lacks stage4 diffims): subtract on-the-fly from PVI(stage2)+
  template(stage3) using the existing simulate_inject_diffim/driver.py subtraction.

## Architecture (producer/consumer, bounded buffer = memory-efficient)
Cross-env reality: Butler lives in lsst_distrib; the scripted v7 needs torch
(asteroid_cnn). So split into two coupled processes sharing a BOUNDED rolling buffer
(node-local lscratch or shared mem) — never the full dataset on disk:

    Butler query (region + time window) -> list of (visit,detector)
        |
  PRODUCER (lsst_distrib, N CPU workers): butler.get(difference_image, stage4)
        |  -> diffim array + WCS + MJD -> bounded rolling buffer (keep ~M panels)
  CONSUMER (asteroid_cnn, 4 GPU workers): reg2 v7 -> compute_v2_features -> reg2 RF
        |  -> kept candidate centroids -> wcs.pixelToSky(x+xy0,y+xy0) -> (RA,Dec,MJD,mag,band)
  detection catalog (lightweight CSV/parquet: the ONLY persistent output, KB-MB)
        |
  HelioLinC: make_tracklets -> heliolinc (NEO grid heliohypo_neo01) -> link_refine
        |  run per ~2-week window (heliolinc's design horizon; mjdref = window mid)
  crossmatch linked tracks to known SSObject/MPC ephemerides
        -> CONFIRMED (matches known) + NEW (no match) asteroid candidates

## Throughput / "as fast as possible"
- Bottlenecks: (1) diffim fetch ~19s I/O (parallelize across ~60 CPU workers ->
  ~3/s); (2) compute_v2_features (CPU/GPU, the 72-feature MF/PCA suite — seconds/panel
  with many candidates). The 4 GPUs handle v7 inference cheaply (~0.5s/panel).
- Recipe: ~60 producer fetchers + 4 GPU consumers + a CPU pool for v2-features; bounded
  ~200-panel buffer (~26 GB RAM/lscratch, rolling). Scales with more CPU nodes for fetch.
- For a focused discovery run, process a sky region over a ~2-week window (heliolinc's
  horizon) -> a few thousand panels -> hours, not days.

## Reused components (all already built)
- `experiments/explore_simreal_gap/stream_producer.py` (rolling-buffer producer pattern;
  swap "subtract" for "butler.get(difference_image, stage4)").
- reg2 v7 (`pilot_v7_reg2/ckpts/v7_reg2_best_scripted.pt`) + reg2 RF
  (`rf_postproc_v2_reg2_neg5.pkl`) + `predict_panel_overlap_3ch_full` + `compute_v2_features`
  + `apply_rf_v2`.
- `experiments/heliolinc/adcnn_wcs.py` (WCS pixelToSky -> RA/Dec).
- `experiments/heliolinc/heliolinc2/` (make_tracklets/heliolinc/link_refine, built).
- `build_catalog.py` (visit MJD + colformat).

## Build steps (next)
1. `pipeline_producer.py` (lsst): parallel butler.get(difference_image, stage4) for a
   visit/detector list -> rolling buffer of (diffim, wcs, mjd) panels + manifest.
2. `pipeline_consumer.py` (asteroid_cnn, 4 GPU): consume buffer -> reg2 v7+RF candidates
   -> RA/Dec -> append detection catalog.
3. `pipeline_link.py`: window the catalog -> make_tracklets/heliolinc/link_refine ->
   crossmatch to SSObject -> new/confirmed report.
4. SLURM: 1 ampere 4-GPU node (consumer) + roma CPU node(s) (producer) sharing scratch.

## Status of validation (parallel track)
- HelioLinC links real asteroids PURELY (neo grid: 17 objects, 0 false links) on the
  2-week truth catalog. ADCNN-detection -> RA/Dec bridge + ADCNN-fed link test in progress.
