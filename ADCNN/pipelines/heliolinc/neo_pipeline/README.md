# NEO trail-tracklet discovery pipeline

End-to-end, config-driven, parallel-at-every-stage pipeline to find fast movers (NEOs) in Rubin
difference images by turning each **trailed detection into a tracklet** (a single 30 s trail =
position + on-sky velocity), so linking needs only one detection per night instead of two.

```
  butler diffims
        │  STAGE 1  detect      (GPU, asteroid_cnn)   discover_stream.py        — multi-GPU stream
        ▼          ADCNN segmentation model + focal cutout CNN -> sky detections + de-biased trail endpoints
  adcnn_dets.csv
        │  STAGE 2  measure     (CPU, lsst_distrib)   veres_measure_catalog.py  — parallel over panels
        ▼          precise per-panel-PSF Veres trailed fit -> accurate sky endpoints
  adcnn_dets_veres.csv
        │  STAGE 3a clean FP    (CPU)                 clean_fp.py               — *** the missing step ***
        ▼          score / Veres-rChiSq / length cuts  + HOOK for reliability+coherence cleaning
  adcnn_dets_clean.csv
        │  STAGE 3b link        (CPU, grid-parallel)  trail_tracklets.py + heliolinc + link_refine
        ▼          ~49k-pt NEO hypothesis grid sharded across N cores; minobsnights=2 (fast movers)
  lr.csv ──────────  STAGE 3c crossmatch  crossmatch.py  -> confirmed.csv (known) + new_candidates.csv (NEW)
```

## Run it

```bash
cd ADCNN/pipelines/heliolinc/neo_pipeline
./run.sh                                   # full chain (SLURM dependency-chained)
RUN_NAME=run_aug MANIFEST=/path/manifest.csv ./run.sh    # a different region/epoch
./run.sh --from 2        # detections already exist -> measure onward
./run.sh --only 3        # re-link only (e.g. retune linking / FP cuts) — fast iteration
```
All knobs live in `config.sh` (paths, models, thresholds, `*_WORKERS`/`N_GPUS`/`NSHARD` for speed,
HelioLinC `MINNIGHTS`/`NPT`/grid). Stages are SLURM jobs chained `--dependency=afterok`.

## Speed / scaling
- **Stage 1** multi-GPU streaming (`N_GPUS`), gated features (`gate_pmax=0.10`).
- **Stage 2** parallel over panels (`MEAS_WORKERS`), thread-pinned (1 BLAS thread/worker — avoids the
  oversubscription that tanked throughput), one Butler per worker, bounded L-BFGS-B fit.
- **Stage 3b** the hypothesis grid is embarrassingly parallel: sharded across `NSHARD` heliolinc
  processes, merged by `link_refine`. The 110k-pt general grid runs in ~20 min on 96 cores; the
  default ~49k-pt NEO grid is ~2× faster.
- To process **more data**: run one pipeline per region/epoch manifest (independent `RUN_NAME`s) —
  fully horizontal.

## The FP-cleaning step (known TODO — see `clean_fp.py`)
On real diffims the trailed-detection set is FP-dominated (~18k/night of subtraction residuals),
which drowns the linker in chance alignments (verified: 15k spurious 2-night clusters, all huge
RMS, 0 real). The cheap cuts (score, Veres rChiSq, length) are in place; the real reducers to add,
in the `clean_fp.py` hook, are: (1) diaSource real/bogus **reliability** join, (2) trail-**coherence**
cut, (3) de-dup. This is the gate between "pipeline runs" and "pipeline finds NEOs".

## Scope note
Trail-tracklets only help **fast movers (≥~1 deg/day → ≥6 px trails)**; slow main-belt objects make
sub-pixel trails and need standard 2-detection-per-night pairing instead.
```
trail_px(true) ≈ 6.25 × speed(deg/day)      (30 s exposure, 0.2″/px)
```
