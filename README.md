# ADCNN — Asteroid Trail Detection in LSST Difference Images

Two-stage detector for asteroid trails in LSST difference images, plus linking to
discover/confirm asteroids:

```
Butler diffim → segmentation model (segmentation + orientation) → candidates → focal-loss cutout-CNN 2nd stage
              → scored detections → (RA,Dec,MJD) → HelioLinC linking → new/confirmed asteroids
```

## Deployed model
The production model is **reg2**: the segmentation model (`UNetResSEOrientHough`, half-width)
trained with `lambda_orient=0` + dropout 0.15 + weight-decay 1e-4 + intensity-augmentation +
D4 augmentation, plus a **focal-loss cutout-CNN** second-stage false-positive filter. The
segmentation stage reaches **96.0% objectwise recall on test_5sigma** (real fire@truth 77%);
the cutout-CNN stage removes ~56% of false positives at 95% recall (test_5sigma). Weights
live in `models/`:
- `models/segmentation_model.pt` — the TorchScript segmentation model (stage 1)
- `models/cnn_postproc.pt` — the focal-loss cutout-CNN false-positive filter (stage 2)

## Entry points (`ADCNN/pipelines/`)
| Command | Purpose |
|---|---|
| `python -m ADCNN.pipelines.make_sim_data`    | build SIMULATED (injected-trail) train/test diffim datasets from the Butler |
| `python -m ADCNN.pipelines.make_real_data`   | build the REAL-asteroid test diffim dataset from the Butler |
| `python -m ADCNN.pipelines.train_end_to_end` | train the full detector: segmentation model (reg2 recipe) + cutout-CNN stage-2 |
| `python -m ADCNN.pipelines.run_inference`    | run segmentation model + cutout CNN on one diffim h5 → detection catalog CSV |
| `python -m ADCNN.pipelines.make_eval_catalogs` | run the optimized multi-GPU engine on the eval test sets → catalogs + metrics |

Run each with `--help`. Butler entries use the `lsst_distrib` env; train/inference use
the `asteroid_cnn` (torch) env.

## Inference engine + catalog evaluation
`ADCNN.inference.catalog.build_detection_catalog[_multigpu]` is the end-to-end engine:
images → segmentation model → candidates → trail measurement + 48² cutout → cutout-CNN → one CSV
row per detection (measured trail geometry `x,y,beta,length` + flux + `score`, plus
visit/detector/band routing keys for HelioLinC). It is optimized to use the whole GPU node — GPU
inference with parallel CPU prep (`ADCNN_PREP_WORKERS`), pipelined across all GPUs, candidate
features + the cutout CNN in a process pool — reaching **~1.0–1.4 s/panel images→catalog on 4×A100** (a full ~189-detector visit in
~3–4 min; profiled 7.3→1.4 s/panel, all safe/accuracy-preserving). Evaluation is then pure
catalog analysis: `ADCNN.evaluation.catalog_match.evaluate_catalog(measured, truth)` does
trail-overlap matching (any-pixel-overlap criterion, all-trails denominator, fixed pre-chosen
`tol_px`) → TP/FP/FN + the flagged truth catalog for the `evaluation.plots` completeness/
histograms. No training and no threshold tuning happen at evaluation time.

## Package layout
- `ADCNN/core/`       — `model.py` (UNetResSE backbone), `detector.py` (segmentation model), `losses.py` (AFTL + orientation)
- `ADCNN/data/`       — `dataset.py`, `preprocessing.py` (3-channel build / MAD-sigma / orientation maps);
                        `dataset_creation/` (`simulate`, `build_real`, `butler_tasks`, `photometry`, `realistic_trail`)
- `ADCNN/training/`   — `train.py` (segmentation model trainer), `cnn_postproc.py` (stage-2 cutout-CNN trainer), `ema.py`
- `ADCNN/inference/`  — `predict.py` (sliding-window segmentation model), `candidates.py`, `matched_filter.py`,
                        `features.py` (candidate feature extraction), `cnn_postproc.py` (stage-2 FP-filter CNN apply),
                        `catalog.py` (images→catalog engine), `export.py`
- `ADCNN/evaluation/` — `detection.py` (object-level metrics), `geometry.py` (mask/component primitives),
                        `catalog_match.py` (trail-overlap matching), `plots.py` (notebook viz),
                        `architecture.py` (paper architecture figures)
- `models/`           — deployed weights (above)
- `Evaluation/`       — `Evaluation.ipynb` (synthetic) + `Evaluation_Real.ipynb` (real), evaluating `models/`
- `experiments/heliolinc/` — HelioLinC linking suite + the ADCNN→HelioLinC bridge + `PIPELINE_DESIGN.md`

## Branches
- `diffim` — this production diffim pipeline (current).
- `direct_image` — the earlier direct-image detection phase (archived).
