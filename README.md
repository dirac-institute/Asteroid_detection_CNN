# ADCNN — Asteroid Trail Detection in LSST Difference Images

Two-stage detector for asteroid trails in LSST difference images, plus linking to
discover/confirm asteroids:

```
Butler diffim → v7 CNN (segmentation + orientation) → candidates → RandomForest 2nd stage
              → scored detections → (RA,Dec,MJD) → HelioLinC linking → new/confirmed asteroids
```

## Deployed model
The production model is **reg2**: v7 (`UNetResSEOrientHough`, half-width) trained with
`lambda_orient=0` + dropout 0.15 + weight-decay 1e-4 + intensity-augmentation + D4
augmentation, plus the **neg5 RandomForest** second stage. Verified: **96.0% objectwise
recall on test_5sigma**, real fire@truth 77%; on the real operating curve it beats the
prior model (more TP at equal FP). Weights live in `models/`:
- `models/v7_diffim_scripted.pt` — the TorchScript v7
- `models/rf_postproc.pkl` — the neg5 RandomForest

## Entry points (`ADCNN/pipelines/`)
| Command | Purpose |
|---|---|
| `python -m ADCNN.pipelines.make_sim_data`    | build SIMULATED (injected-trail) train/test diffim datasets from the Butler |
| `python -m ADCNN.pipelines.make_real_data`   | build the REAL-asteroid test diffim dataset from the Butler |
| `python -m ADCNN.pipelines.train_end_to_end` | train the full detector: v7 (reg2 recipe) + RandomForest |
| `python -m ADCNN.pipelines.run_inference`    | run v7 + RF on diffim panels → scored candidate detections |

Run each with `--help`. Butler entries use the `lsst_distrib` env; train/inference use
the `asteroid_cnn` (torch) env.

## Package layout
- `ADCNN/core/`       — `model.py` (UNetResSE backbone), `diffim_model.py` (v7)
- `ADCNN/data/`       — `diffim_dataset.py`; `dataset_creation/` (Butler sim + real data builders)
- `ADCNN/training/`   — `diffim_train.py` (v7 trainer), `ema.py`
- `ADCNN/inference/`  — v7 prediction, candidate extraction, matched-filter features, RF
                        post-processor (`diffim_postproc_v2`), RF training (`rf_train`), export
- `ADCNN/evaluation/` — object/pixel metrics, geometry, real-data eval, threshold scan
- `models/`           — deployed weights (above)
- `Evaluation/`       — `Evaluation.ipynb` (synthetic) + `Evaluation_Real.ipynb` (real), evaluating `models/`
- `experiments/heliolinc/` — HelioLinC linking suite + the ADCNN→HelioLinC bridge + `PIPELINE_DESIGN.md`

## Branches
- `diffim` — this production diffim pipeline (current).
- `direct_image` — the earlier direct-image detection phase (archived).
