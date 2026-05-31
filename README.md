# ADCNN — Asteroid-Trail Detection in LSST Difference Images

Rendered evaluation notebooks (always available; GitHub's notebook viewer is occasionally flaky on large notebooks):
[`Evaluation.ipynb`](https://nbviewer.org/github/dirac-institute/Asteroid_detection_CNN/blob/heliolinc-discovery/Evaluation/Evaluation.ipynb)
· [`Evaluation_Real.ipynb`](https://nbviewer.org/github/dirac-institute/Asteroid_detection_CNN/blob/heliolinc-discovery/Evaluation/Evaluation_Real.ipynb)

Two-stage detector for asteroid trails in LSST difference images, plus downstream linking
for discovery of new objects:

```
Butler diffim → segmentation model (UNet + orientation + Hough aggregator)
              → candidate components + matched-filter measurement
              → focal-loss cutout-CNN false-positive filter
              → detection catalog (with measured trail geometry)
              → (RA, Dec, MJD)   → HelioLinC linking → confirmed / new asteroids
```

## Deployed models

The production checkpoints live in `models/`:

| file | role |
|---|---|
| `models/segmentation_model.pt` | TorchScript segmentation model (stage 1) |
| `models/cnn_postproc.pt`       | focal-loss cutout-CNN false-positive filter (stage 2) |
| `models/cnn_postproc.json`     | architecture sidecar + calibrated CNN threshold |

Stage 1 reaches **96.0 % object-wise recall on the simulated 5σ test set**; stage 2 cuts
~56 % of stage-1 false positives at 95 % recall. The combined 5σ-stack + ADCNN detector is
calibrated to **100 FP/panel** on the calibration set
(`ADCNN.training.cnn_postproc.calibrate_combined_threshold`).

## Entry points (`python -m ADCNN.pipelines.<name>`)

| command | purpose |
|---|---|
| `make_sim_data`      | build the simulated injected-trail train / val / test datasets from the Butler |
| `make_real_data`     | build the real-asteroid test diffim dataset from the Butler |
| `train_end_to_end`   | train the segmentation model, then the focal cutout CNN, and persist both with a sidecar JSON |
| `make_eval_catalogs` | score the test sets with the deployed models, emit detection catalogs and metrics |

Single-h5 inference:

```bash
python -m ADCNN.inference.catalog --h5 <test.h5> --panels <panels.csv> --n-gpus 4 --out detections.csv
```

Real-data streaming (Butler → catalog, no panels written to disk) — two-env pipeline,
producer in the LSST stack env, consumer in the torch env:

```bash
python -m ADCNN.inference.stream_real_butler --workers 6 \
  | python -m ADCNN.inference.stream_real_inference --out test_real_detections.csv
```

Butler entries (`make_sim_data`, `make_real_data`, `stream_real_butler`) need the LSST stack
env (`loadLSST.sh` + `setup lsst_distrib`); training and inference need the `asteroid_cnn`
torch env.

## Inference engine + catalog evaluation

`ADCNN.inference.catalog.build_detection_catalog[_multigpu]` is the end-to-end engine:
images → segmentation → candidates → matched-filter measurement → 96² cutout → cutout CNN →
one CSV row per detection with measured trail geometry (`x, y, beta, length`), brightness
(`flux`), the raw NN peak (`nn_pmax`), and the stage-2 CNN `score`, plus routing keys
(`visit / detector / band` joined from `panels.csv`) for downstream linking.

On 4 × A100 with parallel CPU prep + batched tile inference, the engine reaches
**~1.0–1.4 s / panel** (images → catalog), i.e. a full ~189-detector LSST visit in 3–4 min.

Evaluation is then pure catalog analysis:
`ADCNN.evaluation.catalog_match.evaluate_catalog(measured, truth)` does trail-overlap
matching (any-pixel-overlap criterion, all-trails denominator, fixed `tol_px`)
→ TP / FP / FN + the flagged truth catalog for the completeness and histogram plots in
`Evaluation/Evaluation.ipynb` and `Evaluation/Evaluation_Real.ipynb`.

## Package layout

```
ADCNN/
├── core/                 model architectures
│   ├── model.py           UNet-ResSE backbone
│   ├── detector.py        segmentation model (UNet + orientation + Hough aggregator)
│   └── losses.py          masked Asymmetric Focal Tversky + orientation MSE
├── data/                 dataset loading + preprocessing
│   ├── dataset.py         3-channel diffim random-crop Dataset + concat wrapper
│   ├── preprocessing.py   MAD-sigma, local-std, orientation-map builders
│   └── dataset_creation/  Butler-side builders (simulated injections + real-asteroid set)
├── training/             trainers
│   ├── train.py           stage-1 segmentation trainer
│   ├── cnn_postproc.py    stage-2 cutout-CNN trainer + combined-FPP threshold calibration
│   └── ema.py             EMA over weights
├── inference/            inference primitives + the end-to-end engine
│   ├── predict.py         sliding-window segmentation
│   ├── candidates.py      connected-component candidate extractor
│   ├── matched_filter.py  per-candidate matched-filter geometry + SNR
│   ├── features.py        candidate features + injection-overlap labelling
│   ├── cnn_postproc.py    focal cutout CNN: build / load / score
│   ├── catalog.py         end-to-end h5 -> detection catalog (multi-GPU)
│   ├── export.py          TorchScript exporter
│   ├── stream_real_butler.py     producer half of the real-data stream
│   ├── stream_real_inference.py  consumer half of the real-data stream
│   └── build_real_eval_catalog.py join stream output to truth -> per-sighting eval CSV
├── evaluation/           catalog-based evaluation + plots
│   ├── catalog_match.py   trail-overlap matching + dedup
│   ├── geometry.py        mask + component primitives
│   ├── plots.py           notebook visualisations
│   └── architecture.py    paper architecture figures
└── pipelines/            CLI entry points + SLURM submission wrappers
    ├── make_sim_data.py
    ├── make_real_data.py
    ├── train_end_to_end.py
    ├── make_eval_catalogs.py
    ├── slurm/             SLURM scripts (data build, train, eval)
    └── heliolinc/         downstream linking pipeline (HelioLinC bridge + NEO discovery DAG)

models/                  deployed weights + sidecar
Evaluation/              Evaluation.ipynb (simulated) + Evaluation_Real.ipynb (real)
DATA/                    inputs (Butler manifests + real-asteroid truth + forced-photometry CSVs)
DATA_DIFFIM/             built diffim h5/csv datasets (not tracked)
```

## Reproducing v1.0

```bash
# Stage A — build the simulated dataset family + the real-data test set
python -m ADCNN.pipelines.make_sim_data --save-path DATA_DIFFIM/
python -m ADCNN.pipelines.make_real_data --save-path DATA_DIFFIM/test_real/

# Stage B — train the deployed two-stage detector
python -m ADCNN.pipelines.train_end_to_end --run-name seg \
    --data-sources DATA_DIFFIM/train.h5:DATA_DIFFIM/train.csv \
    --val-h5  DATA_DIFFIM/val.h5  --val-csv  DATA_DIFFIM/val.csv \
    --cnn-train-h5 DATA_DIFFIM/cnn_train.h5 --cnn-train-csv DATA_DIFFIM/cnn_train.csv \
    --cnn-val-h5  DATA_DIFFIM/cnn_val.h5   --cnn-val-csv  DATA_DIFFIM/cnn_val.csv

# Stage C — evaluate on the simulated + real test sets
python -m ADCNN.pipelines.make_eval_catalogs --sets test test_real

# Stage D — render the notebooks
jupyter nbconvert --to notebook --execute --inplace Evaluation/*.ipynb
```

SLURM wrappers for the same pipeline live under `ADCNN/pipelines/slurm/`.

## Branches

- `diffim` — this production pipeline (canonical).
- `heliolinc-discovery` — `diffim` plus the downstream linking pipeline under
  `ADCNN/pipelines/heliolinc/`. `ADCNN/` and `Evaluation/` are byte-identical to `diffim`.
