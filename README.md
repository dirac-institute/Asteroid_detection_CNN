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

## Active pipeline (the default)

There is **one current ADCNN pipeline**, selected through a single config
(`ADCNN/config.py` → `models/<name>/pipeline.json`). The default is `current` — the
domain-adapted detector (frozen `adcnn-v2_D-rc1`). A pipeline bundles, as one unit, the
stage-1/stage-2 model files **and** the model-specific MF_LEN trail-length de-bias, so they
always travel together (mixing a model with another model's de-bias silently corrupts
`len_db`). Select a different pipeline with `ADCNN_PIPELINE` (a name or a path):

| pipeline | what | de-bias |
|---|---|---|
| `current` *(default)* | `models/current/` → the promoted detector (`adcnn-v2_D-rc1`) | 7.67 / 0.9425 |
| `legacy_v1` | `models/legacy_v1/` → the prior v1.0 baseline (provenance / regression) | 33.4 / 0.887 |

Headline product result (current pipeline): faint-fast same-night 2-visit **alert
completeness 3.64% → 10.33%** (+184%) at held purity, on the DM-53195 blind fields —
reproduce with `python -m ADCNN.pipelines.run_experiment --stage report`. The frozen models
live under `models/v2_D/` (md5s in `v2_D_release.json`); `models/current/` points into them.

## Single entry point

Everything runs through the repo-root `./adcnn` executable (equivalently
`python -m ADCNN`, run from the repo root):

```bash
./adcnn night --collection <DRP-collection> --night <mjd> --tracts <t> [--dry-run]
                                                                # score one real night end-to-end
./adcnn experiment --stage report                               # reproduce the headline (CPU)
./adcnn experiment --stages all --dry-run                       # the full ordered dev plan
./adcnn train-and-validate --stages calibrate-mflen,threshold-select,freeze \
    --out models/current_candidate                              # freeze a release (CPU)
```

Three commands, mapping onto the two top-level pipelines plus the dev driver:

| command | pipeline | what it does |
|---|---|---|
| `night` | `ADCNN.pipelines.run_night` | **APPLY** a frozen release to one night: manifest → GPU detect → known catalog → mask → static-veto catalog → 2-visit linking (frozen alert op, FLAG-never-drop vetoes) → pixel vet → MPC crossmatch |
| `train-and-validate` | `ADCNN.pipelines.train_and_validate` | **DECIDE + FREEZE**: train, re-derive calibrations, regenerate validation curves, select + confirm the operating point, freeze a self-contained release dir |
| `experiment` | `ADCNN.pipelines.run_experiment` | detector-development driver (`data`, `train-stage1`, `train-stage2`, `calibrate-mflen`, `detect`, `alert-eval`, `report`) |

CPU stages run in-process; GPU/Butler stages print the exact `sbatch` command (submit with
`--submit` where supported). All runtime output lands under repo-root `outputs/`
(override with `ADCNN_OUTPUTS`); see `outputs/README.md`. Details:
**REPRODUCE.md**, **TRAINING_PROTOCOL.md**, **EVALUATION_PROTOCOL.md**.

### Underlying stage modules (`python -m ADCNN.pipelines.<name>`)

| command | purpose |
|---|---|
| `run_experiment`     | **the single canonical driver** wrapping all stages below |
| `make_sim_data`      | build the simulated injected-trail train / val / test datasets from the Butler |
| `make_real_data`     | build the real-asteroid test diffim dataset from the Butler |
| `train_end_to_end`   | train the segmentation model, then the focal cutout CNN, and persist both with a sidecar JSON |
| `make_eval_catalogs` | score the test sets with the active pipeline, emit detection catalogs and metrics |

Detector defaults (`--seg-model` / `--cnn` and the MF_LEN de-bias) resolve from the active
pipeline; no per-run env combos are needed. Stage 1 reaches **96.0 % object-wise recall on the
simulated 5σ test set** (v1.0 baseline); the combined 5σ-stack + ADCNN detector is calibrated
to **100 FP/panel** (`ADCNN.training.cnn_postproc.calibrate_combined_threshold`).

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
├── config.py            active-pipeline resolver (models/<name>/pipeline.json; ADCNN_PIPELINE)
└── pipelines/           CLI entry points + SLURM submission wrappers
    ├── run_experiment.py the single canonical workflow driver (all stages)
    ├── leakage_guard.py  fail-loud (visit,detector) blind/test leakage check
    ├── make_sim_data.py
    ├── make_real_data.py
    ├── train_end_to_end.py
    ├── make_eval_catalogs.py
    ├── slurm/             SLURM scripts (data build, train, eval)
    └── heliolinc/         downstream linking pipeline (HelioLinC bridge + NEO discovery DAG)

models/                  current/ + legacy_v1/ (pipeline.json + pointers) over the frozen v2_D/ release
Evaluation/              Evaluation.ipynb (current) + Evaluation_legacy_v1.ipynb + Evaluation_Real.ipynb
DATA/                    inputs (Butler manifests + real-asteroid truth + forced-photometry CSVs)
DATA_DIFFIM/             built diffim h5/csv datasets (not tracked)
```

## Reproducing the result

For the **current** pipeline (the headline), see **REPRODUCE.md** (one-command verdict +
the full GPU chain) and **TRAINING_PROTOCOL.md** / **EVALUATION_PROTOCOL.md**. The legacy
v1.0 recipe below is kept for provenance.

### Legacy v1.0

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
python -m ADCNN.evaluation.make_notebook_inputs --sets test test_real

# Stage D — render the notebooks
jupyter nbconvert --to notebook --execute --inplace Evaluation/*.ipynb
```

SLURM wrappers for the same pipeline live under `ADCNN/pipelines/slurm/`.

## Branches

- `diffim` — this production pipeline (canonical).
- `heliolinc-discovery` — `diffim` plus the downstream linking pipeline under
  `ADCNN/pipelines/heliolinc/`. `ADCNN/` and `Evaluation/` are byte-identical to `diffim`.
