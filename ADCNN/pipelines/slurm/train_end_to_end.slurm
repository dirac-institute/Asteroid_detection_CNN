#!/bin/bash
#SBATCH --job-name=adcnn-train-e2e
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append
#
# ENTRY POINT (b) — train the two-stage detector end to end, deterministically.
#
#   stage 1: segmentation model (UNetResSEOrientHough, reg2 recipe) on TRAIN, model selection on
#            TRAIN2 (held out from stage-1 training), then TorchScript export.
#   stage 2: focal-loss cutout-CNN false-positive filter trained on ALL of the dedicated TRAIN2 set.
#   TEST is never touched here (held out for evaluation).
#
# Each dataset is used for exactly its purpose: TRAIN -> segmentation weights, TRAIN2 -> CNN filter
# (and stage-1 best-epoch selection only — the segmentation model never TRAINS on train2, so the
# stage-2 cutouts are leakage-free), TEST -> eval. The three are disjoint by construction (see
# make_datasets.slurm).
#
# DETERMINISM: the reg2 recipe pins seed=2026 (weight init, anchor sampling, D4 + intensity
# augmentation, data order); the stage-2 train_cnn is seeded too (weight init + minibatch order +
# FP subsampling). Reruns reproduce to GPU floating-point tolerance on the same GPU type — CUDA
# reductions are deliberately NOT bit-forced (torch deterministic mode would diverge from the
# validated reg2 numerics and may error on ops lacking deterministic kernels).
#
# Run AFTER make_datasets.slurm. Single A100 (training is single-GPU). Usage:
#   sbatch ADCNN/pipelines/slurm/train_end_to_end.slurm
#   RUN_NAME=seg_v2 DATA_ROOT=/path/DATA_DIFFIM sbatch ADCNN/pipelines/slurm/train_end_to_end.slurm
set -euo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA_ROOT="${DATA_ROOT:-$REPO/DATA_DIFFIM}"
RUN_NAME="${RUN_NAME:-seg_repro}"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic cuBLAS reductions where supported
export MPLBACKEND=Agg
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"

for f in train/train.h5 train/train.csv train2/train.h5 train2/train.csv; do
  [[ -s "$DATA_ROOT/$f" ]] || { echo "MISSING $DATA_ROOT/$f — run make_datasets.slurm first"; exit 1; }
done

python3 -u -m ADCNN.pipelines.train_end_to_end \
  --run-name "$RUN_NAME" \
  --data-sources "$DATA_ROOT/train/train.h5:$DATA_ROOT/train/train.csv" \
  --val-h5  "$DATA_ROOT/train2/train.h5" --val-csv  "$DATA_ROOT/train2/train.csv" \
  --cnn-train-h5 "$DATA_ROOT/train2/train.h5" --cnn-train-csv "$DATA_ROOT/train2/train.csv" \
  --out-root "$REPO/experiments/diffim_runs" --models-dir "$MODELS_DIR"

echo "=== TRAINING DONE ==="
echo "  segmentation: $MODELS_DIR/${RUN_NAME}_segmentation_scripted.pt"
echo "  cnn filter  : $MODELS_DIR/${RUN_NAME}_cnn_postproc.pt"
