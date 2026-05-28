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
#   stage 1: segmentation model (UNetResSEOrientHough, reg2 recipe) trained on TRAIN, model
#            selection on the held-out VAL set, then TorchScript export.
#   stage 2: focal-loss cutout-CNN FP filter trained on all of TRAIN2, operating threshold + AUC
#            measured on the held-out VAL2 set.
#   TEST is never touched here (held out for evaluation).
#
# Each set is used for exactly its purpose (all disjoint by construction — see make_datasets.slurm):
# TRAIN -> segmentation weights, VAL -> stage-1 selection, TRAIN2 -> CNN filter, VAL2 -> CNN
# threshold, TEST -> eval.
#
# DETERMINISM: the reg2 recipe pins seed=2026 (weight init, anchor sampling, D4 + intensity
# augmentation, data order); the stage-2 CNN trainer is seeded too (init + minibatch order + FP
# subsampling). Reruns reproduce to GPU floating-point tolerance on the same GPU type — CUDA
# reductions are deliberately NOT bit-forced (would diverge from the validated reg2 numerics).
#
# Run AFTER make_datasets.slurm. Single A100 (training is single-GPU). Usage:
#   sbatch ADCNN/pipelines/slurm/train_end_to_end.slurm
#   RUN_NAME=seg_v2 DATA_ROOT=/path sbatch ADCNN/pipelines/slurm/train_end_to_end.slurm
set -eo pipefail

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

for f in train.h5 train.csv val.h5 val.csv train2.h5 train2.csv val2.h5 val2.csv; do
  [[ -s "$DATA_ROOT/$f" ]] || { echo "MISSING $DATA_ROOT/$f — run make_datasets.slurm first"; exit 1; }
done

python3 -u -m ADCNN.pipelines.train_end_to_end \
  --run-name "$RUN_NAME" \
  --data-sources "$DATA_ROOT/train.h5:$DATA_ROOT/train.csv" \
  --val-h5  "$DATA_ROOT/val.h5"  --val-csv  "$DATA_ROOT/val.csv" \
  --cnn-train-h5 "$DATA_ROOT/train2.h5" --cnn-train-csv "$DATA_ROOT/train2.csv" \
  --val2-h5 "$DATA_ROOT/val2.h5" --val2-csv "$DATA_ROOT/val2.csv" \
  --out-root "$REPO/experiments/diffim_runs" --models-dir "$MODELS_DIR"

echo "=== TRAINING DONE ==="
echo "  segmentation: $MODELS_DIR/${RUN_NAME}_segmentation_scripted.pt"
echo "  cnn filter  : $MODELS_DIR/${RUN_NAME}_cnn_postproc.pt"
