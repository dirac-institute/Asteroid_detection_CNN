#!/bin/bash
#SBATCH -J adcnn_dbg
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /sdf/home/m/mrakovci/logs/adcnn_debug.out

set -euo pipefail

# -----------------------------
# User-configurable paths
# -----------------------------
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
ENV="/sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn"

TRAIN_H5="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/train.h5"
TRAIN_CSV="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/train.csv"

# If train.csv is relative, make it absolute here to avoid cwd surprises.
# Example:
# TRAIN_CSV="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA/train.csv"

BATCH_SIZE=64
NUM_WORKERS=2
TILE=128
VAL_FRAC=0.10
REAL_KEY="real_labels"

# Debugger knobs
AUC_BINS=256
VAL_BATCHES=24

# Overfit knobs
OVERFIT_STEPS=1500
OVERFIT_LR="3e-4"
OVERFIT_POSW="8.0"
LOG_EVERY=25

# Subset knobs
SUBSET_TRAIN=2048
SUBSET_VAL=512
SUBSET_STEPS=3000

# -----------------------------
# Environment
# -----------------------------
echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  ${SLURMD_NODENAME}"
echo "GPU:   ${CUDA_VISIBLE_DEVICES:-<unset>}"

# Optional: reduce nondeterminism warning if you use --deterministic
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${REPO}"

# Activate conda
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU avail:', torch.cuda.is_available())"

COMMON_ARGS=(
  --train-h5 "${TRAIN_H5}"
  --train-csv "${TRAIN_CSV}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --tile "${TILE}"
  --val-frac "${VAL_FRAC}"
  --real-labels-key "${REAL_KEY}"
  --auc-bins "${AUC_BINS}"
  --val-batches "${VAL_BATCHES}"
  --device single
  --deterministic
)

run_test () {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "TEST: ${name}"
  echo "CMD:  python3 debugging.py ${COMMON_ARGS[*]} $*"
  echo "============================================================"
  python3 debugging.py "${COMMON_ARGS[@]}" "$@"
}

cd "ADCNN"
export PYTHONPATH=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN:$PYTHONPATH

# -----------------------------
# 4 tests
# -----------------------------
# 1) Smoke: dataset stats + random-model val probe
run_test "1_smoke" --mode smoke

# 2) Overfit1: single-batch overfit sanity check
run_test "2_overfit1" \
  --mode overfit1 \
  --steps "${OVERFIT_STEPS}" \
  --overfit-lr "${OVERFIT_LR}" \
  --overfit-posw "${OVERFIT_POSW}" \
  --log-every "${LOG_EVERY}"

# 3) Subset: short train loop (only if Trainer.train_full_probe exists)
run_test "3_subset" \
  --mode subset \
  --steps "${SUBSET_STEPS}" \
  --subset-train "${SUBSET_TRAIN}" \
  --subset-val "${SUBSET_VAL}"

# 4) DDP check: val-only probe under DDP init (still 1 GPU; useful to ensure init path works)
# This will run with world_size=1 but exercises the ddp init path if your init_distributed tolerates it.
run_test "4_ddpcheck" --mode ddpcheck --device ddp

echo
echo "All debug tests completed."