#!/bin/bash
#SBATCH --job-name=AC_I2
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=5-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x.out

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs

# === Conda ===
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# ---[ Project paths ]---
PROJECT_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN/experiments"
DATA_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
cd "$PROJECT_DIR"

# ---[ GPU healthcheck + mask ]---
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
NGPU=$(python -c "import os; print(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))")
echo "[launcher] Using SLURM_JOB_GPUS=${SLURM_JOB_GPUS}  (count=${NGPU})"

# ---[ NCCL/DDP env ]---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

# ---[ Launch ]---
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  idea2.py \
  --repo-root "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN" \
  --train-h5 "${DATA_DIR}/train.h5" \
  --train-csv "${DATA_DIR}/train.csv" \
  --test-h5  "${DATA_DIR}/test.h5" \
  --tile 128 \
  --batch-size 256 \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 1337 \
  --max-epochs 50 \
  --val-every 3 \
  --hard-pos-boost 4.0 \
  --ramp-kind linear \
  --ramp-start-epoch 10 \
  --ramp-end-epoch 50 \
  --long-batches 0
