#!/bin/bash
#SBATCH --job-name=ADCNN_base
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x.out

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs

# --- Conda ---
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# --- Project paths ---
PROJ="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
EXPDIR="${PROJ}/experiments"
cd "${EXPDIR}"

# --- Use SLURM-provided GPU allocation (robust parsing) ---
GPU_LIST_RAW="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
GPU_LIST="$(echo "${GPU_LIST_RAW}" | tr ',' '\n' | sed -E 's/[^0-9]//g' | awk 'NF' | paste -sd, -)"

if [[ -z "${GPU_LIST}" ]]; then
  echo "[launcher] ERROR: No GPUs visible from SLURM vars."
  echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
NGPU="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"

echo "[launcher] Using GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (count=${NGPU})"
echo "[launcher] Host: $(hostname)"
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'n', torch.cuda.device_count())"

# --- DDP/NCCL env ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- Run ---
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" baseline.py
