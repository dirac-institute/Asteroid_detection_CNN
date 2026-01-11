#!/bin/bash
#SBATCH --job-name=AC_I1
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=70G
#SBATCH --time=5-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x.out

set -euo pipefail
mkdir -p /sdf/home/m/mrakovci/logs

source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

PROJ="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
EXPDIR="${PROJ}/experiments"
DATA_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
cd "${EXPDIR}"

echo "[launcher] Host: $(hostname)"
echo "[launcher] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "[launcher] SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "[launcher] CUDA_VISIBLE_DEVICES(before)=${CUDA_VISIBLE_DEVICES:-<unset>}"

# IMPORTANT:
# On SDF, SLURM typically sets CUDA_VISIBLE_DEVICES correctly for the job (local 0..N-1).
# If it is not set, force a sane default for a 2-GPU job.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1"
fi

echo "[launcher] CUDA_VISIBLE_DEVICES(after)=${CUDA_VISIBLE_DEVICES}"

# Sanity
python - <<'PY'
import torch, os
print("torch", torch.__version__)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda_avail", torch.cuda.is_available(), "n", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

GPU_LOG="/sdf/home/m/mrakovci/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu_dmon.log"

# Start GPU utilization logging in background
nvidia-smi dmon -s pucvmet -d 2 > "$GPU_LOG" &
DMON_PID=$!
echo "[launcher] GPU monitor PID=$DMON_PID log=$GPU_LOG"

# Ensure the monitor is killed on exit (success/fail)
cleanup() { kill "$DMON_PID" 2>/dev/null || true; }
trap cleanup EXIT


# Run with 4 processes (one per GPU visible to the job)
srun --ntasks=1 --gpus=4 --cpus-per-task=${SLURM_CPUS_PER_TASK:-2} \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  idea1.py \
  --repo-root "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN" \
  --train-h5 "${DATA_DIR}/train.h5" \
  --train-csv "${DATA_DIR}/train.csv" \
  --test-h5  "${DATA_DIR}/test.h5" \
  --tile 128 \
  --batch-size 256 \
  --num-workers "${SLURM_CPUS_PER_TASK:-2}" \
  --seed 1337 \
  --max-epochs 50 \
  --val-every 25
