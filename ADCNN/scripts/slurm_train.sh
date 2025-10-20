#!/bin/bash
#SBATCH --job-name=ADCNN         # Job name
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1                # One node
#SBATCH --gres=gpu:4             # 4 GPUs on the node
#SBATCH --ntasks-per-node=4      # One task per GPU (DDP)
#SBATCH --cpus-per-task=8        # CPU threads per task (tune to your cluster)
#SBATCH --mem=20G                # Total RAM for the node (adjust)
#SBATCH --time=5-00:00:00          # Max runtime
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x_%j.out  # STDOUT

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs
# === Activate Conda environment ===
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# 1) Probe GPUs on this node
JSON=$(python3 utils/gpu_healthcheck.py || true)
HEALTHY=$(python3 - <<'PY'
import os, json, sys
j=json.loads(os.environ['JSON'])
print(",".join(map(str, j.get("healthy", []))))
PY
)

if [[ -z "${HEALTHY}" ]]; then
  echo "[launcher] No healthy GPUs detected on this node. Failing fast."
  exit 1
fi

NGPU=$(( $(awk -F',' '{print NF}' <<< "${HEALTHY}") ))
echo "[launcher] Healthy GPUs: ${HEALTHY}  (count=${NGPU})"

# 2) Mask to only healthy GPUs for this process
export CUDA_VISIBLE_DEVICES="${HEALTHY}"

echo "=== Environment info ==="
echo "Hostname: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
which python3
python3 -c "import torch; print('Torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"

# ---[ Recommended env for NCCL/DDP ]---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0
# If your cluster uses specific IB/ethernet interface names, uncomment & set:
# export NCCL_SOCKET_IFNAME=bond0,eth0,eno1
# export NCCL_IB_HCA=mlx5

# ---[ Project paths (edit these) ]---
PROJECT_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN"
DATA_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"

# ---[ Torchrun launch ]---
# Single node; torchrun handles rank/world-size envs automatically here.
# Use $SLURM_GPUS_ON_NODE if your site sets it; fallback to 4.
NP=${SLURM_GPUS_ON_NODE:-4}

cd "$PROJECT_DIR"

# Example args — replace with your actual flags/configs
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NP}" \
  main.py \
  --train_h5 "${DATA_DIR}/train_chunked.h5" \
  --epochs 50 \
  --batch 256