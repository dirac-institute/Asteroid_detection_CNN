#!/bin/bash
#SBATCH --job-name=ADCNN         # Job name
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1                # One node
#SBATCH --gres=gpu:4             # 4 GPUs on the node
#SBATCH --ntasks-per-node=4      # One task per GPU (DDP)
#SBATCH --cpus-per-task=8        # CPU threads per task (tune to your cluster)
#SBATCH --mem=60G                # Total RAM for the node (adjust)
#SBATCH --time=5-00:00:00          # Max runtime
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x_%j.out  # STDOUT

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs
# === Activate Conda environment ===
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# ---[ Project paths]---
PROJECT_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN"
DATA_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
cd "$PROJECT_DIR"

#Probe GPUs on this node
# 1) Probe GPUs on this node (returns JSON: {"healthy":[...],"bad":{...}})
JSON="$(python utils/gpu_healthcheck.py || true)"

# 2) Extract healthy indices from JSON using a tiny Python one-liner
HEALTHY="$(python -c 'import sys,json; j=json.loads(sys.stdin.read()); print(",".join(map(str, j.get("healthy", []))))' <<< "$JSON")"

if [[ -z "${HEALTHY}" ]]; then
  echo "[launcher] No healthy GPUs detected on this node."
  echo "[launcher] Healthcheck JSON: $JSON"
  exit 1
fi

# Count how many
IFS=',' read -r -a _arr <<< "$HEALTHY"
NGPU="${#_arr[@]}"

echo "[launcher] Healthy GPUs: ${HEALTHY}  (count=${NGPU})"

# 3) Mask to only healthy GPUs
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

# ---[ Torchrun launch ]---
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  main.py \
  --train_h5 "${DATA_DIR}/train_chunked.h5" \
  --epochs 50 \
  --batch 256