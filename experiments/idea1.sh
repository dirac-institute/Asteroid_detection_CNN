#!/bin/bash
#SBATCH --job-name=AC_I1
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=5-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x_%j.out

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs

# === Conda ===
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# ---[ Project paths ]---
PROJECT_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN"
DATA_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
cd "$PROJECT_DIR"

# ---[ GPU healthcheck + mask ]---
JSON="$(python utils/gpu_healthcheck.py || true)"
HEALTHY="$(python -c 'import sys,json; j=json.loads(sys.stdin.read()); print(",".join(map(str, j.get("healthy", []))))' <<< "$JSON")"

if [[ -z "${HEALTHY}" ]]; then
  echo "[launcher] No healthy GPUs detected."
  echo "[launcher] Healthcheck JSON: $JSON"
  exit 1
fi

IFS=',' read -r -a _arr <<< "$HEALTHY"
NGPU="${#_arr[@]}"
export CUDA_VISIBLE_DEVICES="${HEALTHY}"

echo "[launcher] Healthy GPUs: ${CUDA_VISIBLE_DEVICES} (count=${NGPU})"

echo "=== Environment info ==="
echo "Hostname: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
which python3
python3 -c "import torch; print('Torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"

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
  idea1.py \
  --repo-root "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN" \
  --train-h5 "${DATA_DIR}/train.h5" \
  --train-csv "${DATA_DIR}/train.csv" \
  --test-h5  "${DATA_DIR}/test.h5" \
  --tile 128 \
  --batch-size 256 \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 1337 \
  --max-epochs 50 \
  --val-every 3
