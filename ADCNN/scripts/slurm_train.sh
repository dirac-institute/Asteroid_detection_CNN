#!/bin/bash
#SBATCH --job-name=ADCNN
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x_train.out

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs

# === Activate Conda environment ===
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

echo "CUDA_VISIBLE_DEVICES (pre): ${CUDA_VISIBLE_DEVICES-<unset>}"

REPO_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN"
DATA_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

python -c "import ADCNN; print('ADCNN import OK from', ADCNN.__file__)"
JSON="$(python ADCNN/utils/gpu_healthcheck.py || true)"

HEALTHY="$(python -c 'import sys,json; j=json.loads(sys.stdin.read()); print(",".join(map(str, j.get("healthy", []))))' <<< "$JSON")"
if [[ -z "${HEALTHY}" ]]; then
  echo "[launcher] No healthy GPUs detected on this node."
  echo "[launcher] Healthcheck JSON: $JSON"
  exit 1
fi

IFS=',' read -r -a _arr <<< "$HEALTHY"
NGPU="${#_arr[@]}"
export CUDA_VISIBLE_DEVICES="${HEALTHY}"

echo "=== Environment info ==="
echo "Hostname: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
which python
python -c "import torch; print('Torch:', torch.__version__, '| GPUs visible:', torch.cuda.device_count())"

export OMP_NUM_THREADS=8
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

cd "ADCNN"

PYTHONUNBUFFERED=1 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPU}" \
  -m main \
  --train-h5 "${DATA_DIR}/train.h5" \
  --epochs 50 \
  --batch 256