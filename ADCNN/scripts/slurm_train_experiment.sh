#!/bin/bash
#SBATCH --job-name=ADCNN-exp
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/sdf/home/m/mrakovci/logs/%x_%A_%a.out

set -euo pipefail

mkdir -p /sdf/home/m/mrakovci/logs

# === Activate Conda environment ===
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

echo "CUDA_VISIBLE_DEVICES (pre): ${CUDA_VISIBLE_DEVICES-<unset>}"

REPO_DIR="${REPO_DIR:-/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN}"
DATA_DIR="${DATA_DIR:-/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA}"
SOFT_MASK_CACHE_DIR="${SOFT_MASK_CACHE_DIR:-${DATA_DIR}/soft_mask_cache}"
EXPERIMENT_SET="${EXPERIMENT_SET:-loss_ablation_v1}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAIN_LR="${MAIN_LR:-1.0e-4}"
WARMUP_LR="${WARMUP_LR:-1.0e-4}"
RESCUE_VAL_MAX_IMAGES="${RESCUE_VAL_MAX_IMAGES:-8}"
RESCUE_VAL_EVERY_EARLY="${RESCUE_VAL_EVERY_EARLY:-1}"
RESCUE_VAL_EARLY_EPOCHS="${RESCUE_VAL_EARLY_EPOCHS:-8}"
RESCUE_VAL_EVERY="${RESCUE_VAL_EVERY:-3}"

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

EXP_NAME=""
declare -a EXP_ARGS

case "${TASK_ID}" in
  0)
    EXP_NAME="hard_bce_baseline"
    EXP_ARGS=(
      --target-mask-mode hard
      --loss-mode bce
    )
    ;;
  1)
    EXP_NAME="soft_bce_baseline"
    EXP_ARGS=(
      --target-mask-mode soft
      --soft-mask-cache-dir "${SOFT_MASK_CACHE_DIR}"
      --loss-mode bce
    )
    ;;
  2)
    EXP_NAME="soft_bce_dice_weak"
    EXP_ARGS=(
      --target-mask-mode soft
      --soft-mask-cache-dir "${SOFT_MASK_CACHE_DIR}"
      --loss-mode bce_dice
      --lam-max 0.05
      --ramp-start 10
      --ramp-end 18
    )
    ;;
  3)
    EXP_NAME="soft_bce_ft_weak"
    EXP_ARGS=(
      --target-mask-mode soft
      --soft-mask-cache-dir "${SOFT_MASK_CACHE_DIR}"
      --loss-mode bce_ft
      --lam-max 0.05
      --ramp-start 10
      --ramp-end 18
    )
    ;;
  *)
    echo "[launcher] Unknown SLURM_ARRAY_TASK_ID=${TASK_ID} for set ${EXPERIMENT_SET}"
    exit 2
    ;;
esac

echo "=== Experiment ==="
echo "Set: ${EXPERIMENT_SET}"
echo "Task: ${TASK_ID}"
echo "Name: ${EXP_NAME}"
echo "Data: ${DATA_DIR}"
echo "Soft mask cache: ${SOFT_MASK_CACHE_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Main LR: ${MAIN_LR}"
echo "Warmup LR: ${WARMUP_LR}"

RUN_CMD=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${NGPU}"
  -m main
  --train-h5 "${DATA_DIR}/train.h5"
  --train-csv "${DATA_DIR}/train.csv"
  --epochs "${EPOCHS}"
  --batch "${BATCH_SIZE}"
  --main-lr "${MAIN_LR}"
  --warmup-lr "${WARMUP_LR}"
  --rescue-val-max-images "${RESCUE_VAL_MAX_IMAGES}"
  --rescue-val-every-early "${RESCUE_VAL_EVERY_EARLY}"
  --rescue-val-early-epochs "${RESCUE_VAL_EARLY_EPOCHS}"
  --rescue-val-every "${RESCUE_VAL_EVERY}"
)
RUN_CMD+=("${EXP_ARGS[@]}")

printf '[launcher] Command:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'

PYTHONUNBUFFERED=1 "${RUN_CMD[@]}"
