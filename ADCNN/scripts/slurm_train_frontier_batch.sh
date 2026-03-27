#!/bin/bash
#SBATCH --job-name=ADCNN-frontier
#SBATCH --account kipac:kipac
#SBATCH --partition ada
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/sdf/home/m/mrakovci/%x_%A_%a.out

set -euo pipefail

source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

echo "CUDA_VISIBLE_DEVICES (pre): ${CUDA_VISIBLE_DEVICES-<unset>}"

REPO_DIR="${REPO_DIR:-/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN}"
DATA_DIR="${DATA_DIR:-/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA}"
SOFT_MASK_CACHE_ROOT="${SOFT_MASK_CACHE_ROOT:-${DATA_DIR}/soft_mask_cache}"
SOFT_MASK_CACHE_SIZE="${SOFT_MASK_CACHE_SIZE:-64}"
SOFT_MASK_SIGMA_PIX="${SOFT_MASK_SIGMA_PIX:-2.0}"
EXPERIMENT_SET="${EXPERIMENT_SET:-frontier_batch_v1}"
EPOCHS="${EPOCHS:-18}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAIN_LR="${MAIN_LR:-1.0e-4}"
WARMUP_LR="${WARMUP_LR:-1.0e-4}"
RESCUE_VAL_MAX_IMAGES="${RESCUE_VAL_MAX_IMAGES:-8}"
RESCUE_VAL_EVERY_EARLY="${RESCUE_VAL_EVERY_EARLY:-1}"
RESCUE_VAL_EARLY_EPOCHS="${RESCUE_VAL_EARLY_EPOCHS:-8}"
RESCUE_VAL_EVERY="${RESCUE_VAL_EVERY:-3}"
RESCUE_BUDGET_GRID="${RESCUE_BUDGET_GRID:-50,200,1000,15000}"
SOFT_TARGET_GAIN="${SOFT_TARGET_GAIN:-4.0}"
ASL_GAMMA_NEG="${ASL_GAMMA_NEG:-4.0}"
ASL_GAMMA_POS="${ASL_GAMMA_POS:-0.0}"
ASL_CLIP="${ASL_CLIP:-0.05}"
LAMBDA_MAX_WEAK="${LAMBDA_MAX_WEAK:-0.05}"
RAMP_START="${RAMP_START:-10}"
RAMP_END="${RAMP_END:-18}"

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
      --loss-mode bce
      --soft-target-gain 1.0
    )
    ;;
  2)
    EXP_NAME="soft_bce_gain"
    EXP_ARGS=(
      --target-mask-mode soft
      --loss-mode bce
      --soft-target-gain "${SOFT_TARGET_GAIN}"
    )
    ;;
  3)
    EXP_NAME="soft_bce_dice_gain"
    EXP_ARGS=(
      --target-mask-mode soft
      --loss-mode bce_dice
      --soft-target-gain "${SOFT_TARGET_GAIN}"
      --lam-max "${LAMBDA_MAX_WEAK}"
      --ramp-start "${RAMP_START}"
      --ramp-end "${RAMP_END}"
    )
    ;;
  4)
    EXP_NAME="soft_asl_gain"
    EXP_ARGS=(
      --target-mask-mode soft
      --loss-mode asl
      --soft-target-gain "${SOFT_TARGET_GAIN}"
      --asl-gamma-neg "${ASL_GAMMA_NEG}"
      --asl-gamma-pos "${ASL_GAMMA_POS}"
      --asl-clip "${ASL_CLIP}"
    )
    ;;
  5)
    EXP_NAME="hard_asl"
    EXP_ARGS=(
      --target-mask-mode hard
      --loss-mode asl
      --asl-gamma-neg "${ASL_GAMMA_NEG}"
      --asl-gamma-pos "${ASL_GAMMA_POS}"
      --asl-clip "${ASL_CLIP}"
    )
    ;;
  *)
    echo "[launcher] Unknown SLURM_ARRAY_TASK_ID=${TASK_ID} for set ${EXPERIMENT_SET}"
    exit 2
    ;;
esac

CHECKPOINT_DIR="../checkpoints/${EXPERIMENT_SET}"
SUMMARY_PATH="${CHECKPOINT_DIR}/${EXP_NAME}_rescue_frontier.jsonl"
SOFT_MASK_CACHE_DIR="${SOFT_MASK_CACHE_ROOT}/${EXPERIMENT_SET}/${EXP_NAME}"

mkdir -p "${CHECKPOINT_DIR}"

echo "=== Experiment ==="
echo "Set: ${EXPERIMENT_SET}"
echo "Task: ${TASK_ID}"
echo "Name: ${EXP_NAME}"
echo "Data: ${DATA_DIR}"
echo "Soft mask cache: ${SOFT_MASK_CACHE_DIR}"
echo "Soft mask cache size: ${SOFT_MASK_CACHE_SIZE}"
echo "Soft mask sigma: ${SOFT_MASK_SIGMA_PIX}"
echo "Soft target gain: ${SOFT_TARGET_GAIN}"
echo "Rescue budget grid: ${RESCUE_BUDGET_GRID}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Main LR: ${MAIN_LR}"
echo "Warmup LR: ${WARMUP_LR}"
echo "Summary path: ${SUMMARY_PATH}"

rm -f "${SUMMARY_PATH}"

if [[ "${EXP_NAME}" == soft_* ]]; then
  rm -rf "${SOFT_MASK_CACHE_DIR}"
  mkdir -p "${SOFT_MASK_CACHE_DIR}"
  echo "[launcher] Precomputing soft masks into cache..."
  python scripts/precompute_soft_masks.py \
    --train-h5 "${DATA_DIR}/train.h5" \
    --train-csv "${DATA_DIR}/train.csv" \
    --cache-dir "${SOFT_MASK_CACHE_DIR}" \
    --sigma-pix "${SOFT_MASK_SIGMA_PIX}"
fi

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
  --rescue-budget-grid "${RESCUE_BUDGET_GRID}"
  --rescue-budget-primary 50
  --rescue-budget-secondary 15000
  --rescue-val-summary-path "${SUMMARY_PATH}"
)

if [[ "${EXP_NAME}" == soft_* ]]; then
  RUN_CMD+=(
    --soft-mask-cache-dir "${SOFT_MASK_CACHE_DIR}"
    --soft-mask-cache-size "${SOFT_MASK_CACHE_SIZE}"
    --soft-mask-sigma-pix "${SOFT_MASK_SIGMA_PIX}"
  )
fi

RUN_CMD+=("${EXP_ARGS[@]}")

printf '[launcher] Command:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'

PYTHONUNBUFFERED=1 "${RUN_CMD[@]}"
