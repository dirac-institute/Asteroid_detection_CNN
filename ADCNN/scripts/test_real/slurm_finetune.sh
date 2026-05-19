#!/bin/bash
# Precision-tilted hard-neg FINE-TUNE of v7 (resumes shipped weights via
# --init-from). Reversible: new run dir v7_ft_hn; pilot_v7 untouched.
# Usage: sbatch ADCNN/scripts/test_real/slurm_finetune.sh
#
#SBATCH --job-name=adc-v7-ft
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_v7_ft_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA="${REPO_DIR}/DATA_DIFFIM"
RUN="v7_ft_hn"
RUN_DIR="${REPO_DIR}/experiments/diffim_runs/${RUN}"
INIT="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts/best.pt"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${RUN_DIR}/ckpts"
# Low LR, few epochs, FP-penalising AFTL (alpha>beta), 4x neg anchors, less
# stack-missed-positive emphasis. v7-half arch must match the checkpoint.
PYTHONUNBUFFERED=1 srun python -m ADCNN.training.diffim_train \
  --run-name "${RUN}" --init-from "${INIT}" \
  --data-h5 "${DATA}/train.h5" --data-csv "${DATA}/train.csv" \
  --n-train-panels 740 --n-val-panels 50 --tile 128 \
  --epochs 12 --batch-size 24 \
  --n-pos-anchors-per-epoch 3000 --n-neg-anchors-per-epoch 4000 \
  --stk-balance 0.3 --anchor-jitter 48 \
  --aftl-alpha 0.6 --aftl-beta 0.4 --aftl-gamma 1.3 --aftl-bce-anchor 0.2 \
  --lambda-orient 0.5 --kernel-lens 11 21 41 --n-angles 12 \
  --widths 24 48 96 192 384 --num-workers 6 --lr 5e-5 \
  --ema-decay 0.999 --ema-exclude agg_alpha --orient-cache-size 50
# Export the LAST (fully fine-tuned) ckpt — best.pt is val_auc-selected and
# peaks at ep01 before the precision-tilt engages.
PYTHONUNBUFFERED=1 python -m ADCNN.inference.diffim_export \
  --ckpt "${RUN_DIR}/ckpts/last.pt" \
  --out  "${RUN_DIR}/ckpts/v7_ft_last_scripted.pt" --no-optimize
echo "FT DONE $(date -Is)"
