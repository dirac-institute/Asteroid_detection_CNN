#!/bin/bash
# EXPERIMENTAL realneg fine-tune: continue from the PROMOTED precision-tilt
# v7 (v7_ft_hn/last.pt) on the real-empty-background dataset. Bounded ~1
# GPU-h (design RESULTS.md §5-6). Reuses tracked ADCNN.training.diffim_train
# (no tracked edits). New run dir; nothing promoted automatically.
# Usage: sbatch experiments/explore_realneg_train/slurm_finetune_realneg.sh
#
#SBATCH --job-name=adc-realneg-ft
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_realneg_ft_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
RUN="v7_ft_realneg"
# Heavy outputs on the 80G scratch area (repo quota full). Trainer writes to
# ${OUTROOT}/${RUN}/ via --out-root.
BASE="/sdf/scratch/users/m/mrakovci/realneg"
OUTROOT="${BASE}/runs"
RUN_DIR="${OUTROOT}/${RUN}"
DS="${BASE}/dataset"
# Continue from the PROMOTED precision-tilt ft trainable ckpt (model+EMA).
INIT="${REPO_DIR}/experiments/diffim_runs/v7_ft_hn/ckpts/last.pt"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
test -f "${INIT}" || { echo "MISSING init ckpt ${INIT}"; exit 2; }
test -f "${DS}/train.h5" || { echo "MISSING ${DS}/train.h5 (run build first)"; exit 2; }
mkdir -p "${RUN_DIR}/ckpts"
# Same precision-tilt AFTL as the promoted run; 2x negative anchors so the
# loss is dominated by "real residual -> ~0" on the real-empty-bg panels
# (~35% of the set are zero-injection empties; the rest are real subtracted
# diffims that also carry residuals). Arch flags MUST match the v7 ckpt.
PYTHONUNBUFFERED=1 srun python -m ADCNN.training.diffim_train \
  --run-name "${RUN}" --out-root "${OUTROOT}" --init-from "${INIT}" \
  --data-h5 "${DS}/train.h5" --data-csv "${DS}/train.csv" \
  --n-train-panels 560 --n-val-panels 40 --tile 128 \
  --epochs 12 --batch-size 24 \
  --n-pos-anchors-per-epoch 3000 --n-neg-anchors-per-epoch 8000 \
  --stk-balance 0.3 --anchor-jitter 48 \
  --aftl-alpha 0.6 --aftl-beta 0.4 --aftl-gamma 1.3 --aftl-bce-anchor 0.2 \
  --lambda-orient 0.5 --kernel-lens 11 21 41 --n-angles 12 \
  --widths 24 48 96 192 384 --num-workers 6 --lr 5e-5 \
  --ema-decay 0.999 --ema-exclude agg_alpha --orient-cache-size 50
# Export the LAST ckpt (best.pt is val_auc-selected, peaks pre-tilt).
PYTHONUNBUFFERED=1 python -m ADCNN.inference.diffim_export \
  --ckpt "${RUN_DIR}/ckpts/last.pt" \
  --out  "${RUN_DIR}/ckpts/v7_ft_realneg_scripted.pt" --no-optimize
echo "REALNEG-FT DONE $(date -Is) -> ${RUN_DIR}/ckpts/v7_ft_realneg_scripted.pt"
