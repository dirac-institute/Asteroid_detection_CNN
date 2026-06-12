#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J v2_variant
#SBATCH -o %x_%j.log
set -eo pipefail
# ADCNN v2 fine-tune variant (ADCNN_V2_SPRINT.md Phase 2). Env: RUN_NAME, LR, STKBAL.
# Variants: v2_B (lr 5e-5, stk 0.6) | v2_Blow (lr 2e-5, stk 0.6) | v2_D (lr 5e-5, stk 0.85).
# No freeze/encoder-lr flags exist in train.py -> charter variant C replaced by B-low-LR (documented).
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
FT=$REPO/ADCNN/pipelines/heliolinc/run_ft
python -m ADCNN.training.train --run-name "${RUN_NAME:?}" \
  --data-sources "$FT/train.h5:$FT/train.csv" \
  --data-h5 "$FT/val.h5" --data-csv "$FT/val.csv" \
  --init-from "$REPO/models/seg_v1_trainable_init.pt" \
  --n-train-panels 1429 --n-val-panels 292 \
  --epochs 10 --lr "${LR:?}" --stk-balance "${STKBAL:?}" --intensity-aug
echo "V2_VARIANT_DONE ${RUN_NAME}"
