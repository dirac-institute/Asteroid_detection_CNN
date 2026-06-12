#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH -J v2_smoke
#SBATCH -o %x_%j.log
set -eo pipefail
# ADCNN v2 SMOKE TEST (ADCNN_V2_SPRINT.md Phase 2): 1 epoch, init from reconstructed v1 weights,
# verifies the fine-tune loop runs end-to-end on the run_ft DM-53195 dataset. NOT a science run.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
FT=$REPO/ADCNN/pipelines/heliolinc/run_ft
python -m ADCNN.training.train --run-name v2_smoke \
  --data-sources "$FT/train.h5:$FT/train.csv" \
  --data-h5 "$FT/val.h5" --data-csv "$FT/val.csv" \
  --init-from "$REPO/models/seg_v1_trainable_init.pt" \
  --epochs 1 --lr 5e-5 --intensity-aug
echo "V2_SMOKE_DONE"
