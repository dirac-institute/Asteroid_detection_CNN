#!/bin/bash
#SBATCH --job-name=adc-seg_model-stream
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_seg_stream_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
echo "=== seg_model streaming train === $(date -Is)"
srun python3 -u -m ADCNN.training.diffim_train \
  --run-name pilot_seg_stream \
  --stream-buffer "$REPO/DATA_DIFFIM_stream_buffer" \
  --data-h5 "$REPO/DATA_DIFFIM_realistic/train.h5" \
  --data-csv "$REPO/DATA_DIFFIM_realistic/train.csv" \
  --tile 128 --epochs 120 --batch-size 24 --lr 0.0003 --wd 1e-5 \
  --n-pos-anchors-per-epoch 3000 --n-neg-anchors-per-epoch 900 \
  --anchor-jitter 48 --aftl-alpha 0.3 --aftl-beta 0.7 --aftl-gamma 1.3 \
  --aftl-bce-anchor 0.1 --lambda-orient 0.5 --kernel-lens 11 21 41 --n-angles 12 \
  --widths 24 48 96 192 384 --num-workers 8 --seed 2026 \
  --ema-decay 0.999 --ema-exclude agg_alpha --augment --device cuda
touch "$REPO/DATA_DIFFIM_stream_buffer/STOP"   # halt the producer
echo "SEG_MODEL STREAM DONE $(date -Is)"
