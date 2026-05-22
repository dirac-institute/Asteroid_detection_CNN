#!/bin/bash
#SBATCH --job-name=adc-v7-huge
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_v7_huge_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
D="$REPO/DATA_DIFFIM_realistic"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
echo "=== v7 on sharded realistic (4 shards, multi-h5) + augment === $(date -Is)"
srun python3 -u -m ADCNN.training.diffim_train \
  --run-name pilot_v7_huge \
  --data-sources \
    "$D/shard_0/train.h5:$D/shard_0/train.csv" \
    "$D/shard_1/train.h5:$D/shard_1/train.csv" \
    "$D/shard_2/train.h5:$D/shard_2/train.csv" \
    "$D/shard_3/train.h5:$D/shard_3_train.csv" \
  --data-h5 "$D/shard_3/train.h5" --data-csv "$D/shard_3_val.csv" \
  --tile 128 --epochs 60 --batch-size 24 --lr 0.0003 --wd 1e-5 \
  --n-pos-anchors-per-epoch 3000 --n-neg-anchors-per-epoch 900 --n-val-panels 64 \
  --anchor-jitter 48 --aftl-alpha 0.3 --aftl-beta 0.7 --aftl-gamma 1.3 \
  --aftl-bce-anchor 0.1 --lambda-orient 0.5 --kernel-lens 11 21 41 --n-angles 12 \
  --widths 24 48 96 192 384 --num-workers 8 --seed 2026 \
  --ema-decay 0.999 --ema-exclude agg_alpha --augment --device cuda
echo "V7 HUGE DONE $(date -Is)"
