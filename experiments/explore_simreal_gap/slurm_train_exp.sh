#!/bin/bash
#SBATCH --job-name=adc-v7-exp
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_%x_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"; D="$REPO/DATA_DIFFIM_realistic"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
: "${NPOS:=3000}" "${NNEG:=900}" "${LR:=0.0003}" "${EPOCHS:=30}" "${RUNNAME:?}" "${EXTRA:=}"
echo "=== $RUNNAME : NPOS=$NPOS NNEG=$NNEG LR=$LR EPOCHS=$EPOCHS EXTRA='$EXTRA' === $(date -Is)"
srun python3 -u -m ADCNN.training.diffim_train \
  --run-name "$RUNNAME" \
  --data-sources "$D/shard_0/train.h5:$D/shard_0/train.csv" "$D/shard_1/train.h5:$D/shard_1/train.csv" \
                 "$D/shard_2/train.h5:$D/shard_2/train.csv" "$D/shard_3/train.h5:$D/shard_3_train.csv" \
  --data-h5 "$D/shard_3/train.h5" --data-csv "$D/shard_3_val.csv" \
  --tile 128 --epochs "$EPOCHS" --batch-size 24 --lr "$LR" --wd 1e-5 \
  --n-pos-anchors-per-epoch "$NPOS" --n-neg-anchors-per-epoch "$NNEG" --n-val-panels 64 \
  --anchor-jitter 48 --aftl-alpha 0.3 --aftl-beta 0.7 --aftl-gamma 1.3 \
  --aftl-bce-anchor 0.1 --lambda-orient 0.5 --kernel-lens 11 21 41 --n-angles 12 \
  --widths 24 48 96 192 384 --num-workers 8 --seed 2026 \
  --ema-decay 0.999 --ema-exclude agg_alpha --augment --device cuda $EXTRA
echo "$RUNNAME DONE $(date -Is)"
