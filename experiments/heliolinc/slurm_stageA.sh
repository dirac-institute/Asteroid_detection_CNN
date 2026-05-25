#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 1:00:00
#SBATCH -J adcnn_stageA
#SBATCH -o %x_%j.log
# Stage A: ADCNN detection catalog over the discovery window (4 GPUs).
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks /sdf/data/rubin mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/experiments/heliolinc
RUN=${RUN:-$HL/run_disco}
RF_THR=${RF_THR:-0.15}
mkdir -p "$RUN"
cd "$REPO"
python -m ADCNN.inference.catalog \
  --h5 "$REPO/DATA_DIFFIM/test_real/test.h5" \
  --panels "$HL/window_panels.csv" --panel-ids "$HL/window_panels.csv" \
  --rf-thr "$RF_THR" --gate-pmax 0.10 --n-gpus 4 \
  --out "$RUN/catalog.csv"
echo "STAGE A DONE -> $RUN/catalog.csv"
