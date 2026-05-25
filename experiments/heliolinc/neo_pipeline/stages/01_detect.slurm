#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 2:30:00
#SBATCH -J neo1_detect
#SBATCH -o %x_%j.log
set -eo pipefail
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_pipeline/config.sh"
eval "$TORCH_ENV"
cd "$REPO"
# Stage 1: stream butler diffims -> ADCNN+RF -> sky detection catalog + de-biased trail endpoints.
python experiments/heliolinc/discover_stream.py --n-gpus "$N_GPUS" --rf-thr "$RF_THR" \
  --manifest "$MANIFEST" --out "$RUN/adcnn_dets.csv"
echo "NEO STAGE1 DETECT DONE"
