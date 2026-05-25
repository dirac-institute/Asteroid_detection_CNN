#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 2:30:00
#SBATCH -J adcnn_disco
#SBATCH -o %x_%j.log
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks /sdf/data/rubin mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
python experiments/heliolinc/discover_stream.py --n-gpus 4 --rf-thr "${RF_THR:-0.5}" \
  ${MANIFEST:+--manifest "$MANIFEST"} ${OUT:+--out "$OUT"} ${LIMIT:+--limit $LIMIT}
echo "DISCOVER STREAM DONE"
