#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -J blind_detect
#SBATCH -o %x_%A_%a.log
set -eo pipefail
# BLIND-TEST detection (EVALUATION_CONTRACT.md): inline injection + mask flags for one field
# (array index = field k). --cnn-thr 0.50 = the contract's S>=0.50 retention floor.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
RUN=${RUN:?set RUN}
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
k=$SLURM_ARRAY_TASK_ID
M=$RUN/manifest_$k.csv
[ -f "$M" ] || { echo "no manifest $M -- skip"; exit 0; }
[ -f "$RUN/adcnn_dets_masked_$k.csv" ] && { echo "masked_$k exists -- skip"; exit 0; }
python ADCNN/pipelines/heliolinc/discover_stream.py --n-gpus 4 --cnn-thr 0.50 \
  --manifest "$M" --out "$RUN/adcnn_dets_$k.csv" --inject "$RUN/inject_$k.csv"
python ADCNN/pipelines/heliolinc/mask_flags.py \
  --dets "$RUN/adcnn_dets_$k.csv" --manifest "$M" --out "$RUN/adcnn_dets_masked_$k.csv" --workers 32
echo "DETECT DONE $k"
