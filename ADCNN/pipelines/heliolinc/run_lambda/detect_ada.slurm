#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -J lam_detect
#SBATCH -o %x_%A_%a.log
set -eo pipefail
# ADCNN detection (inline injection) + mask flags for one field (array index = field k). ada/kipac normal
# QOS = NON-PREEMPTABLE L40S. Needs inject_$k.csv from inject.slurm.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
RUN=${RUN:?set RUN}
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
k=$SLURM_ARRAY_TASK_ID
M=$RUN/manifest_$k.csv
[ -f "$M" ] || { echo "no manifest $M -- skip"; exit 0; }
[ -f "$RUN/adcnn_dets_masked_$k.csv" ] && { echo "masked_$k exists -- skip"; exit 0; }
python ADCNN/inference/detect_night.py --n-gpus 4 \
  --manifest "$M" --out "$RUN/adcnn_dets_$k.csv" --inject "$RUN/inject_$k.csv"
python ADCNN/pipelines/heliolinc/mask_flags.py \
  --dets "$RUN/adcnn_dets_$k.csv" --manifest "$M" --out "$RUN/adcnn_dets_masked_$k.csv" --workers 32
echo "DETECT DONE $k"
