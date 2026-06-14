#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -J v2_detect
#SBATCH -o %x_%A_%a.log
set -eo pipefail
# ADCNN v2 Phase-3 eval ladder: rerun dev detection with a VARIANT segmentation model.
# Env: SEGMODEL (scripted .pt), OUTDIR (run_dev/v2_<name>, pre-made with symlinked inject/manifest).
# Array index = dev field k (0-20). Identical flags to the v1 run EXCEPT --seg-model.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
RUN=${RUN:-$REPO/ADCNN/pipelines/heliolinc/run_dev}
OUT=${OUTDIR:?}; SEG=${SEGMODEL:?}; CNNMODEL=${CNNMODEL:?}
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
k=$SLURM_ARRAY_TASK_ID
M=$RUN/manifest_$k.csv
[ -f "$M" ] || { echo "no manifest $M -- skip"; exit 0; }
[ -f "$OUT/adcnn_dets_masked_$k.csv" ] && { echo "masked_$k exists -- skip"; exit 0; }
python ADCNN/pipelines/heliolinc/discover_stream.py --n-gpus 4 --cnn-thr 0.50 \
  --seg-model "$SEG" --cnn "$CNNMODEL" --cnn-thr 0.50 \
  --manifest "$M" --out "$OUT/adcnn_dets_$k.csv" --inject "$RUN/inject_$k.csv"
python ADCNN/pipelines/heliolinc/mask_flags.py \
  --dets "$OUT/adcnn_dets_$k.csv" --manifest "$M" --out "$OUT/adcnn_dets_masked_$k.csv" --workers 16
echo "V2_DETECT DONE $k"
