#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:2
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -J calib_detect
#SBATCH -o outputs/logs/%x_%j.log
set -eo pipefail
# Detection WITH pair-consistent injection on a real night's own panels -- the labelled set for
# the alert-ranking likelihood fit (ADCNN/calibration/night_pair_injection.py). Injection happens
# in-memory at detect time (detect_night --inject), exactly as the run_lambda campaign did, so the
# negatives are this night's GENUINE residual population. Submit from the repo root.
#
# rubin:commissioning (not kipac:kipac): separate association, so it does not queue behind the
# kipac group node cap that has been holding these jobs.
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
source ADCNN/pipelines/heliolinc/pipeline_config.sh   # S3 creds/endpoints for the embargo datastore
RUN=${RUN:-outputs/runs/run_calib_0630}
python ADCNN/inference/detect_night.py --n-gpus 2 \
  --manifest "$RUN/manifest_inj.csv" --out "$RUN/adcnn_dets.csv" --inject "$RUN/inject.csv"
python ADCNN/pipelines/heliolinc/mask_flags.py \
  --dets "$RUN/adcnn_dets.csv" --manifest "$RUN/manifest_inj.csv" \
  --out "$RUN/adcnn_dets_masked.csv" --workers 32
echo "CALIB_DETECT_DONE"
