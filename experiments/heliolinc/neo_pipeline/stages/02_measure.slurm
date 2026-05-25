#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 1:30:00
#SBATCH -J neo2_measure
#SBATCH -o %x_%j.log
set -eo pipefail
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_pipeline/config.sh"
eval "$LSST_ENV"
cd "$HL"
# Stage 2: precise per-panel-PSF Veres trailed fit -> accurate sky endpoints (parallel over panels).
python -u veres_measure_catalog.py \
  --dets "$RUN/adcnn_dets.csv" --manifest "$MANIFEST" \
  --score-min "$RF_THR" --length-min "$MEAS_LENGTH_MIN" --workers "$MEAS_WORKERS" \
  --out "$RUN/adcnn_dets_veres.csv"
echo "NEO STAGE2 MEASURE DONE"
