#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 1:30:00
#SBATCH -J neo2_measure
#SBATCH -o %x_%j.log
set -euo pipefail
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/neo_pipeline/config.sh"
eval "$LSST_ENV"
cd "$HL"
# Stage 2: precise per-panel-PSF Veres trailed fit -> accurate sky endpoints (parallel over panels).
python -u veres_measure_catalog.py \
  --dets "$RUN/adcnn_dets.csv" --manifest "$MANIFEST" \
  --length-min "$MEAS_LENGTH_MIN" --workers "$MEAS_WORKERS" \
  --out "$RUN/adcnn_dets_veres.csv"
# also build the stack diaSource catalog (carries Rubin real/bogus reliability + trailLength)
python -u butler_diasource_catalog.py --manifest "$MANIFEST" \
  --region-from "$RUN/adcnn_dets.csv" --out "$DIASRC"
echo "NEO STAGE2 MEASURE DONE"
