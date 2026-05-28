#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 1:00:00
#SBATCH -J veres_meas
#SBATCH -o %x_%j.log
set -eo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
# Precise Veres trailed-fit measurement -> sky endpoints, for trailed (fast-mover) detections.
python -u veres_measure_catalog.py \
  --dets NEO_small/adcnn_dets.csv --manifest NEO_small/manifest.csv \
  --score-min 0.5 --length-min 40 --workers 60 \
  --out NEO_small_v2/adcnn_dets_veres.csv
echo "VERES MEASURE DONE"
