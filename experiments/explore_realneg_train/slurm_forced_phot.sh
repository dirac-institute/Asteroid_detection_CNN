#!/bin/bash
#SBATCH --job-name=adc-forced-phot
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=5G
#SBATCH --time=04:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_forced_phot_%j.out
set -eo pipefail
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
OUT=/sdf/scratch/users/m/mrakovci/realneg/per_sighting_forced.csv
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python experiments/explore_realneg_train/forced_photometry.py \
  --per-sighting "$REPO/experiments/diffim_runs/test_real/results/per_sighting.csv" \
  --test-csv     "$REPO/DATA_DIFFIM/test_real/test.csv" \
  --out-csv      "$OUT" \
  --parallel 40
echo "FORCED-PHOT DONE $(date -Is) -> $OUT"
