#!/bin/bash
#SBATCH --job-name=adc-lsst-fphot
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=5G
#SBATCH --time=04:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_lsst_fphot_%j.out
set -eo pipefail
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
LIMIT="${LIMIT:-0}"
OUT="${OUT:-/sdf/scratch/users/m/mrakovci/realneg/per_sighting_forced_lsst.csv}"
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python experiments/explore_realneg_train/forced_photometry_lsst.py \
  --per-sighting "$REPO/experiments/diffim_runs/test_real/results/per_sighting.csv" \
  --test-csv     "$REPO/DATA_DIFFIM/test_real/test.csv" \
  --out-csv      "$OUT" \
  --parallel 40 --limit "$LIMIT"
echo "LSST-FPHOT DONE $(date -Is) -> $OUT"
