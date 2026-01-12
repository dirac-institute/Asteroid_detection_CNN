#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-inject
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_fixsigma_inject.out
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00

set -eo pipefail

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2025_50/loadLSST.sh
setup lsst_distrib

cd /sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN/data/dataset_creation

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
REPO="/repo/main"
COLL="LSSTComCam/runs/DRP/DP1/w_2025_17/DM-50530"
WHERE="instrument='LSSTComCam' AND skymap='lsst_cells_v1' AND day_obs>=20241101 AND day_obs<=20241127 AND exposure.observation_type='science' AND band in ('u','g','r','i','z','y') AND (exposure not in (2024110600163, 2024112400111, 2024110800318, 2024111200185, 2024111400039, 2024111500225, 2024111500226, 2024111500239, 2024111500240, 2024111500242, 2024111500288, 2024111500289, 2024111800077, 2024111800078, 2024112300230, 2024112400094, 2024112400225, 2024112600327))"

mkdir -p "$OUT/5sigma/"
mkdir -p "$OUT/6sigma/"
BAD="$OUT/bad_visits.csv"

# Clean outputs safely
rm -f "$OUT/5sigma/test.h5" "$OUT/5sigma/test.csv" "$OUT/6sigma/test.h5" "$OUT/6sigma/test.csv"

echo "================== Running 5-sigma injection =================="
srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/5sigma/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 20 --trail-length-max 20 \
  --mag-min 5 --mag-max 5 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"

echo "================== Running 6-sigma injection =================="
  srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/6sigma/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 20 --trail-length-max 20 \
  --mag-min 6 --mag-max 6 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"
