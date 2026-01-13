#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-special-inject
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_special_inject.out
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=51
#SBATCH --mem=20G
#SBATCH --time=3-00:00:00

set -eo pipefail

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2025_50/loadLSST.sh
setup lsst_distrib

cd /sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN/data/dataset_creation

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
REPO="/repo/main"
COLL="LSSTComCam/runs/DRP/DP1/w_2025_17/DM-50530"
WHERE="instrument='LSSTComCam' AND skymap='lsst_cells_v1' AND day_obs>=20241101 AND day_obs<=20241127 AND exposure.observation_type='science' AND band in ('u','g','r','i','z','y') AND (exposure not in (2024110600163, 2024112400111, 2024110800318, 2024111200185, 2024111400039, 2024111500225, 2024111500226, 2024111500239, 2024111500240, 2024111500242, 2024111500288, 2024111500289, 2024111800077, 2024111800078, 2024112300230, 2024112400094, 2024112400225, 2024112600327))"

mkdir -p "$OUT/4_4s_6_60_l/"
mkdir -p "$OUT/5_5s_6_60_l/"
mkdir -p "$OUT/6_6s_6_60_l/"

mkdir -p "$OUT/2_10s_10_10_l/"
mkdir -p "$OUT/2_10s_30_30_l/"
mkdir -p "$OUT/2_10s_50_50_l/"

BAD="$OUT/bad_visits.csv"

# Clean outputs safely
rm -f "$OUT/4_4s_6_60_l/test.h5" "$OUT/4_4s_6_60_l/test.csv"
rm -f "$OUT/5_5s_6_60_l/test.h5" "$OUT/5_5s_6_60_l/test.csv"
rm -f "$OUT/6_6s_6_60_l/test.h5" "$OUT/6_6s_6_60_l/test.csv"

rm -f "$OUT/2_10s_10_10_l/test.h5" "$OUT/2_10s_10_10_l/test.csv"
rm -f "$OUT/2_10s_30_30_l/test.h5" "$OUT/2_10s_30_30_l/test.csv"
rm -f "$OUT/2_10s_50_50_l/test.h5" "$OUT/2_10s_50_50_l/test.csv"


echo "================== Running 4-sigma injection =================="
srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/4_4s_6_60_l/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 4 --mag-max 4 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"

echo "================== Running 5-sigma injection =================="
srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/5_5s_6_60_l/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
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
  --save-path "$OUT/6_6s_6_60_l/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 6 --mag-max 6 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"

echo "================== Running 10 len injection =================="
  srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/2_10s_10_10_l/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 10 --trail-length-max 10 \
  --mag-min 2 --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"

echo "================== Running 30 len injection =================="
  srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/2_10s_30_30_l/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 30 --trail-length-max 30 \
  --mag-min 2 --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"

echo "================== Running 50 len injection =================="
  srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT/2_10s_50_50_l/" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 50 --trail-length-max 50 \
  --mag-min 2 --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  --test-only \
  --where "$WHERE"