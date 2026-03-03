#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-inject
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_train_inject.out
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3-00:00:00

set -eo pipefail

TEST_ONLY_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --test) TEST_ONLY_FLAG="--test-only" ;;
  esac
done

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_07/loadLSST.sh
setup lsst_distrib

cd /sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN/data/dataset_creation

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
REPO="dp2_prep"
COLL="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
WHERE="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') "

mkdir -p "$OUT"

# Clean outputs safely
rm -f "$OUT/test.h5" "$OUT/test.csv"
if [[ -z "$TEST_ONLY_FLAG" ]]; then
  rm -f "$OUT/train.h5" "$OUT/train.csv"
fi

srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 2 --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  $TEST_ONLY_FLAG \
  --where "$WHERE"
