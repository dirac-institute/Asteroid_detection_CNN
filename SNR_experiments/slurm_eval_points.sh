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

DO_SCAN=0
TEST_ONLY_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --scan) DO_SCAN=1 ;;
    --test) TEST_ONLY_FLAG="--test-only" ;;
  esac
done

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_07/loadLSST.sh
setup lsst_distrib

cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_SNR/SNR_experiments/

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/point_sources"
REPO="dp2_prep"
COLL="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
WHERE="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') "

mkdir -p "$OUT"
BAD="$OUT/bad_visits.csv"

if [[ "$DO_SCAN" -eq 1 ]]; then
  echo "Running bad-visits scan..."
  srun python3 -u scan_bad_data.py \
    --repo "$REPO" \
    --collections "$COLL" \
    --where "$WHERE" \
    --out "$BAD" \
    --no-progress
else
  echo "Skipping bad-visits scan"
fi

# Clean outputs safely
rm -f "$OUT/test.h5" "$OUT/test.csv"
if [[ -z "$TEST_ONLY_FLAG" ]]; then
  rm -f "$OUT/train.h5" "$OUT/train.csv"
fi

export PYTHONPATH=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_SNR:$PYTHONPATH

srun python3 -u inject_point_sources_lsstcam.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 1.0 \
  --random-subset 850 \
  --trail-length-min 0 --trail-length-max 0 \
  --mag-min 2 --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  $TEST_ONLY_FLAG \
  --where "$WHERE"
