#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-inject
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_train_inject.out
#SBATCH --partition=milano
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

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2025_50/loadLSST.sh
setup lsst_distrib

cd /sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN/data/dataset_creation

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA"
REPO="/repo/main"
COLL="LSSTComCam/runs/DRP/DP1/w_2025_17/DM-50530"
WHERE="instrument='LSSTComCam' AND skymap='lsst_cells_v1' AND day_obs>=20241101 AND day_obs<=20241127 AND exposure.observation_type='science' AND band in ('u','g','r','i','z','y') AND (exposure not in (2024110600163, 2024112400111, 2024110800318, 2024111200185, 2024111400039, 2024111500225, 2024111500226, 2024111500239, 2024111500240, 2024111500242, 2024111500288, 2024111500289, 2024111800077, 2024111800078, 2024112300230, 2024112400094, 2024112400225, 2024112600327))"

mkdir -p "$OUT"
BAD="$OUT/bad_visits.csv"

echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-8}"
echo "Flags: scan=$DO_SCAN test_only=$([[ -n "$TEST_ONLY_FLAG" ]] && echo 1 || echo 0)"

if [[ "$DO_SCAN" -eq 1 ]]; then
  echo "Running bad-visits scan..."
  srun python3 -u check_bad_visits.py \
    --repo "$REPO" \
    --collections "$COLL" \
    --where "$WHERE" \
    --out "$BAD"
else
  echo "Skipping bad-visits scan"
fi

# Clean outputs safely
rm -f "$OUT/test.h5" "$OUT/test.csv"
if [[ -z "$TEST_ONLY_FLAG" ]]; then
  rm -f "$OUT/train.h5" "$OUT/train.csv"
fi

srun python3 -u simulate_inject.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT" \
  --bad-visits-file "$BAD" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 0.01 --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --chunks 128 \
  $TEST_ONLY_FLAG \
  --where "$WHERE"
