#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-inject-diffim
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_train_inject_diffim.out
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=3G
#SBATCH --time=3-00:00:00

set -eo pipefail

TEST_ONLY_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --test) TEST_ONLY_FLAG="--test-only" ;;
  esac
done

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib

cd /sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA_DIFFIM"
REPO="dp2_prep"
STAGE3="LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP="lsst_cells_v2"
WHERE="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') "

mkdir -p "$OUT"

rm -f "$OUT/test.h5" "$OUT/test.csv"
if [[ -z "$TEST_ONLY_FLAG" ]]; then
  rm -f "$OUT/train.h5" "$OUT/train.csv"
fi

# NOTE: add `--exclude-pairs-csv "$OUT/test_5sigma/test.csv" "$OUT/test_real/test.csv"`
# (or the unified --split-json/--split-key flow) to keep training off the held-out test
# panels. --realistic-trail matches the deployed segmentation-model training data.
srun python3 -u -m ADCNN.pipelines.make_sim_data \
  --repo "$REPO" \
  --collections "$STAGE3" "$STAGE2" \
  --stage3-collection "$STAGE3" \
  --skymap "$SKYMAP" \
  --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --realistic-trail \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 2 --mag-max 8 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --stack-detection-threshold 5.0 \
  --chunks 128 \
  $TEST_ONLY_FLAG \
  --where "$WHERE"
