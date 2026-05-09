#!/bin/bash
# Generate three diffim TEST datasets that differ ONLY in the LSST stack
# source-detection threshold (5sigma, 4sigma, 3sigma). Used to compare the
# CNN's combined detector against the stack at progressively looser thresholds
# in Evaluation.ipynb.
#
# Mirrors the pattern of SNR_experiments/slurm_eval_points.sh (which did this
# for direct-image point sources): one SLURM array task per threshold, same
# visit pool across all three (deterministic --seed), 50 test panels each.
#
# Output:
#   DATA_DIFFIM/test_5sigma/test.h5  test.csv
#   DATA_DIFFIM/test_4sigma/test.h5  test.csv
#   DATA_DIFFIM/test_3sigma/test.h5  test.csv
#
# Usage:
#   sbatch ADCNN/data/dataset_creation/slurm_inject_diffim_threshold.sh
#
#SBATCH --requeue
#SBATCH --job-name=adc-diffim-threshold
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_inject_diffim_threshold_%a.out
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=3G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2

set -eo pipefail

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib

REPO_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN"
cd "${REPO_DIR}/ADCNN/data/dataset_creation"

BASE_OUT="${REPO_DIR}/DATA_DIFFIM"
REPO="dp2_prep"
STAGE3="LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP="lsst_cells_v2"
WHERE="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') "

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
  0) LABEL="5sigma"; THRESHOLD="5.0" ;;
  1) LABEL="4sigma"; THRESHOLD="4.0" ;;
  2) LABEL="3sigma"; THRESHOLD="3.0" ;;
  *) echo "[launcher] Unknown SLURM_ARRAY_TASK_ID=${TASK_ID}"; exit 2 ;;
esac

OUT="${BASE_OUT}/test_${LABEL}"
mkdir -p "${OUT}"
rm -f "${OUT}/test.h5" "${OUT}/test.csv"

echo "================== Threshold ${LABEL} (--stack-detection-threshold ${THRESHOLD}) =================="
echo "Output: ${OUT}"

# Same injection settings as the (working) training run:
#   slurm_inject_diffim.sh -- SNR mode, SNR 2-8, length 6-60 px.
# --test-only --train-test-split 0.94117 --random-subset 850
# yields ~50 panels in test.h5 (same visit selection across all 3 thresholds
# because seed=123 is the default and deterministic).
srun python3 -u simulate_inject_diffim.py \
  --repo "$REPO" \
  --collections "$STAGE3" "$STAGE2" \
  --stage3-collection "$STAGE3" \
  --skymap "$SKYMAP" \
  --save-path "${OUT}" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 2 --mag-max 8 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --stack-detection-threshold "${THRESHOLD}" \
  --chunks 128 \
  --test-only \
  --where "$WHERE"
