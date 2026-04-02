#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-points
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_eval_points_%a.out
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-2
#SBATCH --time=3-00:00:00

set -eo pipefail
set +u
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
set -u

PROJECT_ROOT="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
SCRIPT_DIR="${PROJECT_ROOT}/SNR_experiments"
BASE_OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/point_sources1"
REPO="dp2_prep"
COLL="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
WHERE="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') "

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
  0)
    LABEL="5sigma"
    THRESHOLD="5.0"
    ;;
  1)
    LABEL="4sigma"
    THRESHOLD="4.0"
    ;;
  2)
    LABEL="3sigma"
    THRESHOLD="3.0"
    ;;
  *)
    echo "[launcher] Unknown SLURM_ARRAY_TASK_ID=${TASK_ID}"
    exit 2
    ;;
esac

OUT="${BASE_OUT}/${LABEL}"

mkdir -p "${OUT}"
rm -f "${OUT}/train.csv" "${OUT}/test.csv" "${OUT}/train.h5" "${OUT}/test.h5"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ADCNN/data/dataset_creation:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"

srun python3 -u inject_point_sources_lsstcam.py \
  --repo "${REPO}" \
  --collections "${COLL}" \
  --save-path "${OUT}" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 1.0 \
  --random-subset 850 \
  --trail-length-min 0 \
  --trail-length-max 0 \
  --mag-min 2 \
  --mag-max 10 \
  --mag-mode snr \
  --beta-min 0 \
  --beta-max 180 \
  --number 20 \
  --where "${WHERE}" \
  --stack-detection-threshold "${THRESHOLD}"
