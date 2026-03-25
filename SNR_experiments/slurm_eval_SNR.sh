#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-inject
#SBATCH --account=rubin:developers
#SBATCH --output=/sdf/home/m/mrakovci/logs/SNR_exp.out
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=3G
#SBATCH --time=3-00:00:00

set -eo pipefail

SNR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snr)
      SNR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: sbatch $0 --snr <value>"
      exit 1
      ;;
  esac
done

if [[ -z "$SNR" ]]; then
  echo "Error: --snr is required"
  echo "Usage: sbatch $0 --snr <value>"
  exit 1
fi

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib

cd /sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/ADCNN/data/dataset_creation

OUT="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/SNR_${SNR}"
REPO="dp2_prep"
COLL="LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
WHERE="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y') "

mkdir -p "$OUT"

rm -f "$OUT/test.h5" "$OUT/test.csv"

srun python3 -u simulate_inject_fill_deterministic.py \
  --repo "$REPO" \
  --collections "$COLL" \
  --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-8}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 2 --mag-max 8 \
  --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --stack-detection-threshold "$SNR" \
  --chunks 128 \
  --test-only \
  --where "$WHERE"