#!/bin/bash
# Scan Butler availability OR build the real-diffim test set.
# Usage:  sbatch slurm_build.sh scan  /path/real.csv
#         sbatch slurm_build.sh build /path/real.csv
#
#SBATCH --job-name=adc-test-real-build
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=3G
#SBATCH --time=2-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_test_real_build_%j.out
set -eo pipefail
MODE="${1:?scan|build}"; REAL_CSV="${2:?path to real fast-mover csv}"
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
OUT="${REPO_DIR}/DATA_DIFFIM/test_real"
MAN="${OUT}/manifest.csv"
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
if [[ "${MODE}" == "scan" ]]; then
  srun python -m ADCNN.data.dataset_creation.build_test_real scan \
    --real-csv "${REAL_CSV}" --out-dir "${OUT}" --n-empty 150
else
  srun python -m ADCNN.data.dataset_creation.build_test_real build \
    --real-csv "${REAL_CSV}" --manifest "${MAN}" --out "${OUT}" \
    --threshold 5.0 --parallel "${SLURM_CPUS_PER_TASK:-40}" --chunks 128
fi
echo "BUILD ${MODE} DONE $(date -Is)"
