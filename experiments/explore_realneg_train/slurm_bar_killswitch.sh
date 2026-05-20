#!/bin/bash
# Rerun only the synthetic-objectwise BAR + killswitch step for a given run
# (fp-fix step already produced fp_fix.txt; this is the BAR gate retry with
# the corrected --syn-root path).
# Usage: sbatch --export=ALL,RUN=<run-name> slurm_bar_killswitch.sh
#
#SBATCH --job-name=adc-rn-bar
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_rn_bar_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
: "${RUN:?must set RUN=<run-name>}"
BASE="/sdf/scratch/users/m/mrakovci/realneg"
RES="${BASE}/eval/${RUN}"
RNM="${BASE}/runs/${RUN}/ckpts/${RUN}_scripted.pt"
RNRF="${BASE}/ckpts/rf_postproc_v2_${RUN}.pkl"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
test -f "${RNM}" || { echo "MISSING ${RNM}"; exit 2; }
test -f "${RNRF}" || { echo "MISSING ${RNRF} (run fp-fix step first)"; exit 2; }
mkdir -p "${RES}"
# Synthetic objectwise BAR — must not regress vs bar_ft.txt.
python -m ADCNN.evaluation.fp_analysis bar \
  --syn-root "${REPO_DIR}/DATA_DIFFIM" \
  --model "${RNM}" --rf "${RNRF}" \
  --tag "${RUN}" --splits test_5sigma --results-dir "${RES}"
# Kill-switch decision.
python "${REPO_DIR}/experiments/explore_realneg_train/killswitch.py" \
  --eval-dir "${RES}" \
  --baseline-bar "${REPO_DIR}/experiments/diffim_runs/test_real/results/bar_ft.txt"
echo "REALNEG-BAR-${RUN} DONE $(date -Is) -> ${RES}"
