#!/bin/bash
# Synthetic objectwise BAR gate: shipped vs fine-tuned on test_5sigma.
# Promote the FT artifacts only if ft holds ~840 cTP at <= ~10k cFP.
# Usage: sbatch ADCNN/scripts/test_real/slurm_bar.sh
#
#SBATCH --job-name=adc-bar
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_bar_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
SYN="${REPO_DIR}/DATA_DIFFIM"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts"
FT="${REPO_DIR}/experiments/diffim_runs/v7_ft_hn/ckpts"
RES="${REPO_DIR}/experiments/diffim_runs/test_real/results"
cd "${REPO_DIR}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
M=ADCNN.evaluation.fp_analysis
python -m $M bar --syn-root "${SYN}" --model "${CK}/v7_scripted.pt.preft.bak" \
  --rf "${CK}/rf_postproc_v2.pkl.preft.bak" --tag shipped \
  --splits test_5sigma --results-dir "${RES}" 2>/dev/null \
  || python -m $M bar --syn-root "${SYN}" --model "${CK}/v7_scripted.pt" \
       --rf "${CK}/rf_postproc_v2.pkl" --tag shipped \
       --splits test_5sigma --results-dir "${RES}"
python -m $M bar --syn-root "${SYN}" \
  --model "${FT}/v7_ft_last_scripted.pt" --rf "${CK}/rf_postproc_v2_ft.pkl" \
  --tag ft --splits test_5sigma --results-dir "${RES}"
echo "BAR DONE $(date -Is)"
