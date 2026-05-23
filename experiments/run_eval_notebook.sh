#!/bin/bash
# Execute the trimmed Evaluation.ipynb end-to-end on GPU so the hero
# (5sigma + NN two-stage) tables/histograms/heatmaps + param-recovery render.
# Writes outputs in-place. Original is preserved in git HEAD.
#SBATCH --job-name=adc-eval-nb
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_eval_nb_%j.out
set -euo pipefail
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=7200 \
  Evaluation.ipynb
echo "EVAL-NB DONE $(date -Is)"
