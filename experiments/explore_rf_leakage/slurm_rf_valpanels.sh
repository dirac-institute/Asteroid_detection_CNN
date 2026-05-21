#!/bin/bash
# Leak-free stage-2 RF: train on the 50 held-out val panels, eval on test_5sigma
# with old (leaky) vs new (clean) RF. New RF written to scratch, NOT promoted.
#SBATCH --job-name=adc-rf-leak
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_rf_leak_%j.out
set -euo pipefail
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python experiments/explore_rf_leakage/train_eval_rf_valpanels.py
echo "RF-LEAK DONE $(date -Is)"
