#!/bin/bash
#SBATCH --job-name=adc-rfkiller
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_rfkiller_%j.out
set -euo pipefail
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "=== rf-killer === $(date -Is)"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/probe_rf_killer.py
echo "RFKILLER JOB DONE $(date -Is)"
