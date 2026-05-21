#!/bin/bash
#SBATCH --job-name=adc-probe-simreal
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_probe_simreal_%j.out
set -euo pipefail
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "=== probe simreal === $(date -Is)"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/probe_prob_at_truth.py
echo "PROBE JOB DONE $(date -Is)"
