#!/bin/bash
#SBATCH --job-name=adc-retrain-realistic
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_retrain_realistic_%j.out
set -euo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "=== retrain realistic === $(date -Is)"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/retrain_realistic_rf.py
echo "RETRAIN JOB DONE $(date -Is)"
