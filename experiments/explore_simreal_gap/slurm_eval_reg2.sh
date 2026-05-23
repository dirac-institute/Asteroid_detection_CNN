#!/bin/bash
#SBATCH --job-name=adc-reg2-e2e
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_reg2_e2e_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"; cd "$REPO"
export PYTHONPATH="$REPO:$REPO/experiments/explore_rf_leakage:$REPO/experiments/explore_simreal_gap:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/eval_reg2_full.py
