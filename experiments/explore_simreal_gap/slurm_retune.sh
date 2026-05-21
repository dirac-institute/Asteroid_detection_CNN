#!/bin/bash
#SBATCH --job-name=adc-retune-rf
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_retune_rf_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
echo "=== retune RF (realistic pool) === $(date -Is)"
srun python3 -u experiments/explore_simreal_gap/retune_rf_realistic.py
echo "RETUNE JOB DONE $(date -Is)"
