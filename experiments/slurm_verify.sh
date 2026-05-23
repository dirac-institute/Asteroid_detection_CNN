#!/bin/bash
#SBATCH --job-name=adc-verify
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_verify_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"; cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
srun python3 -u experiments/verify_consolidated.py
