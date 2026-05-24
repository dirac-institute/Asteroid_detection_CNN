#!/bin/bash
#SBATCH --job-name=adc-prof
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=sdfampere017
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_prof_%j.out
set -euo pipefail
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python experiments/eval_render/profile_engine.py
