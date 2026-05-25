#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere017
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0:25:00
#SBATCH -J rf_eval
#SBATCH -o %x_%j.log
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
python -u experiments/heliolinc/rf_eval_compare.py
echo "RF EVAL COMPARE DONE"
