#!/bin/bash
#SBATCH --job-name=adc-valgate
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=sdfampere017
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_valgate_%j.out
set -euo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
[ -d "$REPO" ] || { echo "FATAL no mount on $(hostname)"; scontrol requeue "$SLURM_JOB_ID"||true; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python experiments/eval_render/validate_gate.py
