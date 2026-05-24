#!/bin/bash
#SBATCH -J adc-e2e -A kipac:kipac -p ampere --gres=gpu:1 -c 16 --mem=128G -t 02:00:00
#SBATCH --exclude=sdfampere017 -o /sdf/home/m/mrakovci/logs/ADCNN_e2e_%j.out
set -euo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
[ -d "$REPO" ] || { echo "FATAL no mount on $(hostname)"; scontrol requeue "$SLURM_JOB_ID"||true; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -m ADCNN.pipelines.train_end_to_end --run-name smoke_e2e --epochs 2
echo "SMOKE-E2E DONE $(date -Is)"
