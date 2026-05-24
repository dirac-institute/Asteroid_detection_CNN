#!/bin/bash
#SBATCH --job-name=adc-fp16 -A kipac:kipac -p ampere --gres=gpu:1 -c 16 --mem=64G -t 01:00:00
#SBATCH --exclude=sdfampere017 -o /sdf/home/m/mrakovci/logs/ADCNN_fp16_%j.out
set -euo pipefail
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN; export PYTHONPATH="$PWD:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python experiments/eval_render/experiment_fp16.py
