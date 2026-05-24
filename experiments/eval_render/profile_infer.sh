#!/bin/bash
#SBATCH --job-name=adc-profinf
#SBATCH -A kipac:kipac -p ampere --gres=gpu:1 -c 8 --mem=48G -t 00:20:00
#SBATCH --exclude=sdfampere017 -o /sdf/home/m/mrakovci/logs/ADCNN_profinf_%j.out
set -euo pipefail
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN; export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python experiments/eval_render/profile_infer.py
