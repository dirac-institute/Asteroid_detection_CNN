#!/bin/bash
#SBATCH --job-name=adc-catalogs
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=sdfampere017
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_catalogs_%j.out
set -euo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
if [ ! -d "$REPO" ]; then echo "FATAL: $REPO not mounted on $(hostname)" >&2; scontrol requeue "$SLURM_JOB_ID" || true; exit 1; fi
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python experiments/eval_render/make_catalogs.py
echo "DONE $(date -Is)"
