#!/bin/bash
#SBATCH --job-name=adc-eval-nb
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=sdfampere017
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_eval_nb_%j.out
set -euo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
# Some ampere nodes don't mount /sdf/data/rubin; fail fast + requeue instead of
# silently running from /tmp (which produced a 1-second "couldn't chdir" failure).
if [ ! -d "$REPO" ]; then
  echo "FATAL: $REPO not mounted on $(hostname) — requeueing" >&2
  scontrol requeue "$SLURM_JOB_ID" || true
  exit 1
fi
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== rendering synthetic Evaluation.ipynb (GPU inference on 5/4/3-sigma) ==="
python experiments/eval_render/run_nb.py Evaluation/Evaluation.ipynb
echo "=== rendering Evaluation_Real.ipynb (CSV + plots) ==="
python experiments/eval_render/run_nb.py Evaluation/Evaluation_Real.ipynb
echo "EVAL-NB DONE $(date -Is)"
