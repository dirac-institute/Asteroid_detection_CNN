#!/bin/bash
#SBATCH --job-name=adc-best-vs-last
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_best_vs_last_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
CK="$REPO/experiments/diffim_runs/pilot_seg_big/ckpts"
echo "=== export best+last -> scripted ==="
python3 -u -m ADCNN.inference.diffim_export --ckpt "$CK/best.pt" --out "$CK/seg_big_best_scripted.pt" --no-optimize
python3 -u -m ADCNN.inference.diffim_export --ckpt "$CK/last.pt" --out "$CK/seg_big_last_scripted.pt" --no-optimize
echo "=== eval best vs last on task metric ==="
srun python3 -u experiments/explore_simreal_gap/eval_best_vs_last.py
echo "BEST-VS-LAST JOB DONE $(date -Is)"
