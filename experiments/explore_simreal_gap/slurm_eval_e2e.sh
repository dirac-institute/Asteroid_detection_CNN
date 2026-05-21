#!/bin/bash
#SBATCH --job-name=adc-eval-e2e
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_eval_e2e_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
echo "=== e2e eval (realistic v7+RF) === $(date -Is)"
srun python3 -u experiments/explore_simreal_gap/eval_realistic_e2e.py
echo "EVAL E2E DONE $(date -Is)"
