#!/bin/bash
#SBATCH --job-name=adc-run-eval-nb
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_run_eval_nb_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
echo "=== run Evaluation.ipynb (realistic v7+neg5 RF) === $(date -Is)"
srun jupyter nbconvert --to notebook --execute --inplace Evaluation.ipynb \
  --ExecutePreprocessor.timeout=4000
echo "EVAL NB DONE $(date -Is)"
