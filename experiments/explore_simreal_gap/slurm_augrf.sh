#!/bin/bash
#SBATCH --job-name=adc-augrf
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_augrf_%j.out
set -euo pipefail
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "=== augrf === $(date -Is)"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/augment_rf.py
echo "AUGRF JOB DONE $(date -Is)"
