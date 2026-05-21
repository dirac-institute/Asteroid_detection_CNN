#!/bin/bash
#SBATCH --job-name=adc-resim-realistic
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_resim_realistic_%j.out
set -eo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/ADCNN/data/dataset_creation:${PYTHONPATH:-}"
source /sdf/group/rubin/sw/loadLSST.bash
setup lsst_distrib
echo "=== resim realistic val === $(date -Is)"
srun python3 -u experiments/explore_simreal_gap/resim_realistic_val.py \
  --out /sdf/scratch/users/m/mrakovci/resim_realistic_val
echo "RESIM JOB DONE $(date -Is)"
