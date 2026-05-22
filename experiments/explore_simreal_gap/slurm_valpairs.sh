#!/bin/bash
#SBATCH --job-name=adc-valpairs
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_valpairs_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; source /sdf/group/rubin/sw/loadLSST.bash; setup lsst_distrib
export PYTHONPATH="$REPO:$REPO/ADCNN/data/dataset_creation:${PYTHONPATH:-}"
srun python3 -u experiments/explore_simreal_gap/gen_validated_pairs.py
