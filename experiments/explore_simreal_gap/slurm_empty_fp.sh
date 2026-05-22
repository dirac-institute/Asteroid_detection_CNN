#!/bin/bash
#SBATCH --job-name=adc-empty-fp
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_empty_fp_%A_%a.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/probe_empty_fp.py --shard ${SLURM_ARRAY_TASK_ID} --nshards 4
echo "EMPTY-FP SHARD ${SLURM_ARRAY_TASK_ID} DONE"
