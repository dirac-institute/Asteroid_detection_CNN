#!/bin/bash
#SBATCH --job-name=adc-score-real-realistic
#SBATCH --requeue
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=2,3,4,5
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_score_real_realistic_%A_%a.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
NSHARDS=20
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
MODEL="${REPO_DIR}/experiments/diffim_runs/pilot_v7_realistic/ckpts/v7_realistic_scripted.pt"
RF="${REPO_DIR}/experiments/explore_simreal_gap/rf_postproc_v2_realistic_neg5.pkl"
RES="/sdf/scratch/users/m/mrakovci/realistic/test_real_realistic"
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "=== realistic score shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS} === $(date -Is)"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/diffim_runs/test_real/score_test_real.py \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" \
  --model "${MODEL}" --rf "${RF}" --res "${RES}"
echo "REALISTIC SHARD ${SLURM_ARRAY_TASK_ID} DONE"
