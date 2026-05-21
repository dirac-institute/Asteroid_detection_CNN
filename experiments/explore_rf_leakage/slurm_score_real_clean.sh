#!/bin/bash
# Re-score test_real with the LEAK-FREE val-trained RF (clean). 20-shard array,
# writes to scratch results dir; merge with merge step after array completes.
#SBATCH --job-name=adc-score-real-clean
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_score_real_clean_%A_%a.out
set -euo pipefail
NSHARDS=20
REPO_DIR="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
RF="${REPO_DIR}/experiments/explore_rf_leakage/rf_postproc_v2_valtrain.pkl"
RES="/sdf/scratch/users/m/mrakovci/rf_leakage/test_real_clean"
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
echo "=== clean score shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS} === $(date -Is)"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
srun python3 -u experiments/diffim_runs/test_real/score_test_real.py \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" \
  --rf "${RF}" --res "${RES}"
echo "CLEAN SHARD ${SLURM_ARRAY_TASK_ID} DONE"
