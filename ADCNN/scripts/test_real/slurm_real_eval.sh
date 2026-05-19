#!/bin/bash
# Score v7+V2 on test_real vs the stack — 20-shard GPU array; the shard-0
# afterok dependency merges into results/summary.txt.
# Usage: sbatch ADCNN/scripts/test_real/slurm_real_eval.sh
#
#SBATCH --job-name=adc-real-eval
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_real_eval_%A_%a.out
set -euo pipefail
NSHARDS=20
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA="${REPO_DIR}/DATA_DIFFIM/test_real"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts"
cd "${REPO_DIR}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
srun python -m ADCNN.evaluation.real_eval score \
  --data "${DATA}" --model "${CK}/v7_scripted.pt" \
  --rf "${CK}/rf_postproc_v2.pkl" \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}"
# Last shard merges once the others have flushed (simple barrier: only
# task 0 merges after a short wait; or run merge manually afterwards).
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  echo "[shard0] run after all shards finish:"
  echo "  python -m ADCNN.evaluation.real_eval merge --data ${DATA}"
fi
echo "REAL-EVAL SHARD ${SLURM_ARRAY_TASK_ID} DONE"
