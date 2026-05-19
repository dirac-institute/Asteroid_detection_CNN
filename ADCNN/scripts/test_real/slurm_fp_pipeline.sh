#!/bin/bash
# FP analysis on the ORIGINAL (pre-finetune) pipeline:
#   dump-empty (orig model) -> snr-gain -> sweep (16 shards) -> sweep-curve.
# Run AFTER slurm_real_eval.sh (needs results/per_sighting.csv).
# Usage: sbatch ADCNN/scripts/test_real/slurm_fp_pipeline.sh
#
#SBATCH --job-name=adc-fp-pipe
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-15
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_fp_pipe_%A_%a.out
set -euo pipefail
NSHARDS=16
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA="${REPO_DIR}/DATA_DIFFIM/test_real"
RES="${REPO_DIR}/experiments/diffim_runs/test_real/results"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts"
cd "${REPO_DIR}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
M=ADCNN.evaluation.fp_analysis
# every shard contributes to the candidate-score sweep
srun python -m $M sweep --data "${DATA}" --model "${CK}/v7_scripted.pt" \
  --rf "${CK}/rf_postproc_v2.pkl" \
  --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS}" --results-dir "${RES}"
# shard 0 also does the single-process dump-empty + snr-gain + curve
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  python -m $M dump-empty --data "${DATA}" --model "${CK}/v7_scripted.pt" \
    --rf "${CK}/rf_postproc_v2.pkl" --tag emp --results-dir "${RES}"
  python -m $M snr-gain --data "${DATA}" --results-dir "${RES}"
  echo "[shard0] after all sweep shards finish, run:"
  echo "  python -m $M sweep-curve --data ${DATA} --results-dir ${RES}"
fi
echo "FP-PIPE SHARD ${SLURM_ARRAY_TASK_ID} DONE"
