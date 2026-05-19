#!/bin/bash
# Post-fine-tune eval: dump real-empty + synthetic V2 feats with the FT model,
# then fp-fix (fact-correct FP + retrain RF w/ real hard negs; ORIG vs FT).
# Usage: sbatch ADCNN/scripts/test_real/slurm_ft_eval.sh
#
#SBATCH --job-name=adc-ft-eval
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_ft_eval_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA="${REPO_DIR}/DATA_DIFFIM/test_real"
SYN="${REPO_DIR}/DATA_DIFFIM/test_5sigma"
RES="${REPO_DIR}/experiments/diffim_runs/test_real/results"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts"
PI="${REPO_DIR}/experiments/diffim_runs/pilot_v7/postproc_iter"
FT="${REPO_DIR}/experiments/diffim_runs/v7_ft_hn/ckpts/v7_ft_last_scripted.pt"
cd "${REPO_DIR}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
test -f "${FT}" || { echo "MISSING ${FT}"; exit 2; }
M=ADCNN.evaluation.fp_analysis
# FT-model real-empty + synthetic feats (drive the FT side + the promoted RF).
python -m $M dump-empty --data "${DATA}" --model "${FT}" \
  --rf "${CK}/rf_postproc_v2.pkl" --tag empft --results-dir "${RES}"
python -m $M dump-syn --syn-dir "${SYN}" --model "${FT}" --tag ft \
  --results-dir "${RES}"
# ORIGINAL-model synthetic feats — reproducible substitute for the legacy
# (untracked) postproc_iter cache. At this step (pre-promote) ${CK}/v7_scripted.pt
# is still the original shipped model. syn5_orig.pkl carries label_v2, so
# fp-fix needs no --syn-pp-npy. Only the comparison columns use this; the
# promoted rf_postproc_v2_ft.pkl is trained purely from syn5_ft + empft.
python -m $M dump-syn --syn-dir "${SYN}" --model "${CK}/v7_scripted.pt" \
  --tag orig --results-dir "${RES}"
python -m $M fp-fix --results-dir "${RES}" \
  --syn-cached-pkl "${RES}/syn5_orig.pkl" \
  --syn-csv "${SYN}/test.csv" --old-rf "${CK}/rf_postproc_v2.pkl" \
  --ft-syn-pkl "${RES}/syn5_ft.pkl" \
  --ckpt-out "${CK}/rf_postproc_v2_ft.pkl"
# Legacy form (pre-consolidation cache), kept for reference:
#   --syn-cached-pkl "${PI}/test_5sigma_scored_v2.pkl" \
#   --syn-pp-npy "${PI}/test_5sigma_panel_probs_v2.npy" \
echo "FT-EVAL DONE $(date -Is)"
