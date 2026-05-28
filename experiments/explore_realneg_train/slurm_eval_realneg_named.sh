#!/bin/bash
# Parameterised realneg reuse-eval + kill-switch. Same logic as
# slurm_eval_realneg.sh but the run name is passed via $RUN so we can run it
# for cfg0, cfg2, ... in parallel. Eval writes to ${BASE}/eval/${RUN}/.
# Usage: sbatch --export=ALL,RUN=rn_cfg2_fptilt slurm_eval_realneg_named.sh
#
#SBATCH --job-name=adc-rn-eval
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_rn_eval_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
: "${RUN:?must set RUN=<run-name>}"
BASE="/sdf/scratch/users/m/mrakovci/realneg"
RES="${BASE}/eval/${RUN}"
HELD="${BASE}/data/heldout"
SYN="${REPO_DIR}/DATA_DIFFIM/test_5sigma"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_seg/ckpts"
RNM="${BASE}/runs/${RUN}/ckpts/${RUN}_scripted.pt"
RNRF="${BASE}/ckpts/rf_postproc_v2_${RUN}.pkl"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
test -f "${RNM}" || { echo "MISSING ${RNM} (run ft first)"; exit 2; }
test -f "${HELD}/data.h5" || { echo "MISSING ${HELD}/data.h5 (run build first)"; exit 2; }
mkdir -p "${RES}/parts" "${BASE}/ckpts"
ln -sf data.h5 "${HELD}/test.h5"

M=ADCNN.evaluation.fp_analysis
# Held-out real-empty dumps for both models on the SAME panels.
python -m $M dump-empty --data "${HELD}" --model "${CK}/segmentation_scripted.pt" \
  --rf "${CK}/rf_postproc_v2.pkl" --tag emporig --results-dir "${RES}"
python -m $M dump-empty --data "${HELD}" --model "${RNM}" \
  --rf "${CK}/rf_postproc_v2.pkl" --tag emp_${RUN} --results-dir "${RES}"
# Synthetic test_5sigma dumps for both models — posR side.
python -m $M dump-syn --syn-dir "${SYN}" --model "${CK}/segmentation_scripted.pt" \
  --tag ftbase --results-dir "${RES}"
python -m $M dump-syn --syn-dir "${SYN}" --model "${RNM}" \
  --tag syn_${RUN} --results-dir "${RES}"
# Train the realneg RF + print genuine FP/CCD + posR on the SAME held-outs.
python -m $M fp-fix --results-dir "${RES}" \
  --syn-cached-pkl "${RES}/syn5_ftbase.pkl" \
  --syn-csv "${SYN}/test.csv" --old-rf "${CK}/rf_postproc_v2.pkl" \
  --ft-syn-pkl "${RES}/syn5_syn_${RUN}.pkl" \
  --old-tag emporig --ft-tag emp_${RUN} \
  --ckpt-out "${RNRF}"
# Synthetic objectwise BAR — must not regress vs bar_ft.txt.
# bar() does `Path(syn_root) / split`, so syn_root must be the PARENT of the
# split dirs (DATA_DIFFIM/test_5sigma, test_3sigma, ...), not the leaf.
python -m $M bar --syn-root "${REPO_DIR}/DATA_DIFFIM" --model "${RNM}" --rf "${RNRF}" \
  --tag "${RUN}" --splits test_5sigma --results-dir "${RES}"
# Kill-switch decision.
python "${REPO_DIR}/experiments/explore_realneg_train/killswitch.py" \
  --eval-dir "${RES}" \
  --baseline-bar "${REPO_DIR}/experiments/diffim_runs/test_real/results/bar_ft.txt"
echo "REALNEG-EVAL-${RUN} DONE $(date -Is) -> ${RES}"
