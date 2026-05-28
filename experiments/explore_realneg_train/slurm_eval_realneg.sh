#!/bin/bash
# EXPERIMENTAL realneg reuse-eval + kill-switch. Apples-to-apples on the
# SAME leakage-clean held-out real empties: promoted FT vs realneg-FT.
# Reuses tracked ADCNN.evaluation.fp_analysis (dump-empty/dump-syn/fp-fix/bar).
# Usage: sbatch experiments/explore_realneg_train/slurm_eval_realneg.sh
#
#SBATCH --job-name=adc-realneg-eval
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_realneg_eval_%j.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
# Heavy I/O on the 80G scratch area (repo quota full).
BASE="/sdf/scratch/users/m/mrakovci/realneg"
RES="${BASE}/eval"
HELD="${BASE}/data/heldout"
SYN="${REPO_DIR}/DATA_DIFFIM/test_5sigma"                          # read-only
CK="${REPO_DIR}/experiments/diffim_runs/pilot_seg/ckpts"            # promoted ft (read-only)
RNM="${BASE}/runs/seg_ft_realneg/ckpts/seg_ft_realneg_scripted.pt"   # new model
RNRF="${BASE}/ckpts/rf_postproc_v2_realneg.pkl"                    # new RF (made below)
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
test -f "${RNM}" || { echo "MISSING realneg model ${RNM} (run finetune first)"; exit 2; }
test -f "${HELD}/data.h5" || { echo "MISSING ${HELD}/data.h5 (run build first)"; exit 2; }
mkdir -p "${RES}/parts" "${BASE}/ckpts"
# dump-empty expects <data>/test.h5 + panels.csv(role==empty). Provide it.
ln -sf data.h5 "${HELD}/test.h5"

M=ADCNN.evaluation.fp_analysis
# 1) held-out real-empty feature dumps — SAME panels, both models.
python -m $M dump-empty --data "${HELD}" --model "${CK}/segmentation_scripted.pt" \
  --rf "${CK}/rf_postproc_v2.pkl" --tag emporig --results-dir "${RES}"
python -m $M dump-empty --data "${HELD}" --model "${RNM}" \
  --rf "${CK}/rf_postproc_v2.pkl" --tag emprn   --results-dir "${RES}"
# 2) synthetic test_5sigma feature dumps (both models) — posR side.
python -m $M dump-syn --syn-dir "${SYN}" --model "${CK}/segmentation_scripted.pt" \
  --tag ftbase --results-dir "${RES}"
python -m $M dump-syn --syn-dir "${SYN}" --model "${RNM}" \
  --tag rn     --results-dir "${RES}"
# 3) fp-fix: OLD=promoted ft (syn5_ftbase + emporig), NEW=realneg
#    (syn5_rn + emprn). Trains the realneg RF, prints genuine FP/CCD + posR
#    on the SAME held-out empties.
python -m $M fp-fix --results-dir "${RES}" \
  --syn-cached-pkl "${RES}/syn5_ftbase.pkl" \
  --syn-csv "${SYN}/test.csv" --old-rf "${CK}/rf_postproc_v2.pkl" \
  --ft-syn-pkl "${RES}/syn5_rn.pkl" \
  --old-tag emporig --ft-tag emprn \
  --ckpt-out "${RNRF}"
# 4) synthetic objectwise BAR — must not regress vs bar_ft.txt.
# bar() does `Path(syn_root) / split`, so syn_root must be the PARENT of the
# split dirs (DATA_DIFFIM/test_5sigma, test_3sigma, ...), not the leaf.
python -m $M bar --syn-root "${REPO_DIR}/DATA_DIFFIM" --model "${RNM}" --rf "${RNRF}" \
  --tag realneg --splits test_5sigma --results-dir "${RES}"
# 5) kill-switch decision.
python "${REPO_DIR}/experiments/explore_realneg_train/killswitch.py" \
  --eval-dir "${RES}" \
  --baseline-bar "${REPO_DIR}/experiments/diffim_runs/test_real/results/bar_ft.txt"
echo "REALNEG-EVAL DONE $(date -Is)"
