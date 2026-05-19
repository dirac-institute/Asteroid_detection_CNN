#!/bin/bash
# Promote the fine-tuned v7 + hardened RF to the default ckpts, AFTER the
# synthetic bar gate holds. Backs up originals to *.preft.bak (idempotent).
# Reverse with restore.sh. Usage: bash ADCNN/scripts/test_real/promote.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts"
FT="${REPO_DIR}/experiments/diffim_runs/v7_ft_hn/ckpts"
test -f "${FT}/v7_ft_last_scripted.pt" || { echo "no FT scripted model"; exit 2; }
test -f "${CK}/rf_postproc_v2_ft.pkl"  || { echo "no FT rf"; exit 2; }
[ -f "${CK}/v7_scripted.pt.preft.bak" ]   || cp -p "${CK}/v7_scripted.pt"   "${CK}/v7_scripted.pt.preft.bak"
[ -f "${CK}/rf_postproc_v2.pkl.preft.bak" ] || cp -p "${CK}/rf_postproc_v2.pkl" "${CK}/rf_postproc_v2.pkl.preft.bak"
cp -p "${FT}/v7_ft_last_scripted.pt" "${CK}/v7_scripted.pt"
cp -p "${CK}/rf_postproc_v2_ft.pkl"  "${CK}/rf_postproc_v2.pkl"
echo "PROMOTED ft -> default (originals at *.preft.bak)."
echo "Note: ADCNN.inference.diffim_postproc_v2.DEFAULT_THR is 0.50 for ft."
