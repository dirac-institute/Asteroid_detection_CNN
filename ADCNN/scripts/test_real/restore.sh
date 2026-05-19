#!/bin/bash
# Revert promote.sh: restore the pre-finetune shipped ckpts.
# Usage: bash ADCNN/scripts/test_real/restore.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
CK="${REPO_DIR}/experiments/diffim_runs/pilot_v7/ckpts"
for f in v7_scripted.pt rf_postproc_v2.pkl; do
  if [ -f "${CK}/${f}.preft.bak" ]; then
    cp -p "${CK}/${f}.preft.bak" "${CK}/${f}"
    echo "restored ${f} from .preft.bak"
  else
    echo "WARN no backup ${CK}/${f}.preft.bak"
  fi
done
echo "RESTORED. Also set ADCNN.inference.diffim_postproc_v2.DEFAULT_THR back to 0.10 if reverting fully."
