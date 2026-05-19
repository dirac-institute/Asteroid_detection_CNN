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
echo "RESTORED the pre-finetune ckpts."
echo "NOTE: ADCNN.inference.diffim_postproc_v2.DEFAULT_THR stays 0.50 (the"
echo "  committed source value; validate_pipeline.py asserts 0.50). 0.10 was"
echo "  only the operating point for the OLD pre-finetune RF — if you fully"
echo "  revert to that RF and want the old curve, edit DEFAULT_THR to 0.10"
echo "  yourself, but then the self-check (step 0b) will FAIL by design."
