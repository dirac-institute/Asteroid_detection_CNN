#!/bin/bash
# Submit a 3-node FP-budget run chained by dependency: prep -> shard(array 0-2) -> finalize.
# Usage: submit_fpb_mn.sh <pyscript.py> <fpp> [extra prep/finalize args, e.g. --nneo 50]
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
PY="$1"; FPP="$2"; shift 2; EXTRA="${*:-}"
export FPB_PY="$PY" FPB_FPP="$FPP" FPB_EXTRA="$EXTRA"
EXP="ALL,FPB_PY,FPB_FPP,FPB_EXTRA"
P=$(sbatch --parsable --export="$EXP" "$HL/fpb_prep.slurm")
A=$(sbatch --parsable --dependency=afterok:$P --export="$EXP" "$HL/fpb_shard.slurm")
F=$(sbatch --parsable --dependency=afterok:$A --export="$EXP" "$HL/fpb_finalize.slurm")
echo "$(basename $PY) fpp=$FPP  ${EXTRA}:  prep=$P  shard(array)=$A  finalize=$F"
