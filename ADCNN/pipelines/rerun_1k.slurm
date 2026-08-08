#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -J rerun_1k
#SBATCH -o outputs/logs/%x_%j.log
set -uo pipefail
# Build the 1k-cadence product for all listed nights as a batch job (survives login-shell death).
# Reuses existing linking; the cost is the mosaic cutout S3 reads for the top-K alerts per night.
# kipac:kipac/roma has idle headroom (rubin:commissioning hits AssocGrpNodeLimit). Resumable:
# a night with its alerts.jsonl + sheets already present is skipped.
#
# Usage:  NIGHTS="20260629 20260630 ..." sbatch ADCNN/pipelines/rerun_1k.slurm
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
cd "$REPO"
: "${NIGHTS:?set NIGHTS='20260629 20260630 ...'}"
bash ADCNN/pipelines/rerun_1k.sh $NIGHTS
