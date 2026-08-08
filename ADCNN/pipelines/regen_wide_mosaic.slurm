#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH -c 20
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -J regen_wide
#SBATCH -o outputs/logs/%x_%j.log
set -uo pipefail
# Regenerate the wide-view mosaic for all listed nights as a batch job (survives login-shell death,
# like campaign.slurm). Heavy S3 I/O: each visit's overlapping detector panels are read once, and
# the wide pass caps its own worker count internally. 96G covers the per-visit panel cache.
#
# Usage:  NIGHTS="20260629 20260630 ..." sbatch ADCNN/pipelines/regen_wide_mosaic.slurm
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
cd "$REPO"
: "${NIGHTS:?set NIGHTS='20260629 20260630 ...'}"
bash ADCNN/pipelines/regen_wide_mosaic.sh $NIGHTS
