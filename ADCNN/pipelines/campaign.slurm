#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3-00:00:00
#SBATCH -J adcnn_campaign
#SBATCH -o outputs/logs/%x_%j.log
set -uo pipefail
# The campaign DRIVER as a batch job on a CPU node, so it cannot die with a login shell -- the
# nohup driver vanished three times mid-run, leaving nights half-finished. This only orchestrates:
# it calls `./adcnn night`, which submits its OWN GPU detect job and blocks on it (sbatch --wait).
# A 3-day wall covers the whole campaign; it is resumable, so a requeue picks up where it stopped.
#
# Usage:  NIGHTS="20260705 20260706 ..." sbatch ADCNN/pipelines/campaign.slurm
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
: "${NIGHTS:?set NIGHTS='20260705 20260706 ...'}"
bash ADCNN/pipelines/run_campaign.sh $NIGHTS
