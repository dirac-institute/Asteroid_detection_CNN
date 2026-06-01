#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH -J sn_detect
#SBATCH -o %x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append
set -eo pipefail
# Canonical same-night pipeline, Stage 1: ADCNN (segmentation + focal cutout CNN) over the manifest's
# diffims -> sky detections + trail endpoints. Resumable (per-panel .done) so a preemption + resubmit
# (driven by sn_run.slurm) continues. $RUN must point at a dir containing manifest.csv.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
: "${RUN:?set RUN=<dir with manifest.csv>}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
python ADCNN/pipelines/heliolinc/discover_stream.py --n-gpus 4 \
  --manifest "$RUN/manifest.csv" --out "$RUN/adcnn_dets.csv"
echo "SN_DETECT DONE"
