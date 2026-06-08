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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pipeline_config.sh"
: "${RUN:?set RUN=<dir with manifest.csv>}"
adcnn_activate || exit 1
cd "$ADCNN_REPO"
python ADCNN/pipelines/heliolinc/discover_stream.py --n-gpus "${DETECT_N_GPUS:-4}" \
  --manifest "$RUN/manifest.csv" --out "$RUN/adcnn_dets.csv"
echo "SN_DETECT DONE"
