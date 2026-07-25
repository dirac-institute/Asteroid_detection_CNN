#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -J probe_seg
#SBATCH -o outputs/logs/%x_%j.log
set -eo pipefail
# Pixel-level v2_D miss audit (SLOWBAND_AUDIT.md follow-up): seg response to injected trails
# on LIVE DM-53195 diffims. Submit from repo root (relative -o path).
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
source ADCNN/pipelines/heliolinc/pipeline_config.sh   # S3 env for embargo manifests (harmless for posix)
python ADCNN/pipelines/heliolinc/probe_seg_response.py \
  --panels "${PANELS:-150}" --trails-per-panel "${TRAILS:-24}" \
  ${RUN:+--run "$RUN"} \
  --out "${OUT:-outputs/runs/run_probe_seg}"
echo "PROBE SLURM DONE"
