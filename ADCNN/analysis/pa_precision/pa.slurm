#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J pa_precision
#SBATCH -o outputs/logs/%x_%j.log
set -eo pipefail
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
source $REPO/ADCNN/pipelines/heliolinc/pipeline_config.sh; adcnn_activate || exit 1
cd "$REPO"; export PYTHONPATH="$REPO"
python -u outputs/runs/pa_precision/measure_pa.py 30
echo PA_PRECISION_DONE
