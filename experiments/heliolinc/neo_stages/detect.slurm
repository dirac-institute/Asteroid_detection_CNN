#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 2:30:00
#SBATCH -J neo_detect
#SBATCH -o %x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append
set -eo pipefail
# Stage DETECT (GPU): stream the targeted diffim panels through ADCNN+RF -> sky detection catalog
# with de-biased trail endpoints. Incremental/resumable (.done log) so a preempted requeue resumes.
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
eval "$TORCH_ENV"
cd "$REPO"
python experiments/heliolinc/discover_stream.py --n-gpus "$N_GPUS" --rf-thr "$RF_THR" \
  --manifest "$RUN/manifest.csv" --out "$RUN/adcnn_dets.csv"
echo "NEO DETECT DONE -> $RUN/adcnn_dets.csv"
