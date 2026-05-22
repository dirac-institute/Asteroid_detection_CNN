#!/bin/bash
#SBATCH --job-name=adc-stream-prod
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_stream_prod_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; source /sdf/group/rubin/sw/loadLSST.bash; setup lsst_distrib
export PYTHONPATH="$REPO:$REPO/ADCNN/data/dataset_creation:${PYTHONPATH:-}"
echo "=== stream producer (continuous) === $(date -Is)"
srun python3 -u experiments/explore_simreal_gap/stream_producer.py \
  --buffer-dir "$REPO/DATA_DIFFIM_stream_buffer" \
  --pairs-csv "$REPO/experiments/explore_simreal_gap/validated_pairs.csv" \
  --buffer 250 --parallel 64 --max-panels 0 --realistic
