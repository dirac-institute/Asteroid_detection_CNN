#!/bin/bash
#SBATCH --job-name=adc-stream-test
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_stream_test_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; source /sdf/group/rubin/sw/loadLSST.bash; setup lsst_distrib
export PYTHONPATH="$REPO:$REPO/ADCNN/data/dataset_creation:${PYTHONPATH:-}"
echo "=== stream producer throughput test === $(date -Is)"
srun python3 -u experiments/explore_simreal_gap/stream_producer.py \
  --buffer-dir /sdf/scratch/users/m/mrakovci/stream_test \
  --pairs-csv "$REPO/DATA_DIFFIM_realistic_big/train.csv" \
  --buffer 40 --parallel 16 --max-panels 16 --realistic
echo "STREAM TEST DONE $(date -Is)"
