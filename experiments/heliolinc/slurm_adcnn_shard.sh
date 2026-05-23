#!/bin/bash
#SBATCH --job-name=adc-cand-sh
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_cand_sh%a_%A.out
#SBATCH --array=0-3
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"; cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
srun python3 -u experiments/heliolinc/adcnn_candidates.py --panels-csv experiments/heliolinc/window_panels.csv \
  --rf-thr 0.5 --shard ${SLURM_ARRAY_TASK_ID} --nshards 4 \
  --out experiments/heliolinc/run_adcnn/cand_shard${SLURM_ARRAY_TASK_ID}.parquet
echo DONE
