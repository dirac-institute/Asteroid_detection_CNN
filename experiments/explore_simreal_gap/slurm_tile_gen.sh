#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-tile-gen
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_tile_gen_%A_%a.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO"; export PYTHONPATH="$REPO:$REPO/ADCNN/data/dataset_creation:${PYTHONPATH:-}"
source /sdf/group/rubin/sw/loadLSST.bash; setup lsst_distrib
echo "=== tile-gen shard ${SLURM_ARRAY_TASK_ID}/16 === $(date -Is)"
srun python3 -u experiments/explore_simreal_gap/tile_gen.py \
  --pairs-csv "$REPO/experiments/explore_simreal_gap/candidate_pairs.csv" \
  --out "$REPO/DATA_DIFFIM_tiles" --tile 176 --n-neg 40 \
  --shard ${SLURM_ARRAY_TASK_ID} --nshards 16 --limit 150 --realistic --seed 456
echo "TILE-GEN SHARD ${SLURM_ARRAY_TASK_ID} DONE $(date -Is)"
