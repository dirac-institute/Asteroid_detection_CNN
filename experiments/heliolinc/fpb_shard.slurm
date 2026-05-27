#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 120
#SBATCH --mem=360G
#SBATCH -t 12:00:00
#SBATCH -J fpb_shard
#SBATCH -o %x_%A_%a.log
#SBATCH --array=0-2
set -eo pipefail
# Stage 2/3: this node links its grid third (grid[task::3]) via a bounded local pool with streaming
# progress/ETA, publishing clusters to <run>/clusters_mn. 3 nodes run concurrently.
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc"
python "$FPB_PY" --fpp "$FPB_FPP" --stage shard --node-idx "$SLURM_ARRAY_TASK_ID" --nnode 3   --ncores "${SLURM_CPUS_PER_TASK:-120}"
echo "FPB SHARD node $SLURM_ARRAY_TASK_ID DONE"
