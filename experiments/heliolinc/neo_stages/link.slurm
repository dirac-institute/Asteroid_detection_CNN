#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 96
#SBATCH --mem=200G
#SBATCH -t 2:00:00
#SBATCH -J neo_link
#SBATCH -o %x_%A_%a.log
#SBATCH --array=0-3
set -eo pipefail
# Stage LINK (4-task job array, one node each): grid-parallel HelioLinC over the NEO grid, reusing the
# tracklets. Each task processes a round-robin 1/NNODE slice and publishes clusters to shared disk.
# A job ARRAY (not multi-node srun, which gets RaisedSignal:53 on this cluster) gives ~4x speedup.
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
[ "$SLURM_ARRAY_TASK_ID" = "0" ] && { rm -rf "$RUN/clusters_mn"; mkdir -p "$RUN/clusters_mn"; }
sleep 5   # let task 0 create the shared dir
NODE_IDX=$SLURM_ARRAY_TASK_ID NNODE=$NNODE NSHARD=$NSHARD CLUSTRAD=$CLUSTRAD \
NPT=$NPT MINNIGHTS=$MINNIGHTS MINTIMESPAN=$MINTIMESPAN \
  bash "$HL/node_worker_neo.sh"
