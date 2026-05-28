#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 96
#SBATCH --mem=200G
#SBATCH -t 2:00:00
#SBATCH -J hl_neo_arr
#SBATCH -o %x_%A_%a.log
#SBATCH --array=0-3
set -eo pipefail
# Array task k of 4: process grid lines where (line %% 4 == k) with 90 local heliolinc shards, publish
# clusters to shared disk. Independent single-node jobs (robust) -> ~4x faster than one node, same math.
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
[ "$SLURM_ARRAY_TASK_ID" = "0" ] && { rm -rf "$HL/NEO_large/clusters_mn"; mkdir -p "$HL/NEO_large/clusters_mn"; }
sleep 3   # let task 0 create the shared dir
NODE_IDX=$SLURM_ARRAY_TASK_ID NNODE=4 NSHARD=90 CLUSTRAD=${CLUSTRAD:-100000} NPT=3 MINNIGHTS=2 MINTIMESPAN=0.05 \
  bash "$HL/node_worker_neo.sh"
