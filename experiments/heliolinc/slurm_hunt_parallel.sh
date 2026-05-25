#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 96
#SBATCH --mem=256G
#SBATCH -t 3:00:00
#SBATCH -J hunt_par
#SBATCH -o %x_%j.log
set -eo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
# Grid-parallel hunt over the FULL fine hypothesis grid (109,983 pts), sharded across 96 cores.
# NEO config: minobsnights=2 (fast movers cross a single field in ~2 nights, never 3).
HELIODIST=heliohypo_all.txt NSHARD=96 MINNIGHTS=2 NPT=3 MINTIMESPAN=0.05 \
bash /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/hunt_parallel.sh \
  /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/run_wide_v2
echo "SLURM PARALLEL HUNT WRAPPER DONE"
