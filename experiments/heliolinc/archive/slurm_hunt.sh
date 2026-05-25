#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -J hunt
#SBATCH -o %x_%j.log
set -eo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
# Trail-tracklet asteroid hunt on the Veres-measured detections (precise endpoints).
# heliolinc is single-threaded and verbose -> its live progress lands in this log.
HELIODIST=heliohypo_coarse.txt \
bash /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/hunt_new.sh \
  /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/run_wide_v2
echo "SLURM HUNT WRAPPER DONE"
