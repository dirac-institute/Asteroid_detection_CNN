#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 120
#SBATCH --mem=360G
#SBATCH -t 16:00:00
#SBATCH -J fp_mc_2800
#SBATCH -o %x_%j.log
#SBATCH --requeue
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HL"
NSHARD=${SLURM_CPUS_PER_TASK:-$(nproc)}; echo "allocated $NSHARD cores -> $NSHARD grid shards"
python fp_budget_mc.py --fpp 2800 --nshard "$NSHARD"
echo "FP MC 2800 DONE"
