#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -J fp_budget
#SBATCH -o %x_%j.log
#SBATCH --requeue
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
cd "$HL"
# low -> high FP/visit; brackets the budget cliff. Deployed 110k-hypothesis grid (defensible).
python fp_budget_sweep.py --fpv 0 2 5 10 20 40 80 137 200
echo "FP BUDGET SWEEP COMPLETE"
