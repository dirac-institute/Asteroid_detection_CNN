#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 16:00:00
#SBATCH -J fp_mc
#SBATCH -o %x_%A_%a.log
#SBATCH --requeue
#SBATCH --array=0-2%3
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "$HL"
FPP=(2 4 12)        # crossing-region low points (full grid)
python fp_budget_mc.py --fpp "${FPP[$SLURM_ARRAY_TASK_ID]}"
echo "FP MC LEVEL ${FPP[$SLURM_ARRAY_TASK_ID]} DONE"
