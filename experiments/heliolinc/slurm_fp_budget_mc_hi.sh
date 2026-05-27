#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -J fp_mc_hi
#SBATCH -o %x_%A_%a.log
#SBATCH --requeue
#SBATCH --array=0-1%2
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HL"
FPP=(75 150)
python fp_budget_mc.py --fpp "${FPP[$SLURM_ARRAY_TASK_ID]}" --nshard 16
echo "FP MC HI ${FPP[$SLURM_ARRAY_TASK_ID]} DONE"
