#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 120
#SBATCH --mem=200G
#SBATCH -t 6:00:00
#SBATCH -J fp_mc_r
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
python fp_budget_mc.py --fpp "${FPP[$SLURM_ARRAY_TASK_ID]}" --nshard 120
echo "FP MC ROMA ${FPP[$SLURM_ARRAY_TASK_ID]} DONE"
