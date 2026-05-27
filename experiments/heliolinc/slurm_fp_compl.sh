#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 120
#SBATCH --mem=200G
#SBATCH -t 6:00:00
#SBATCH -J fp_compl
#SBATCH -o %x_%A_%a.log
#SBATCH --requeue
#SBATCH --array=0-4%3
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HL"
FPP=(0 6 32 75 150)
python fp_budget_completeness.py --fpp "${FPP[$SLURM_ARRAY_TASK_ID]}" --nshard 120 --nneo 200
echo "FP COMPL ${FPP[$SLURM_ARRAY_TASK_ID]} DONE"
