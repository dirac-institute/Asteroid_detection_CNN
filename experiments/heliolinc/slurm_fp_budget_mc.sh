#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -J fp_mc
#SBATCH -o %x_%A_%a.log
#SBATCH --requeue
#SBATCH --array=0-4%2
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 96 single-thread grid shards
cd "$HL"
FPP=(1 6 32 75 150)    # 1 -> LSST-stack 5sigma FP/panel level (~150), 96-way grid sharding
python fp_budget_mc.py --fpp "${FPP[$SLURM_ARRAY_TASK_ID]}" --nshard 32
echo "FP MC LEVEL ${FPP[$SLURM_ARRAY_TASK_ID]} DONE"
