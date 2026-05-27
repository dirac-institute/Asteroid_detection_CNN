#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 5:00:00
#SBATCH -J fpb_prep
#SBATCH -o %x_%j.log
set -eo pipefail
# Stage 1/3: generate dets + build tracklets ONCE into the run dir (shared disk). FPB_PY/FPB_FPP/FPB_EXTRA from sbatch --export.
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc"
python "$FPB_PY" --fpp "$FPB_FPP" --stage prep $FPB_EXTRA
echo "FPB PREP DONE"
