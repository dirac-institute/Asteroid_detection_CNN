#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 16
#SBATCH --mem=360G
#SBATCH -t 3:00:00
#SBATCH -J fpb_fin
#SBATCH -o %x_%j.log
set -eo pipefail
# Stage 3/3: link_refine over all 3 nodes' published shards + count/crossmatch -> result_<fpp>.csv.
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc"
python "$FPB_PY" --fpp "$FPB_FPP" --stage finalize $FPB_EXTRA
echo "FPB FINALIZE DONE"
