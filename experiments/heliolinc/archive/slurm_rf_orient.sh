#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 0:25:00
#SBATCH -J rf_orient
#SBATCH -o %x_%j.log
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
# A/B: corrected (PCA) orientation vs original (NN-head) orientation, trained on the SAME 64
# held-out val panels the deployed RF uses (train_end_to_end stage 2), evaluated on the synthetic
# test sets. Streaming -> bounded memory.
python experiments/heliolinc/rf_orient_compare.py --neg-ratio 5 --n-gpus 4 \
  --out-pca models/rf_postproc_pca.pkl \
  --out-nn  models/rf_postproc_nnhead.pkl
echo "RF ORIENT COMPARE DONE"
