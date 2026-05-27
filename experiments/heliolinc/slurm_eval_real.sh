#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -J adcnn_eval_real
#SBATCH -o %x_%j.log
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
python -m ADCNN.pipelines.make_eval_catalogs --sets test_real --n-gpus 4   # CNN@0.63, progress heartbeat
echo "MAKE_EVAL_REAL DONE"
