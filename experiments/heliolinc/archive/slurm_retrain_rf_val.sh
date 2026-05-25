#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere017
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0:25:00
#SBATCH -J rf_val
#SBATCH -o %x_%j.log
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd /sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
# Exactly the end-to-end stage-2 RF retrain, on the trained v7's held-out 64 val panels.
# Features now default to the corrected footprint-PCA orientation (orient_mode="pca"), so this
# rewires the RF onto the accurate beta. Overwrites the deployed RF (old one backed up).
python - <<'PY'
import pandas as pd
from ADCNN.inference.rf_train import train_rf_from_val
VAL_H5 = "DATA_DIFFIM_realistic/shard_3/train.h5"
VAL_CSV = "DATA_DIFFIM_realistic/shard_3_val.csv"
ids = sorted(pd.read_csv(VAL_CSV)["image_id"].unique())[:64]   # n_val_panels=64
train_rf_from_val("models/v7_diffim_scripted.pt", VAL_H5, VAL_CSV, ids,
                  "models/rf_postproc.pkl", neg_ratio=5)        # deployed reg2 = neg5
print("RF VAL RETRAIN DONE")
PY
