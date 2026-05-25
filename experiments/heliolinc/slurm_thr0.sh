#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:a100:4
#SBATCH --exclude=sdfampere017
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 1:30:00
#SBATCH -J adcnn_thr0
#SBATCH -o %x_%j.log
set -euo pipefail
[ -d /sdf/data/rubin ] || { echo "node lacks /sdf/data/rubin mount"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/experiments/heliolinc
cd "$REPO"
echo "=== synthetic test sets @ rf_thr=0 (uniform distributions -> unbiased recall vs threshold) ==="
python -m ADCNN.pipelines.make_eval_catalogs --sets test_5sigma test_4sigma test_3sigma \
  --rf-thr 0.0 --gate-pmax 0.10 --out "$REPO/Evaluation/catalogs_thr0"
echo "=== discovery window (test_real subset) @ rf_thr=0 ==="
python -m ADCNN.inference.catalog --h5 "$REPO/DATA_DIFFIM/test_real/test.h5" \
  --panels "$HL/window_panels.csv" --panel-ids "$HL/window_panels.csv" \
  --rf-thr 0.0 --gate-pmax 0.10 --n-gpus 4 --out "$HL/run_disco/catalog.csv"
echo "THR0 JOB DONE"
