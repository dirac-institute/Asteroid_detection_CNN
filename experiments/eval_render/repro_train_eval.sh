#!/bin/bash
# Full reg2 reproduction: train end-to-end (30 ep) then eval the reproduced model on the
# synthetic test sets, so we can compare recall/FP to the deployed reg2.
#SBATCH -J adc-repro -A kipac:kipac -p ampere --gres=gpu:1 -c 32 --mem=160G -t 06:00:00
#SBATCH --requeue --exclude=sdfampere017 -o /sdf/home/m/mrakovci/logs/ADCNN_repro_%j.out
set -euo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
[ -d "$REPO" ] || { echo "FATAL no mount on $(hostname)"; scontrol requeue "$SLURM_JOB_ID"||true; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "===== STAGE A: full end-to-end training (reg2 recipe, 30 epochs) ====="
python -m ADCNN.pipelines.train_end_to_end --run-name reg2_repro

echo "===== STAGE B: evaluate the REPRODUCED model on synthetic test sets ====="
python -m ADCNN.pipelines.make_eval_catalogs \
    --v7 models/reg2_repro_v7_scripted.pt --rf models/reg2_repro_rf_postproc.pkl \
    --out Evaluation/catalogs_repro --sets test_5sigma test_4sigma test_3sigma
echo "DEPLOYED reg2 reference:  5sigma recall=0.725  4sigma=0.744  3sigma=0.788"
echo "REPRO-TRAIN-EVAL DONE $(date -Is)"
