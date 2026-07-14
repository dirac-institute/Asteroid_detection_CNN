#!/bin/bash
# ADCNN v2 Phase-3 eval ladder, step 1: export a trained variant + submit its dev detection array.
# Usage: bash eval_variant.sh <run-name>      (e.g. v2_B)
# Post-detection scoring (miss audit, pair tables, reduce) runs from the wakeup chain once the
# array drains — see ADCNN_V2_SPRINT.md Phase 3. Thresholds frozen; blind set untouched.
set -euo pipefail
NAME=${1:?usage: eval_variant.sh <run-name>}
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
HL=$REPO/ADCNN/pipelines/heliolinc                     # tracked scripts
RUNS="${ADCNN_OUTPUTS:-$REPO/outputs}/runs"            # run data (outputs/ layout)
BEST="${ADCNN_OUTPUTS:-$REPO/outputs}/training_runs/diffim_runs/$NAME/ckpts/best.pt"
SCRIPTED=$RUNS/run_ft/${NAME}_segmentation_scripted.pt
OUT=$RUNS/run_dev/$NAME
[ -f "$BEST" ] || { echo "no best.pt for $NAME"; exit 1; }
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
python -m ADCNN.inference.export --ckpt "$BEST" --out "$SCRIPTED" --no-optimize
echo "[eval] scripted -> $SCRIPTED"
mkdir -p "$OUT"
for k in $(seq 0 20); do
  for pre in inject truth retime manifest; do
    ln -sf "../${pre}_$k.csv" "$OUT/${pre}_$k.csv"
  done
done
cp "$HL/run_dev/reduce_dev.py" "$OUT/reduce_dev.py"
cp "$HL/run_dev/miss_audit.py" "$OUT/miss_audit.py"
cd "$REPO"   # submit from repo root: slurm -o outputs/logs/ resolves relative to CWD
sbatch --exclude=sdfada006 --export=ALL,SEGMODEL=$SCRIPTED,OUTDIR=$OUT -J det_$NAME --array=0-20 "$HL/run_ft/detect_variant.slurm"
echo "EVAL_VARIANT_SUBMITTED $NAME"
