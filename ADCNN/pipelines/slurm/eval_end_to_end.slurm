#!/bin/bash
#SBATCH --job-name=adcnn-eval-e2e
#SBATCH --account=rubin:commissioning
#SBATCH --partition=ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
#SBATCH --exclude=sdfada009,sdfada012
#SBATCH --requeue
#SBATCH --open-mode=append
#
# ENTRY POINT (c) — evaluate the shipped two-stage detector on the held-out test set(s)
# (synthetic `test` and, when present, real `test_real`) and render the evaluation notebook
# to HTML. Runs on 4 L40S of an ada node; the catalog build is sharded round-robin across them
# by ``build_detection_catalog_multigpu``.
#
# The detector is scored at its CALIBRATED operating point — the threshold in
# ``models/<RUN_NAME>_cnn_postproc.json``, set by ``train_end_to_end`` to make the deduplicated
# 5sigma-stack ∪ ADCNN union hit ``FPP_BUDGET`` FP/panel on cnn_val.
#
#   sbatch --dependency=afterok:<train_jobid> ADCNN/pipelines/slurm/eval_end_to_end.slurm
#   RUN_NAME=seg SETS="test test_real" sbatch ADCNN/pipelines/slurm/eval_end_to_end.slurm
set -eo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA_ROOT="${DATA_ROOT:-$REPO/DATA}"
RUN_NAME="${RUN_NAME:-seg}"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
SETS="${SETS:-test test_real}"
SIGMAS="${SIGMAS:-5,4,3}"

SEG="$MODELS_DIR/${RUN_NAME}_segmentation_scripted.pt"
CNN="$MODELS_DIR/${RUN_NAME}_cnn_postproc.pt"
CAT="$REPO/Evaluation/catalogs_${RUN_NAME}"

export PYTHONHASHSEED=0 MPLBACKEND=Agg
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"

for f in "$SEG" "$CNN"; do
  [[ -s "$f" ]] || { echo "MISSING $f"; exit 1; }
done

# Present sets (flat <root>/<set>.h5 or subdir <root>/<set>/test.h5).
PRESENT=()
for s in $SETS; do
  if [[ -s "$DATA_ROOT/$s.h5" || -s "$DATA_ROOT/$s/test.h5" ]]; then PRESENT+=("$s");
  else echo "[skip] $s not on disk under $DATA_ROOT"; fi
done
[[ ${#PRESENT[@]} -gt 0 ]] || { echo "no eval sets present under $DATA_ROOT"; exit 1; }
echo "eval sets: ${PRESENT[*]}"

# Calibrated CNN operating point from the sidecar (set by train_end_to_end / calibrate_combined_threshold).
THR="$(python3 - "$CNN" <<'PY'
import json, sys
from pathlib import Path
from ADCNN.inference.cnn_postproc import CNN_DEFAULT_THR
sc = Path(sys.argv[1]).with_suffix(".json")
print(json.loads(sc.read_text()).get("threshold", CNN_DEFAULT_THR) if sc.exists() else CNN_DEFAULT_THR)
PY
)"
echo "operating threshold: $THR (from sidecar)"

echo "=== [1/2] inference: build detection catalog on ${PRESENT[*]} ==="
python3 -u -m ADCNN.pipelines.make_eval_catalogs --sets "${PRESENT[@]}" \
  --data-root "$DATA_ROOT" --seg-model "$SEG" --cnn "$CNN" --cnn-thr "$THR" --out "$CAT"

echo "=== [2/2] render evaluation notebook per set -> HTML ==="
if command -v jupyter >/dev/null 2>&1; then
  for s in "${PRESENT[@]}"; do
    ADCNN_EVAL_CATALOGS="$CAT" ADCNN_EVAL_DATA="$DATA_ROOT" ADCNN_EVAL_TEST="$s" \
    ADCNN_EVAL_SIGMAS="$SIGMAS" ADCNN_EVAL_SEG="$SEG" ADCNN_EVAL_CNN="$CNN" \
      jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=2400 \
        --output "Evaluation_${RUN_NAME}_${s}" --output-dir "$REPO/Evaluation" \
        "$REPO/Evaluation/Evaluation.ipynb" \
      && echo "  plots -> Evaluation/Evaluation_${RUN_NAME}_${s}.html" \
      || echo "  WARNING: nbconvert failed for $s; continuing"
  done
else
  echo "  WARNING: jupyter not found; skipping notebook render"
fi

echo "=== EVAL DONE ==="
