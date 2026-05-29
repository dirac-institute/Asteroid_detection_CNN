#!/bin/bash
#SBATCH --job-name=adcnn-eval-e2e
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append
#
# ENTRY POINT (c) — evaluate the freshly trained two-stage detector on the held-out test sets
# (synthetic `test` AND real `test_real`), render the evaluation notebook to plots per set, and
# assert the new model is NOT WORSE than the currently deployed one. Run AFTER train_end_to_end.slurm.
#
# Each model's NN runs once per set; recall + FP/panel are sigma-independent, so the gate compares
# those two numbers (the per-sigma / single LSST-stack baseline is for the notebook plots). Each model
# is scored at ITS OWN CNN operating point (sidecar <model>.json "threshold", else CNN_DEFAULT_THR).
# Sets that are not present on disk are skipped (so this never hard-fails if test_real isn't restored).
#
# NOTE: the deployed model was trained on near-ecliptic data while `test` is off-ecliptic, so the
# synthetic gate is a strict "not worse than what we have now" check; test_real is the real target.
#
#   sbatch --dependency=afterok:<train_jobid> ADCNN/pipelines/slurm/eval_end_to_end.slurm
#   RUN_NAME=seg_v2 SETS="test test_real" sbatch ADCNN/pipelines/slurm/eval_end_to_end.slurm
set -eo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA_ROOT="${DATA_ROOT:-$REPO/DATA}"
RUN_NAME="${RUN_NAME:-seg_v2}"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"
SETS="${SETS:-test test_real}"
SIGMAS="${SIGMAS:-5,4,3}"
REC_TOL="${REC_TOL:-0.01}"          # PASS if new recall  >= deployed recall - REC_TOL
FP_TOL="${FP_TOL:-0.15}"            # PASS if new FP/panel <= deployed FP/panel * (1 + FP_TOL)

NEW_SEG="$MODELS_DIR/${RUN_NAME}_segmentation_scripted.pt"
NEW_CNN="$MODELS_DIR/${RUN_NAME}_cnn_postproc.pt"
DEP_SEG="$MODELS_DIR/segmentation_model.pt"
DEP_CNN="$MODELS_DIR/cnn_postproc.pt"
NEW_CAT="$REPO/Evaluation/catalogs_${RUN_NAME}"
DEP_CAT="$REPO/Evaluation/catalogs_deployed"

export PYTHONHASHSEED=0 MPLBACKEND=Agg
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"

for f in "$NEW_SEG" "$NEW_CNN" "$DEP_SEG" "$DEP_CNN"; do
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

# Each model's CNN operating point: sidecar <model>.json "threshold", else module default.
cnn_thr() {  # $1 = cnn .pt path
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
from ADCNN.inference.cnn_postproc import CNN_DEFAULT_THR
sc = Path(sys.argv[1]).with_suffix(".json")
print(json.loads(sc.read_text()).get("threshold", CNN_DEFAULT_THR) if sc.exists() else CNN_DEFAULT_THR)
PY
}
NEW_THR="$(cnn_thr "$NEW_CNN")"; DEP_THR="$(cnn_thr "$DEP_CNN")"
echo "thresholds: NEW=$NEW_THR  DEPLOYED=$DEP_THR"

echo "=== [1/3] inference: NEW ($RUN_NAME) + DEPLOYED on ${PRESENT[*]} ==="
python3 -u -m ADCNN.pipelines.make_eval_catalogs --sets "${PRESENT[@]}" \
  --data-root "$DATA_ROOT" --seg-model "$NEW_SEG" --cnn "$NEW_CNN" --cnn-thr "$NEW_THR" --out "$NEW_CAT"
python3 -u -m ADCNN.pipelines.make_eval_catalogs --sets "${PRESENT[@]}" \
  --data-root "$DATA_ROOT" --seg-model "$DEP_SEG" --cnn "$DEP_CNN" --cnn-thr "$DEP_THR" --out "$DEP_CAT"

echo "=== [2/3] render notebook (NEW model) per set -> plots ==="
if command -v jupyter >/dev/null 2>&1; then
  for s in "${PRESENT[@]}"; do
    ADCNN_EVAL_CATALOGS="$NEW_CAT" ADCNN_EVAL_DATA="$DATA_ROOT" ADCNN_EVAL_TEST="$s" \
    ADCNN_EVAL_SIGMAS="$SIGMAS" ADCNN_EVAL_SEG="$NEW_SEG" ADCNN_EVAL_CNN="$NEW_CNN" \
      jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=2400 \
        --output "Evaluation_${RUN_NAME}_${s}" --output-dir "$REPO/Evaluation" \
        "$REPO/Evaluation/Evaluation.ipynb" \
      && echo "  plots -> Evaluation/Evaluation_${RUN_NAME}_${s}.html" \
      || echo "  WARNING: nbconvert failed for $s; continuing"
  done
else
  echo "  WARNING: jupyter not found; skipping notebook render"
fi

echo "=== [3/3] NOT-WORSE GATE (new vs deployed, per set) ==="
GATE_FAIL=0
for s in "${PRESENT[@]}"; do
  python3 - "$NEW_CAT/${s}_metrics.json" "$DEP_CAT/${s}_metrics.json" "$REC_TOL" "$FP_TOL" "$s" <<'PY' || GATE_FAIL=1
import json, sys
new=json.load(open(sys.argv[1])); dep=json.load(open(sys.argv[2]))
rec_tol=float(sys.argv[3]); fp_tol=float(sys.argv[4]); name=sys.argv[5]
rn,rd=new["recall"],dep["recall"]; fn,fd=new["fp_per_panel"],dep["fp_per_panel"]
print(f"[{name}] NEW      recall={rn:.4f} FP/panel={fn:.3f}  (TP={new['TP']} FP={new['FP']} FN={new['FN']})")
print(f"[{name}] DEPLOYED recall={rd:.4f} FP/panel={fd:.3f}  (TP={dep['TP']} FP={dep['FP']} FN={dep['FN']})")
rec_ok=rn>=rd-rec_tol; fp_ok=fn<=fd*(1+fp_tol)+1e-9
print(f"[{name}] recall not-worse(>= {rd-rec_tol:.4f}): {'PASS' if rec_ok else 'FAIL'} | "
      f"FP/panel not-worse(<= {fd*(1+fp_tol):.3f}): {'PASS' if fp_ok else 'FAIL'}")
sys.exit(0 if (rec_ok and fp_ok) else 2)
PY
done
echo "=== GATE: $([[ $GATE_FAIL -eq 0 ]] && echo 'PASS (all sets) ===' || echo 'FAIL (a set is worse) ===')"
echo "=== EVAL DONE ==="
[[ $GATE_FAIL -eq 0 ]]
