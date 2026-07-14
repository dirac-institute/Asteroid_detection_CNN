#!/bin/bash
#SBATCH --job-name=adcnn-train-e2e
#SBATCH --account=rubin:commissioning
#SBATCH --partition=ada
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
#SBATCH --exclude=sdfada009,sdfada012
#SBATCH --requeue
#SBATCH --open-mode=append
#
# ENTRY POINT (b) — train the two-stage detector end to end.
#
#   stage 1: segmentation model (UNetResSEOrientHough) trained on TRAIN, model
#            selection on the held-out VAL set, then TorchScript export.
#   stage 2: focal-loss cutout-CNN FP filter trained on TRAIN2 with DataParallel across the
#            4 GPUs, best-val-AUC checkpoint saved, operating threshold set by the COMBINED
#            5sigma+ADCNN FP-budget on VAL2.
#   TEST is never touched here (held out for evaluation).
#
# Each set is used for exactly its purpose (all disjoint by construction — see
# make_datasets.slurm): TRAIN -> seg weights, VAL -> stage-1 selection, TRAIN2 -> CNN filter,
# VAL2 -> CNN operating threshold + AUC, TEST -> eval.
#
# DETERMINISM: the training recipe pins seed=2026; the stage-2 CNN trainer is seeded too. CUDA
# reductions are deliberately NOT bit-forced (that would change the training recipe's numerics).
#
#   sbatch ADCNN/pipelines/slurm/train_end_to_end.slurm
#   RUN_NAME=seg DATA_ROOT=/path sbatch ADCNN/pipelines/slurm/train_end_to_end.slurm
set -eo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
DATA_ROOT="${DATA_ROOT:-$REPO/DATA}"
RUN_NAME="${RUN_NAME:-seg}"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic cuBLAS reductions where supported
export MPLBACKEND=Agg
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"

# TRAIN may be multi-node SHARDED (train.shard*.h5) or single train.h5. val/cnn_train/cnn_val are
# single files. Stage 1 trains on all train shards; stage 2's CNN trains on cnn_train and sets its
# threshold on cnn_val.
TRAIN_SRCS=()
for f in "$DATA_ROOT"/train.shard*.h5; do [[ -s "$f" ]] && TRAIN_SRCS+=("${f}:${f%.h5}.csv"); done
[[ ${#TRAIN_SRCS[@]} -eq 0 && -s "$DATA_ROOT/train.h5" ]] && TRAIN_SRCS+=("$DATA_ROOT/train.h5:$DATA_ROOT/train.csv")
[[ ${#TRAIN_SRCS[@]} -gt 0 ]] || { echo "MISSING train shards / train.h5 in $DATA_ROOT"; exit 1; }
echo "train sources (${#TRAIN_SRCS[@]}): ${TRAIN_SRCS[*]}"
for f in val.h5 val.csv cnn_train.h5 cnn_train.csv cnn_val.h5 cnn_val.csv; do
  [[ -s "$DATA_ROOT/$f" ]] || { echo "MISSING $DATA_ROOT/$f — run make_datasets.slurm first"; exit 1; }
done

python3 -u -m ADCNN.pipelines.train_end_to_end \
  --run-name "$RUN_NAME" \
  --data-sources "${TRAIN_SRCS[@]}" \
  --val-h5  "$DATA_ROOT/val.h5"  --val-csv  "$DATA_ROOT/val.csv" \
  --cnn-train-h5 "$DATA_ROOT/cnn_train.h5" --cnn-train-csv "$DATA_ROOT/cnn_train.csv" \
  --cnn_val-h5 "$DATA_ROOT/cnn_val.h5" --cnn_val-csv "$DATA_ROOT/cnn_val.csv" \
  --gpus 4 \
  --out-root "${ADCNN_OUTPUTS:-$REPO/outputs}/training_runs/diffim_runs" --models-dir "$MODELS_DIR"

echo "=== TRAINING DONE ==="
echo "  segmentation: $MODELS_DIR/${RUN_NAME}_segmentation_scripted.pt"
echo "  cnn filter  : $MODELS_DIR/${RUN_NAME}_cnn_postproc.pt"
