#!/bin/bash
#SBATCH --job-name=adcnn-make-datasets
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=3G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --requeue
#
# ENTRY POINT (a) — generate the TRAIN / TRAIN2 / TEST diffim datasets, deterministically.
#
#   train   stage-1 segmentation-model training data
#   train2  stage-2 focal cutout-CNN training data (the dedicated FP-filter set)
#   test    held-out evaluation set (synthetic, at a fixed stack-detection sigma)
#
# DETERMINISM (rerun -> identical datasets, given the same pinned DP2 Butler release):
#   * Fixed per-set SEEDS + a fixed --where region + the Butler collections pinned in
#     ADCNN.data.dataset_creation.simulate. Panel selection is a sorted-then-seeded shuffle;
#     each panel's injections are seeded by (seed, visit, detector); realistic-trail morphology
#     is seeded per injection. No global/unseeded RNG is used.
#   * The three sets are DISJOINT BY CONSTRUCTION: test is built first, train2 excludes the test
#     panels, train excludes test+train2 (--exclude-pairs-csv leakage guard). No (visit,detector)
#     panel is ever in two sets, so the stage-1 / stage-2 / eval data never overlap.
#
# Resumable: a set whose output h5 already exists is skipped (set FORCE=1 to rebuild all).
#
# Runs in the lsst_distrib env (Butler). CPU-bound (~minutes/panel: 2 AlardLupton subtractions
# + 2 DIA detects each). The reg2-scale train set (~4300 panels) was built as several shards
# across jobs; for that scale, run this per shard (distinct --seed + day_obs window) or raise
# N_TRAIN with a longer --time.
#
# Usage:
#   sbatch ADCNN/pipelines/slurm/make_datasets.slurm
#   N_TRAIN=4300 DATA_ROOT=/path/DATA_DIFFIM sbatch ADCNN/pipelines/slurm/make_datasets.slurm
set -euo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"

# ---- deterministic knobs (override via env) -------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$REPO/DATA_DIFFIM}"
WHERE="${WHERE:-instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y')}"
N_TRAIN="${N_TRAIN:-1500}"        # stage-1 segmentation training panels
N_TRAIN2="${N_TRAIN2:-500}"       # stage-2 cutout-CNN training panels (dedicated FP-filter set)
N_TEST_SEL="${N_TEST_SEL:-340}"   # ~90% land in test (test-only split) -> ~300 test panels
SIGMA="${SIGMA:-5}"               # test stack-detection threshold (test_5sigma = canonical eval set)
SEED_TRAIN="${SEED_TRAIN:-123}"   # fixed seeds -> reproducible panel selection + injections
SEED_TRAIN2="${SEED_TRAIN2:-4242}"
SEED_TEST="${SEED_TEST:-905}"
PARALLEL="${PARALLEL:-${SLURM_CPUS_PER_TASK:-32}}"

# common injection recipe — identical across the three sets; only seed / size / role differ.
COMMON=(--repo dp2_prep --skymap lsst_cells_v2 --realistic-trail --skip-prevalidation
        --mag-mode snr --mag-min 2 --mag-max 8
        --trail-length-min 6 --trail-length-max 60 --beta-min 0 --beta-max 180
        --number 20 --parallel "$PARALLEL" --where "$WHERE")

export PYTHONHASHSEED=0
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "$REPO"
mkdir -p "$DATA_ROOT"

run_set() {  # $1 = output h5 to test for existence; rest = the make_sim_data command
  local out_h5="$1"; shift
  if [[ -s "$out_h5" && "${FORCE:-0}" != "1" ]]; then
    echo ">>> skip (exists): $out_h5  (FORCE=1 to rebuild)"; return
  fi
  "$@"
}

echo "=== [1/3] TEST   seed=$SEED_TEST sigma=$SIGMA -> $DATA_ROOT/test ==="
run_set "$DATA_ROOT/test/test.h5" \
  python3 -u -m ADCNN.pipelines.make_sim_data "${COMMON[@]}" \
    --save-path "$DATA_ROOT/test" --seed "$SEED_TEST" \
    --random-subset "$N_TEST_SEL" --test-only --train-test-split 0.1 \
    --stack-detection-threshold "$SIGMA"

echo "=== [2/3] TRAIN2 seed=$SEED_TRAIN2 (excl test) -> $DATA_ROOT/train2 ==="
run_set "$DATA_ROOT/train2/train.h5" \
  python3 -u -m ADCNN.pipelines.make_sim_data "${COMMON[@]}" \
    --save-path "$DATA_ROOT/train2" --seed "$SEED_TRAIN2" \
    --random-subset "$N_TRAIN2" --train-test-split 0 \
    --exclude-pairs-csv "$DATA_ROOT/test/test.csv"

echo "=== [3/3] TRAIN  seed=$SEED_TRAIN (excl test+train2) -> $DATA_ROOT/train ==="
run_set "$DATA_ROOT/train/train.h5" \
  python3 -u -m ADCNN.pipelines.make_sim_data "${COMMON[@]}" \
    --save-path "$DATA_ROOT/train" --seed "$SEED_TRAIN" \
    --random-subset "$N_TRAIN" --train-test-split 0 \
    --exclude-pairs-csv "$DATA_ROOT/test/test.csv" "$DATA_ROOT/train2/train.csv"

echo "=== DATASETS DONE ==="
echo "  train : $DATA_ROOT/train/train.{h5,csv}    (stage-1 segmentation)"
echo "  train2: $DATA_ROOT/train2/train.{h5,csv}   (stage-2 cutout CNN)"
echo "  test  : $DATA_ROOT/test/test.{h5,csv}      (held-out eval)"
