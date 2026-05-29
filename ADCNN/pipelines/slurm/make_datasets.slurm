#!/bin/bash
#SBATCH --job-name=adcnn-make-datasets
#SBATCH --account=rubin:developers
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=460G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --requeue
#
# Build the simulated (injected-trail) diffim datasets on one node:
#
#   train  (+ val)   stage-1 segmentation training   (val  -> model selection)
#   train2 (+ val2)  stage-2 cutout-CNN training      (val2 -> FP-filter threshold)
#   test             held-out evaluation
#
# A single panel universe is selected for --where, partitioned ONCE into the five disjoint sets
# (cached in <DATA_ROOT>/split.json), and each set injected with the same core. A fixed --seed makes
# the whole thing deterministic (panel selection + partition + every panel's injections).
#
# Build only some groups with SETS (the partition still covers all five and split.json is reused, so
# partial builds stay consistent with a full one):
#   SETS="train"            # train + val           SETS="train2"   # train2 + val2
#   SETS="test"             # test                  SETS="train train2 test"  (default)
#
# CPU-bound (~minutes/panel: two AlardLupton subtractions + two DIA detects). For a large train set,
# spread the build across nodes with the job array (see make_datasets_fleet.sh) or raise --time.
#
#   sbatch ADCNN/pipelines/slurm/make_datasets.slurm
#   N_TRAIN=4300 DATA_ROOT=/path SETS="train" sbatch ADCNN/pipelines/slurm/make_datasets.slurm
set -eo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"

# ---- knobs (override via env) ---------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$REPO/DATA}"
WHERE="${WHERE:-instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y')}"
SETS="${SETS:-train train2 test}"     # which groups to build now
SEED="${SEED:-2026}"
# Set sizes. train is the large stage-1 set (~0.11 GB/panel uncompressed); test is gzip'd (~10x
# smaller) and labelled at every TEST_SIGMAS in a single build. ~20% of selected panels can't be built
# (no overlapping template / too few kernel sources); ADCNN_ALLOC_HEADROOM (default 1.5) over-allocates
# to absorb that, and each set is capped at its target so the on-disk size stays bounded.
N_TRAIN="${N_TRAIN:-5000}"
N_VAL="${N_VAL:-150}"
N_TRAIN2="${N_TRAIN2:-500}"
N_VAL2="${N_VAL2:-100}"
N_TEST="${N_TEST:-300}"
SIGMA="${SIGMA:-5}"                   # stack-detection sigma (train family + val)
TEST_SIGMAS="${TEST_SIGMAS:-5 4 3}"  # label test at each of these in one build (real_labels_/stack_detection_<s>sigma)
# Skip the per-panel pre-validation when selecting the universe: it is slow, and unbuildable panels are
# simply dropped at build time anyway (absorbed by the over-allocation above). Set SKIP_PREVAL= to re-enable.
SKIP_PREVAL="${SKIP_PREVAL:-1}"
# --- job-array sharding (the orchestrator submits this as an array; see make_datasets_fleet.sh) ---
PLAN_ONLY="${PLAN_ONLY:-}"                    # =1 -> write split.json then exit (no build)
GATHER="${GATHER:-}"                          # =1 -> concat <set>.shard*.{h5,csv} -> <set>.{h5,csv}, exit
ONLY_SETS="${ONLY_SETS:-}"                    # e.g. "train" or "val" — build exactly these set names
N_SHARDS="${N_SHARDS:-1}"                     # split the built set into this many strided shards
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"  # which shard this task builds (the array task index)
PARALLEL="${PARALLEL:-${SLURM_CPUS_PER_TASK:-32}}"
# --- contamination guards (avoid real asteroids; keep the real test set disjoint) ---
MIN_ECLIPTIC_LAT="${MIN_ECLIPTIC_LAT:-20}"   # only inject into fields >20 deg off the ecliptic (no main-belt)
EXCLUDE_CSV="${EXCLUDE_CSV:-$REPO/DATA/sv_fast_movers_for_karlo_fast_with_pixels_rerun.csv}"  # real (visit,detector) to exclude

export PYTHONHASHSEED=0
# One thread per worker: the build runs one process per CPU, so nested BLAS/OpenMP/numexpr parallelism
# inside each diffim subtraction would only oversubscribe the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE
echo "[slurm] workers=${PARALLEL:-?} cpus=${SLURM_CPUS_PER_TASK:-?}"
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "$REPO"

python3 -u -m ADCNN.pipelines.make_sim_data \
  --save-path "$DATA_ROOT" --where "$WHERE" --seed "$SEED" \
  --sets $SETS \
  --n-train "$N_TRAIN" --n-val "$N_VAL" --n-train2 "$N_TRAIN2" --n-val2 "$N_VAL2" --n-test "$N_TEST" \
  --stack-detection-threshold "$SIGMA" --test-sigmas $TEST_SIGMAS \
  --min-ecliptic-lat "$MIN_ECLIPTIC_LAT" \
  --exclude-pairs-csv "$EXCLUDE_CSV" \
  --realistic-trail --mag-mode snr --mag-min 2 --mag-max 8 \
  --trail-length-min 6 --trail-length-max 60 --beta-min 0 --beta-max 180 --number 20 \
  --parallel "$PARALLEL" \
  --n-shards "$N_SHARDS" --shard "$SHARD" \
  ${ONLY_SETS:+--only-sets $ONLY_SETS} ${PLAN_ONLY:+--plan-only} ${GATHER:+--gather} \
  ${REPARTITION:+--repartition} ${SKIP_PREVAL:+--skip-prevalidation}

echo "=== DATASETS DONE -> $DATA_ROOT/<set>.{h5,csv} + split.json"
echo "    (test.h5: shared images/masks + real_labels_<s>sigma; test.csv: stack_detection_<s>sigma for s in $TEST_SIGMAS) ==="
