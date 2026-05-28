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
# ENTRY POINT (a) — generate ALL the simulated diffim datasets in ONE deterministic command:
#
#   train  (+ val)   stage-1 segmentation training   (val  -> model selection)
#   train2 (+ val2)  stage-2 cutout-CNN training      (val2 -> FP-filter threshold)
#   test             held-out evaluation
#
# A single panel universe is selected for --where, partitioned ONCE into the five mutually
# disjoint sets (cached in <DATA_ROOT>/split.json), and each set injected with the same
# validated core. DETERMINISTIC: a fixed --seed fixes panel selection + partition + every
# panel's injections, so every rerun selects the same panels and injects identical trails
# (a panel that fails to subtract is dropped; failures are repeatable).
#
# LEVERS — build only some groups via SETS (the partition still covers all five, so partial /
# incremental builds stay consistent with a full build, and split.json is reused):
#   SETS="test"            sbatch ...   # only test
#   SETS="train"           sbatch ...   # only train + val
#   SETS="train2"          sbatch ...   # only train2 + val2
#   SETS="train train2 test"            # everything (default)
#
# lsst_distrib env (Butler). CPU-bound (~minutes/panel: 2 AlardLupton subtractions + 2 DIA
# detects each); the universe = sum of all --n-* sizes. For very large train sets, build the
# groups across separate jobs (SETS=...) — split.json keeps them consistent — or raise --time.
#
#   sbatch ADCNN/pipelines/slurm/make_datasets.slurm
#   N_TRAIN=4300 DATA_ROOT=/path SETS="train" sbatch ADCNN/pipelines/slurm/make_datasets.slurm
set -euo pipefail

REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"

# ---- deterministic knobs (override via env) -------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$REPO/DATA_DIFFIM}"
WHERE="${WHERE:-instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y')}"
SETS="${SETS:-train train2 test}"     # which groups to build now
SEED="${SEED:-2026}"
N_TRAIN="${N_TRAIN:-1500}"
N_VAL="${N_VAL:-150}"
N_TRAIN2="${N_TRAIN2:-500}"
N_VAL2="${N_VAL2:-100}"
N_TEST="${N_TEST:-300}"
SIGMA="${SIGMA:-5}"                    # stack-detection sigma (train family + val)
TEST_SIGMA="${TEST_SIGMA:-$SIGMA}"     # test depth (rebuild test at another sigma with SETS=test)
PARALLEL="${PARALLEL:-${SLURM_CPUS_PER_TASK:-32}}"

export PYTHONHASHSEED=0
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "$REPO"

python3 -u -m ADCNN.pipelines.make_sim_data \
  --save-path "$DATA_ROOT" --where "$WHERE" --seed "$SEED" \
  --sets $SETS \
  --n-train "$N_TRAIN" --n-val "$N_VAL" --n-train2 "$N_TRAIN2" --n-val2 "$N_VAL2" --n-test "$N_TEST" \
  --stack-detection-threshold "$SIGMA" --test-sigma "$TEST_SIGMA" \
  --realistic-trail --mag-mode snr --mag-min 2 --mag-max 8 \
  --trail-length-min 6 --trail-length-max 60 --beta-min 0 --beta-max 180 --number 20 \
  --parallel "$PARALLEL" \
  ${REPARTITION:+--repartition} ${SKIP_PREVAL:+--skip-prevalidation}

echo "=== DATASETS DONE -> $DATA_ROOT/{train,val,train2,val2,test}.{h5,csv} + split.json ==="
