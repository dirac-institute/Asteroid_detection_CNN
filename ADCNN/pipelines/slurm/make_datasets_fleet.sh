#!/bin/bash
# Fire-and-forget: build all the simulated diffim datasets across the cluster, then train + eval,
# as one SLURM DAG. EVERY set is sharded across a job array sized to ~PANELS_PER_SHARD panels, so the
# build is evenly split over the nodes with no single-node straggler holding the gate.
#
#   1. PLAN  (1 task)        : select + partition the universe -> split.json (--plan-only)
#   2. BUILD (5 arrays)      : train/val/cnn_train/cnn_val/test, each an array of ceil(size/PANELS_PER_SHARD)
#                             shards -> <set>.shard<k>.{h5,csv}, all running in parallel
#   3. GATHER (1 task)       : concat the small sets' shards -> val/cnn_train/cnn_val/test.{h5,csv}
#                             (train is read straight from its shards via --data-sources, no concat)
#   4. TRAIN (afterok 2+3)   : stage-1 on all train.shard*.h5 + stage-2 CNN on cnn_train/cnn_val
#   5. EVAL  (afterok 4)     : not-worse gate vs the deployed model + notebook plots
#
#   bash ADCNN/pipelines/slurm/make_datasets_fleet.sh
#   SKIP_TRAIN=1 bash ...   # stop after datasets are built+gathered (e.g. to review before training)
set -eo pipefail
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
cd "$REPO"
RUN_NAME="${RUN_NAME:-seg}"
SLURM="ADCNN/pipelines/slurm/make_datasets.slurm"
PANELS_PER_SHARD="${PANELS_PER_SHARD:-150}"   # ~equal shard wall-time; sets #shards per set
SKIP_TRAIN="${SKIP_TRAIN:-}"                  # =1 -> build+gather only, don't submit train/eval
# The CPU builds run on BUILD_PARTITION; BUILD_MEM well below a node's RAM lets them pack onto partially
# free nodes (each build needs only ~2 GB/worker). Train + eval are GPU jobs and keep their own partition.
BUILD_PARTITION="${BUILD_PARTITION:-milano}"
BUILD_MEM="${BUILD_MEM:-120G}"
# A short, realistic walltime is essential for concurrency: SLURM's GrpTRESRunMins budget charges
# nodes x remaining-walltime, so a multi-day --time would throttle how many shards run at once. A
# sharded build finishes in well under an hour.
BUILD_TIME="${BUILD_TIME:-02:00:00}"
BUILD_OPTS="--partition=$BUILD_PARTITION --mem=$BUILD_MEM --time=$BUILD_TIME"
# headroom 1.0 (no over-alloc) + sizes set /0.81 so each set lands ~its target after the ~19% of panels
# that have no overlapping template / too few kernel sources are skipped. Identical on plan + every build.
export N_TRAIN="${N_TRAIN:-5700}" N_VAL="${N_VAL:-185}" N_TRAIN2="${N_TRAIN2:-540}" \
       N_VAL2="${N_VAL2:-125}" N_TEST="${N_TEST:-310}" ADCNN_ALLOC_HEADROOM="${ADCNN_ALLOC_HEADROOM:-1.0}"

# shards for a set: ceil(size / PANELS_PER_SHARD), at least 2 so every set is sharded + gathered uniformly
nshards() { local n=$(( ($1 + PANELS_PER_SHARD - 1) / PANELS_PER_SHARD )); [ "$n" -lt 2 ] && n=2; echo "$n"; }

PLAN=$(sbatch --parsable --partition="$BUILD_PARTITION" --cpus-per-task=16 --mem=64G --time=02:00:00 \
       --job-name=adcnn-plan --export=ALL,PLAN_ONLY=1 "$SLURM")
echo "plan        : $PLAN  (partition=$BUILD_PARTITION, ~$PANELS_PER_SHARD panels/shard)"

# one array per set
declare -A SIZEOF=( [train]=$N_TRAIN [val]=$N_VAL [cnn_train]=$N_TRAIN2 [cnn_val]=$N_VAL2 [test]=$N_TEST )
declare -A ARR
BUILD_DEPS=""
for s in train val cnn_train cnn_val test; do
  n=$(nshards "${SIZEOF[$s]}")
  J=$(sbatch --parsable $BUILD_OPTS --dependency=afterok:$PLAN --array=0-$((n-1)) \
      --job-name=adcnn-$s --export=ALL,N_SHARDS="$n",ONLY_SETS="$s" "$SLURM")
  ARR[$s]=$J
  echo "build $s    : $J  (array 0-$((n-1)))"
  BUILD_DEPS="${BUILD_DEPS}:${J}"
done

# gather the small sets' shards into single files (train stays sharded, read via --data-sources)
GATHER=$(GATHER=1 ONLY_SETS="val cnn_train cnn_val test" sbatch --parsable \
         --dependency=afterok:${ARR[val]}:${ARR[cnn_train]}:${ARR[cnn_val]}:${ARR[test]} \
         --partition="$BUILD_PARTITION" --cpus-per-task=4 --mem=32G --time=01:00:00 \
         --job-name=adcnn-gather "$SLURM")
echo "gather      : $GATHER  (val/cnn_train/cnn_val/test shards -> single files)"

if [ -n "$SKIP_TRAIN" ]; then
  echo "$PLAN ${BUILD_DEPS#:} $GATHER" > /tmp/adcnn_fleet_jobs.txt
  echo "DAG submitted (build+gather only; SKIP_TRAIN set). plan=$PLAN builds=${BUILD_DEPS#:} gather=$GATHER"
  exit 0
fi

TRAIN=$(RUN_NAME="$RUN_NAME" sbatch --parsable --dependency=afterok:${ARR[train]}:${GATHER} \
        ADCNN/pipelines/slurm/train_end_to_end.slurm)
echo "train       : $TRAIN"

EVAL=$(RUN_NAME="$RUN_NAME" sbatch --parsable --dependency=afterok:$TRAIN \
       ADCNN/pipelines/slurm/eval_end_to_end.slurm)
echo "eval        : $EVAL"

echo "$PLAN ${BUILD_DEPS#:} $GATHER $TRAIN $EVAL" > /tmp/adcnn_fleet_jobs.txt
echo "DAG submitted. plan=$PLAN builds=${BUILD_DEPS#:} gather=$GATHER train=$TRAIN eval=$EVAL"
