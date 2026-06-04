#!/bin/bash
# Orchestrate the full ~100-field lambda campaign with chained SLURM arrays (kipac).
#   inject (milano)  --afterok-->  detect_ada (ada GPU)   --\
#                    \-afterok-->  stack (milano CPU)  -----+--> [manual] sweep_S + consolidate
# Usage: NFIELDS=100 RUN=/abs/run_lambda bash submit_full.sh
set -eo pipefail
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/ADCNN/pipelines/heliolinc
RUN=${RUN:?set RUN to an absolute run dir}
NFIELDS=${NFIELDS:?set NFIELDS}
NOBJ=${NOBJ:-300}
THROTTLE=${THROTTLE:-8}            # never above 8 nodes concurrently for data production
# kipac's group node cap is ~2 nodes TOTAL across partitions (counts your own running jobs), too tight for
# a 100-field run -> use rubin:commissioning (normal QOS, real headroom) to actually parallelise.
ACCT=${ACCT:-rubin:commissioning}
CPUP=${CPUP:-roma}                # CPU partition for inject+stack (roma/milano)
GPUP=${GPUP:-ada}                 # GPU partition for detection (ada normal=non-preempt; ampere=preempt)
GRES=${GRES:-gpu:l40s:4}          # match GPUP (ada=l40s, ampere=a100)
last=$((NFIELDS-1))

echo "[submit] inject array 0-$last ($CPUP, $ACCT, throttle %$THROTTLE)"
JIN=$(sbatch --parsable --partition=$CPUP --account=$ACCT --qos=normal \
  --array=0-${last}%${THROTTLE} --export=ALL,RUN=$RUN,NOBJ=$NOBJ $HL/run_lambda/inject.slurm)
echo "  inject job $JIN"

echo "[submit] detect array 0-$last ($GPUP, $ACCT, afterok inject)"
JDET=$(sbatch --parsable --partition=$GPUP --account=$ACCT --gres=$GRES \
  --dependency=aftercorr:$JIN --array=0-${last}%16 --export=ALL,RUN=$RUN $HL/run_lambda/detect_ada.slurm)
echo "  detect job $JDET"

echo "[submit] stack array 0-$last ($CPUP, $ACCT, afterok inject, throttle %$THROTTLE)"
JST=$(sbatch --parsable --partition=$CPUP --account=$ACCT --qos=normal \
  --dependency=aftercorr:$JIN --array=0-${last}%${THROTTLE} --export=ALL,RUN=$RUN $HL/run_lambda/stack.slurm)
echo "  stack job $JST"

cat <<EOF
[submit] submitted. when detect ($JDET) + stack ($JST) finish, run:
  conda activate asteroid_cnn
  python -m ADCNN.pipelines.heliolinc.sweep_S --dir $RUN
  python -m ADCNN.pipelines.heliolinc.consolidate_lambda --dir $RUN --publish
EOF
