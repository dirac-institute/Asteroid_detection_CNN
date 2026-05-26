#!/bin/bash
# =============================================================================
# NEO RECOVERY — single end-to-end entry point.
# Reproduces the run that recovered 4 real NEOs from real Rubin DP2 data.
#
#   ./neo_recovery.sh                 # full chain: prep -> detect -> tracklets -> link -> finalize
#   RUN_NAME=run_neo_test ./neo_recovery.sh        # fresh run in a new dir
#   ./neo_recovery.sh --from tracklets             # resume from a stage (detect already done)
#   ./neo_recovery.sh --only finalize              # run a single stage
#
# Stages (each a SLURM job, chained by afterok dependencies):
#   prep      (roma, lsst) : NEO grid + targeted manifest + known.csv + neo_truth.csv
#   detect    (ampere GPU) : ADCNN+RF -> adcnn_dets.csv
#   tracklets (roma)       : make_tracklets -> pairdets.csv/pairs.txt
#   link      (roma array) : grid-parallel HelioLinC over the NEO grid -> clusters_mn/
#   finalize  (roma)       : link_refine + crossmatch -> recovered_neo.csv, classified.csv
#
# NOTE: the `link` stage is a 4-task job array. If a task dies at launch with RaisedSignal:53
# (intermittent cluster flake, no log), resubmit just that element and then run finalize:
#   sbatch --array=<k> neo_stages/link.slurm   # then: ./neo_recovery.sh --only finalize
# =============================================================================
set -eo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
S=$HL/neo_stages
source "$HL/neo_recovery_config.sh"

STAGES=(prep detect tracklets link finalize)
FROM=prep; ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM=$2; shift 2;; --only) ONLY=$2; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

submit() { # submit() <stage_script> [dependency_jobid]
  local script=$1 dep=$2
  if [ -n "$dep" ]; then sbatch --parsable --dependency=afterok:$dep "$script"
  else sbatch --parsable "$script"; fi
}

echo "=== NEO RECOVERY | RUN=$RUN_NAME | field RA[$RA0,$RA1] Dec[$DEC0,$DEC1] $DAY_START-$DAY_END ==="
echo "    grid r<$RMAX | clustrad=$CLUSTRAD | maxvel=$MAXVEL | array=${NNODE}x${NSHARD} shards"

run_one() { # run a single stage standalone (no chain)
  case $1 in
    prep)      sbatch "$S/prep.slurm";;
    detect)    sbatch "$S/detect.slurm";;
    tracklets) sbatch "$S/tracklets.slurm";;
    link)      sbatch "$S/link.slurm";;
    finalize)  sbatch "$S/finalize.slurm";;
  esac
}
if [ -n "$ONLY" ]; then echo "submitting ONLY stage: $ONLY"; run_one "$ONLY"; exit 0; fi

# build the chain starting at $FROM
started=0; dep=""
for st in "${STAGES[@]}"; do
  [ "$st" = "$FROM" ] && started=1
  [ "$started" = "1" ] || continue
  jid=$(submit "$S/$st.slurm" "$dep")
  echo "  submitted $st -> job $jid${dep:+ (afterok $dep)}"
  dep=$jid
done
echo "=== chain submitted; results land in $RUN/ (recovered_neo.csv, classified.csv, lr.csv) ==="
