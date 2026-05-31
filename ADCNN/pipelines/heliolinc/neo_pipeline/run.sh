#!/bin/bash
# =============================================================================
# NEO trail-tracklet discovery — END-TO-END launcher (one command, large data)
# =============================================================================
#   detect (GPU) -> measure (CPU) -> clean+link+crossmatch (CPU)
# chained via SLURM dependencies so each stage starts when the previous succeeds.
# Each stage is internally parallel (multi-GPU / multi-core / grid-sharded) for speed.
#
#   ./run.sh                       # full chain on the configured RUN_NAME/MANIFEST
#   RUN_NAME=run_aug MANIFEST=/path/manifest.csv ./run.sh
#   ./run.sh --from 2              # resume from stage 2 (e.g. detections already exist)
#   ./run.sh --only 3              # just re-run link+crossmatch (e.g. retune linking)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
S="$HERE/stages"

FROM=1; TO=3; ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM=$2; shift 2;; --to) TO=$2; shift 2;; --only) ONLY=$2; FROM=$2; TO=$2; shift 2;;
  *) echo "unknown arg $1"; exit 1;; esac; done

cp -n "$KNOWN" "$RUN/known.csv" 2>/dev/null || true   # for crossmatch
echo "=== NEO pipeline | RUN=$RUN_NAME | manifest=$(wc -l < "$MANIFEST") panels | stages $FROM..$TO ==="

dep=""; submit() {  # submit <stagefile>; chains afterok on the previous job
  local jid
  if [ -n "$dep" ]; then jid=$(sbatch --parsable --dependency=afterok:$dep "$1")
  else jid=$(sbatch --parsable "$1"); fi
  echo "  submitted $(basename "$1") -> job $jid${dep:+ (afterok:$dep)}"; dep=$jid
}

[ "$FROM" -le 1 ] && [ "$TO" -ge 1 ] && submit "$S/01_detect.slurm"
[ "$FROM" -le 2 ] && [ "$TO" -ge 2 ] && submit "$S/02_measure.slurm"
[ "$FROM" -le 3 ] && [ "$TO" -ge 3 ] && submit "$S/03_link.slurm"
echo "=== chain submitted; final results land in $RUN/ (confirmed.csv, new_candidates.csv) ==="
echo "    watch: tail -f $HL/neo3_link_*.log   |  squeue -u \$USER"
