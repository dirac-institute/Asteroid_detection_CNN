#!/bin/bash
# run_train_with_probes.sh
#
# Launch your training (DDP/torchrun via srun), then automatically run
# GPU/CPU/IO/DDP-straggler probes while it trains.
#
# Works best INSIDE an allocated SLURM job (sbatch or salloc) on the compute node.
#
# Usage (inside allocation):
#   bash run_train_with_probes.sh
#
# Optional overrides:
#   DURATION=300 INTERVAL=2 bash run_train_with_probes.sh
#   TRAIN_CMD="srun ... torchrun ... train.py ..." bash run_train_with_probes.sh
#
# Outputs:
#   creates a timestamped directory: probe_<host>_job<id>_<ts>/
#   contains all logs + a SUMMARY.txt

set -euo pipefail

nvidia-smi

# ----------------------------
# User-tunable parameters
# ----------------------------
: "${DURATION:=300}"     # seconds of probing
: "${INTERVAL:=2}"       # seconds between samples

# If TRAIN_CMD is not set, define a reasonable default that you can edit.
# IMPORTANT: adjust paths/args to match your setup.
if [[ -z "${TRAIN_CMD:-}" ]]; then
  # Example: run baseline.py with 4 GPUs allocated by SLURM
  # - uses 1 node, 4 tasks, 1 GPU per task
  # - master_port chosen randomly-ish to avoid collisions
  MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

  TRAIN_CMD=$(cat <<EOF
  srun --ntasks=1 --gpus=4 --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} \
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  baseline.py \
  --repo-root "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN" \
  --train-h5 "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/train.h5" \
  --train-csv "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/train.csv" \
  --test-h5  "/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN/DATA/test.h5" \
  --tile 128 \
  --batch-size 256 \
  --num-workers 16 \
  --seed 1337 \
  --max-epochs 50 \
  --val-every 3

EOF
)
fi

# If you want to hint what .h5 file to search for in lsof (optional):
: "${H5_HINT:=}"

# ----------------------------
# Helpers
# ----------------------------
have() { command -v "$1" >/dev/null 2>&1; }
ts="$(date +%Y%m%d_%H%M%S)"
host="$(hostname -s)"
jobid="${SLURM_JOB_ID:-no_slurm_jobid}"
outdir="probe_${host}_job${jobid}_${ts}"
mkdir -p "$outdir"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$outdir/PROBE.log" >&2; }

# ----------------------------
# Start training
# ----------------------------
log "Host: $host  JobID: $jobid"
log "Output dir: $outdir"
log "DURATION=${DURATION}s INTERVAL=${INTERVAL}s"
log "TRAIN_CMD:"
echo "$TRAIN_CMD" | tee -a "$outdir/PROBE.log" >&2

# Launch training in background, capture stdout/stderr
# Use eval so TRAIN_CMD can be a multi-line string with escapes.
set +e
( eval "$TRAIN_CMD" ) >"$outdir/train.stdout.log" 2>"$outdir/train.stderr.log" &
TRAIN_PID=$!
set -e
log "Training PID: $TRAIN_PID"

# Give it a moment to spawn ranks/workers
sleep 5

# Try to discover rank PIDs on this node (best-effort).
# We include children of TRAIN_PID plus python/torchrun processes owned by user.
USER_NAME="${USER:-$(id -un)}"

discover_pids() {
  local pids=""
  # Children of TRAIN_PID
  if have pgrep; then
    pids="$(pgrep -P "$TRAIN_PID" 2>/dev/null | tr '\n' ' ' || true)"
  fi
  # If empty, search by process name patterns (torchrun/python) for this user
  if [[ -z "${pids// }" ]] && have pgrep; then
    pids="$(pgrep -u "$USER_NAME" -f 'torchrun|python' | tr '\n' ' ' || true)"
  fi
  # Remove TRAIN_PID itself if present, then unique
  echo "$pids" | tr ' ' '\n' | awk -v tp="$TRAIN_PID" '$1!=tp && $1!=""' | sort -n | uniq | tr '\n' ' '
}

PIDS="$(discover_pids)"
PIDS_CSV="$(echo "$PIDS" | tr ' ' ',' | sed 's/,$//')"

log "Discovered candidate PIDs: ${PIDS:-<none>}"
log "PID CSV: ${PIDS_CSV:-<none>}"

# ----------------------------
# Snapshot: environment + node
# ----------------------------
{
  echo "==== BASIC ===="
  date
  hostname
  echo
  echo "==== SLURM ENV (filtered) ===="
  env | grep -E '^(SLURM|CUDA|NCCL|TORCH|OMP|MKL|OPENBLAS)_' | sort || true
  echo
  echo "==== CPU/MEM ===="
  (lscpu || true)
  echo
  (free -h || true)
  echo
  echo "==== GPU LIST ===="
  (nvidia-smi -L || true)
  echo
  echo "==== NVIDIA-SMI (summary) ===="
  (nvidia-smi || true)
  echo
  echo "==== MOUNTS (top) ===="
  (mount | head -n 200 || true)
} > "$outdir/snapshot_system.txt" 2>&1

# ----------------------------
# Snapshot: per-PID info (if we found any)
# ----------------------------
if [[ -n "${PIDS_CSV:-}" ]]; then
  IFS=',' read -r -a PID_ARR <<< "$PIDS_CSV"
  {
    echo "==== PROCESS TREE (TRAIN_PID) ===="
    (pstree -ap "$TRAIN_PID" || true)
    echo
    echo "==== PS DETAILS (training + discovered PIDs) ===="
    ps -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,psr,cmd -p "$TRAIN_PID" "${PID_ARR[@]}" || true
    echo
    echo "==== THREAD VIEW (top 40 threads per PID) ===="
    for pid in "$TRAIN_PID" "${PID_ARR[@]}"; do
      echo "--- PID $pid ---"
      ps -eLo pid,tid,psr,pcpu,pmem,stat,comm,cmd | awk -v p="$pid" '$1==p' | head -n 40 || true
      echo
    done
  } > "$outdir/snapshot_processes.txt" 2>&1

  {
    echo "==== OPEN FILES (grep .h5 / hint) ===="
    for pid in "$TRAIN_PID" "${PID_ARR[@]}"; do
      echo "--- PID $pid ---"
      if have lsof; then
        if [[ -n "$H5_HINT" ]]; then
          lsof -p "$pid" 2>/dev/null | grep -E '\.h5|'"$H5_HINT" || true
        else
          lsof -p "$pid" 2>/dev/null | grep -E '\.h5' || true
        fi
      else
        echo "lsof not available"
      fi
      echo
    done
  } > "$outdir/open_files_h5.txt" 2>&1
else
  log "No rank PIDs discovered; per-PID probes will be limited."
fi

# ----------------------------
# Timed sampling
# ----------------------------
SAMPLES=$(( DURATION / INTERVAL ))
if [[ "$SAMPLES" -lt 1 ]]; then SAMPLES=1; fi
log "Starting probes: ${SAMPLES} samples..."

# Run tools in background where possible
PROBE_PIDS=()

# GPU dmon
if have nvidia-smi; then
  ( nvidia-smi dmon -s pucvmet -d "$INTERVAL" -c "$SAMPLES" > "$outdir/gpu_dmon.txt" ) &
  PROBE_PIDS+=($!)
else
  log "nvidia-smi not available; skipping gpu_dmon."
fi

# iostat / pidstat / mpstat are in sysstat package
if have iostat; then
  ( iostat -xm "$INTERVAL" "$SAMPLES" > "$outdir/iostat_xm.txt" ) &
  PROBE_PIDS+=($!)
else
  log "iostat not available (sysstat); skipping."
fi

if have mpstat; then
  ( mpstat -P ALL "$INTERVAL" "$SAMPLES" > "$outdir/mpstat_all.txt" ) &
  PROBE_PIDS+=($!)
else
  log "mpstat not available (sysstat); skipping."
fi

# pidstat per process (only if we have PID list)
if have pidstat && [[ -n "${PIDS_CSV:-}" ]]; then
  ( pidstat -h -u -r -p "$TRAIN_PID,${PIDS_CSV}" "$INTERVAL" "$SAMPLES" > "$outdir/pidstat_cpu_mem.txt" ) &
  PROBE_PIDS+=($!)
  ( pidstat -h -d -p "$TRAIN_PID,${PIDS_CSV}" "$INTERVAL" "$SAMPLES" > "$outdir/pidstat_io.txt" ) &
  PROBE_PIDS+=($!)
elif have pidstat; then
  ( pidstat -h -u -r -p "$TRAIN_PID" "$INTERVAL" "$SAMPLES" > "$outdir/pidstat_cpu_mem.txt" ) &
  PROBE_PIDS+=($!)
  ( pidstat -h -d -p "$TRAIN_PID" "$INTERVAL" "$SAMPLES" > "$outdir/pidstat_io.txt" ) &
  PROBE_PIDS+=($!)
else
  log "pidstat not available (sysstat); skipping."
fi

# iotop (often needs permissions; best-effort)
if have iotop; then
  ( iotop -b -o -P -d "$INTERVAL" -n "$SAMPLES" > "$outdir/iotop.txt" 2>&1 ) &
  PROBE_PIDS+=($!)
else
  log "iotop not available; skipping."
fi

# periodic quick ps snapshots
(
  for ((i=1; i<=SAMPLES; i++)); do
    {
      echo "==== sample $i / $SAMPLES  $(date) ===="
      ps -o pid,ppid,stat,etime,%cpu,%mem,psr,cmd -p "$TRAIN_PID" || true
      if [[ -n "${PIDS_CSV:-}" ]]; then
        IFS=',' read -r -a PID_ARR2 <<< "$PIDS_CSV"
        ps -o pid,ppid,stat,etime,%cpu,%mem,psr,cmd -p "${PID_ARR2[@]}" || true
      fi
      echo
    } >> "$outdir/ps_samples.txt" 2>&1
    sleep "$INTERVAL"
  done
) &
PROBE_PIDS+=($!)

# Wait probes to finish
for p in "${PROBE_PIDS[@]}"; do
  wait "$p" || true
done
log "Probes finished."

# ----------------------------
# Summarize key tails for easy pasteback
# ----------------------------
{
  echo "==== SUMMARY ===="
  echo "Host: $host  JobID: $jobid  Timestamp: $ts"
  echo "Training PID: $TRAIN_PID"
  echo "Rank PIDs (best-effort): ${PIDS:-<none>}"
  echo "DURATION=${DURATION}s INTERVAL=${INTERVAL}s"
  echo

  if [[ -f "$outdir/gpu_dmon.txt" ]]; then
    echo "---- GPU DMON (tail) ----"
    tail -n 30 "$outdir/gpu_dmon.txt" || true
    echo
  fi
  if [[ -f "$outdir/mpstat_all.txt" ]]; then
    echo "---- MPSTAT (tail) ----"
    tail -n 40 "$outdir/mpstat_all.txt" || true
    echo
  fi
  if [[ -f "$outdir/iostat_xm.txt" ]]; then
    echo "---- IOSTAT -xm (tail) ----"
    tail -n 80 "$outdir/iostat_xm.txt" || true
    echo
  fi
  if [[ -f "$outdir/pidstat_io.txt" ]]; then
    echo "---- PIDSTAT IO (tail) ----"
    tail -n 60 "$outdir/pidstat_io.txt" || true
    echo
  fi
  if [[ -f "$outdir/pidstat_cpu_mem.txt" ]]; then
    echo "---- PIDSTAT CPU/MEM (tail) ----"
    tail -n 60 "$outdir/pidstat_cpu_mem.txt" || true
    echo
  fi
  if [[ -f "$outdir/iotop.txt" ]]; then
    echo "---- IOTOP (tail) ----"
    tail -n 80 "$outdir/iotop.txt" || true
    echo
  fi
} > "$outdir/SUMMARY.txt" 2>&1

log "Summary written: $outdir/SUMMARY.txt"

# ----------------------------
# Keep script alive until training ends (optional)
# ----------------------------
log "Waiting for training to finish (PID $TRAIN_PID)..."
set +e
wait "$TRAIN_PID"
TRAIN_RC=$?
set -e
log "Training finished with exit code: $TRAIN_RC"
exit "$TRAIN_RC"

