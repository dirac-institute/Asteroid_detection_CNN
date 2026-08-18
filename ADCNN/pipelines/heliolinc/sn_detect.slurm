#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac   # ada grants kipac a NORMAL (non-preemptable) qos; the
                                # preemptable-only problem is specific to AMPERE, where
                                # 20260629 detection was evicted twice as a bare CANCELLED.
                                # rubin:commissioning also runs here, but its ada group CPU
                                # cap rejects a 32-core request (AssocGrpCpuLimit) once
                                # ~12 cores are already in use by the account.
# MEASURED 2026-08-11 on a hopper H200, 40 real panels: detection costs 4.6 s/panel end to end
# (seg inference 64%, cutout CNN 13%, pixel read 11%). A full-cadence night is ~16,937 panels =
# 21.7 GPU-hours. Every ada node HAS TEN L40S, so asking for four left 60% of the node idle:
#     4 GPUs -> 5.4 h/night      10 GPUs -> 2.2 h/night
# The old header comment claiming "~11 h on 4 GPUs" predates the inference optimisation.
#SBATCH --gres=gpu:l40s:5
#SBATCH -c 36
#SBATCH --mem=350G
#SBATCH -t 6:00:00   # ~2.2 h/night on 10 GPUs; the kipac:kipac@ada quota was CUT to
                      # cpu=36/gpu=5/mem=351G (measured 2026-08-18: a 72c/10-GPU/600G ask
                      # pends FOREVER on AssocGrpCpuLimit with zero group usage), so this
                      # now asks half the node: ~4.4 h for a full night, still inside 6 h
                      # 8 h wall killed 20260629 at ~70%% (per-shard .done makes a resubmit
                      # resume, but the wall should not be the thing that ends the job)
#SBATCH -J sn_detect
#SBATCH -o outputs/logs/%x_%j.log
#SBATCH --requeue
#SBATCH --open-mode=append
set -eo pipefail
# Canonical same-night pipeline, Stage 1: ADCNN (segmentation + focal cutout CNN) over the manifest's
# diffims -> sky detections + trail endpoints. Resumable (per-panel .done) so a preemption + resubmit
# (driven by sn_run.slurm) continues. $RUN must point at a dir containing manifest.csv.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# under sbatch the script is copied to a spool dir, so BASH_SOURCE can't see pipeline_config.sh. Resolve
# the heliolinc dir from $ADCNN_REPO (passed through --export=ALL by sn_run) else the submit dir.
if [ ! -f "$HERE/pipeline_config.sh" ]; then
  if [ -n "$ADCNN_REPO" ] && [ -f "$ADCNN_REPO/ADCNN/pipelines/heliolinc/pipeline_config.sh" ]; then
    HERE="$ADCNN_REPO/ADCNN/pipelines/heliolinc"
  else HERE="${SLURM_SUBMIT_DIR:-$HERE}"; fi
fi
# shellcheck disable=SC1091
source "$HERE/pipeline_config.sh"
: "${RUN:?set RUN=<dir with manifest.csv>}"
adcnn_activate || exit 1
cd "$ADCNN_REPO"
python ADCNN/inference/detect_night.py --n-gpus "${DETECT_N_GPUS:-5}" \
  --manifest "$RUN/manifest.csv" --out "$RUN/adcnn_dets.csv"
echo "SN_DETECT DONE"
