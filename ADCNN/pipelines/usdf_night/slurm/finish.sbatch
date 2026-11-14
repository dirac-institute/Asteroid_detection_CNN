#!/bin/bash
# usdf-night finish (CPU): resubmit detection until the manifest is covered, then join the night.
# Submitted by status.py with every resource flag on the sbatch line:
#   finish.sbatch <worktree> <night dir>
# Complete -> join, DONE, work/ removed. Attempts exhausted -> merge the done panels, join, PARTIAL
# (work/ kept). No prep outputs, nothing detected, or a failed merge/join -> FAILED. Lock removed.
set -uo pipefail
umask 002
W=$1 N=$2
PY=/sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn/bin/python
st() { "$PY" -I "$W/ADCNN/pipelines/usdf_night/status.py" "$@"; }
stop() { st stage "$N" finish end "$1"; exit "$1"; }
st stage "$N" finish start
export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 ADCNN_REPO=$W ADCNN_PIPELINE=$W/models/v2_D2/pipeline.json
step=$(st finish "$N") || stop 1                # FAILED already recorded by status.py
case $step in
    retry) stop 0 ;;
    partial)
        "$PY" -m ADCNN.inference.detect_night --manifest "$N/work/manifest.csv" \
            --out "$N/work/adcnn_dets.csv" --allow-partial \
            || { st fail "$N" "merge of the done panels failed (exit $?)"; stop 1; } ;;
esac
"$PY" -m ADCNN.pipelines.usdf_night.join "$N" --workers 32
rc=$?
if (( rc )); then
    st fail "$N" "join exited $rc"
    stop "$rc"
fi
st stage "$N" finish end 0
st close "$N"
