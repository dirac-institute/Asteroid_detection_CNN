#!/bin/bash
# usdf-night prep (CPU, LSST stack): panel manifest, Rubin source tables and Gaia refcat of one night.
# Submitted by status.py with every resource flag on the sbatch line:
#   prep.sbatch <worktree> <night dir> <day_obs> <loadLSST.sh>
# A step whose output is already in work/ is skipped (each output is written atomically).
set -uo pipefail
umask 002
W=$1 N=$2 DAY=$3 STACK=$4
st() { /sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn/bin/python -I "$W/ADCNN/pipelines/usdf_night/status.py" "$@"; }
st stage "$N" prep start
(
    set +u +o pipefail                         # loadLSST.sh and eups are not written for -u
    source "$STACK" && setup lsst_distrib || exit 1
    set -e
    export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED
    export PYTHONPATH=$W${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 ADCNN_REPO=$W ADCNN_PIPELINE=$W/models/v2_D2/pipeline.json
    w=$N/work
    [[ -e $w/manifest.csv ]] || python -m ADCNN.pipelines.usdf_night.night manifest "$DAY" --out "$w/manifest.csv"
    [[ -e $w/stack_sources.parquet ]] || python -m ADCNN.pipelines.usdf_night.stack_tables "$DAY" \
        --manifest "$w/manifest.csv" --out "$w/stack_sources.parquet" --workers 16
    [[ -e $w/refcat.parquet ]] || python -m ADCNN.pipelines.usdf_night.refcat \
        --manifest "$w/manifest.csv" --out "$w/refcat.parquet" --workers 16
)
rc=$?
st stage "$N" prep end "$rc"
exit "$rc"
