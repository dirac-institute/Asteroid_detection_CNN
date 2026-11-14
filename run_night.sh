#!/bin/bash
# ADCNN + Rubin detections of one LSSTCam night on S3DF (Slurm chain prep -> detect -> finish).
#   run_night.sh <YYYY-MM-DD|YYYYMMDD> [--force]   submit the night (an incomplete one resumes) and return
#   run_night.sh status <YYYY-MM-DD|YYYYMMDD>       panels done, job states, errors, lock
# Environment overrides, kept in state.json for the resubmits: ACCOUNT PARTITION GPUS CPUS MEM TIME
# CPU_PARTITION FLOOR OUT_ROOT MAX_ATTEMPTS STACK (defaults in ADCNN/pipelines/usdf_night/status.py).
set -euo pipefail
umask 002
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PY=/sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn/bin/python
STATUS=$W/ADCNN/pipelines/usdf_night/status.py

usage() {
    echo "usage: run_night.sh <YYYY-MM-DD|YYYYMMDD> [--force] | run_night.sh status <YYYY-MM-DD|YYYYMMDD>" >&2
    exit 1
}

night_check() {  # <day> <loadLSST.sh>: night.py check inside the LSST stack; stdout ends with its JSON line
    set +u +o pipefail
    source "$2" >/dev/null 2>&1 && setup lsst_distrib >/dev/null 2>&1 \
        || { echo "run_night: cannot set up the LSST stack from $2" >&2; return 1; }
    export AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED AWS_RESPONSE_CHECKSUM_VALIDATION=WHEN_REQUIRED
    PYTHONPATH=$W${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 python -m ADCNN.pipelines.usdf_night.night check "$1"
}

if [[ ${1:-} == status ]]; then
    [[ $# == 2 ]] || usage
    exec "$PY" -I "$STATUS" show "$2"
fi
[[ $# == 1 || ( $# == 2 && ${2:-} == --force ) ]] && [[ ${1:-} != -* ]] || usage
p=$("$PY" -I "$STATUS" prepare "$1")            # day_obs and loadLSST.sh; exit 1 on a bad day or override
day=${p%%$'\n'*} stack=${p#*$'\n'}
check=$(night_check "$day" "$stack") || exit $?  # 2 = credentials (night.py printed why)
exec "$PY" -I "$STATUS" submit "$day" --check "${check##*$'\n'}" ${2:+--force}
