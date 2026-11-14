#!/bin/bash
# usdf-night detect (GPU): ADCNN detection on every manifest panel, resumed from the shard .done logs.
# Submitted by status.py with every resource flag on the sbatch line:
#   detect.sbatch <worktree> <night dir> <n gpus> <cnn floor>
# Exit 3 = panels still missing (shards kept); the finish job resubmits or merges what is done.
set -uo pipefail
umask 002
W=$1 N=$2 GPUS=$3 FLOOR=$4
PY=/sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn/bin/python
st() { "$PY" -I "$W/ADCNN/pipelines/usdf_night/status.py" "$@"; }
st stage "$N" detect start --gpus "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 ADCNN_REPO=$W ADCNN_PIPELINE=$W/models/v2_D2/pipeline.json
md5sum --quiet -c - <<EOF || { st stage "$N" detect end 1; exit 1; }
147cef8bb4e872b3fa0560a264e4748e  $W/models/v2_D2/segmentation_scripted.pt
0ba43c29fc1c76c63d992b871c83474b  $W/models/v2_D2/cnn_postproc.pt
EOF
"$PY" -m ADCNN.inference.detect_night --manifest "$N/work/manifest.csv" --out "$N/work/adcnn_dets.csv" \
    --n-gpus "$GPUS" --cnn-thr "$FLOOR"
rc=$?
st stage "$N" detect end "$rc"
exit "$rc"
