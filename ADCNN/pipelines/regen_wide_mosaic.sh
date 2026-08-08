#!/bin/bash
# Regenerate the wide-view MOSAIC for already-processed nights: re-cut cutouts.npz with the new
# multi-detector tangent-plane mosaic (ADCNN/qa/alert_cutouts.py) and re-render the per-alert pair
# figures. Contact SHEETS use only zoom stamps (unchanged by the mosaic), so they are left as-is.
#
# The wide view is a CONTEXT panel; the zoom stamps and all alert science are untouched. This only
# repaints the third panel of each alert_*.png so off-detector regions show real neighbour-detector
# pixels instead of a grey void. Heavy S3 I/O (each visit loads its overlapping detector panels
# once); expect hours for the full-cadence nights.
#
# Usage:  bash ADCNN/pipelines/regen_wide_mosaic.sh 20260705 20260706 ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source ADCNN/pipelines/heliolinc/pipeline_config.sh >/dev/null 2>&1
adcnn_activate || { echo "conda env failed"; exit 1; }
LOG=outputs/logs; mkdir -p $LOG

for N in "$@"; do
  R=outputs/runs/run_night_$N; sd=$R/stream
  L=$LOG/regen_wide_$N.log
  if [ ! -s "$sd/alerts.jsonl" ] || [ ! -s "$R/adcnn_dets_masked.csv" ]; then
    echo "[$N] SKIP: missing alerts.jsonl or masked dets"; continue; fi
  echo "[$N] re-cut mosaic cutouts + re-render pairs -> $L"
  {
    echo "=== $N cutouts (mosaic) ==="
    rm -f "$sd/cutouts.npz"
    python -m ADCNN.qa.alert_cutouts --alerts "$sd/alerts.jsonl" \
        --dets "$R/adcnn_dets_masked.csv" --out "$sd/cutouts.npz" \
        --stamp-px 96 --workers 16 --limit 20000 || { echo "CUTOUTS FAILED"; exit 1; }
    echo "=== $N pairs ==="
    python -m ADCNN.qa.alert_pairs --alerts "$sd/alerts.jsonl" \
        --cutouts "$sd/cutouts.npz" --out-dir "$sd/pairs" --top-n 20000 || { echo "PAIRS FAILED"; exit 1; }
    rm -f "$sd/cutouts.npz"       # regenerable intermediate (~1 GB); drop it like run_night does
    echo "=== $N DONE ==="
  } > "$L" 2>&1
  if grep -q "$N DONE" "$L"; then echo "[$N] ok"; else echo "[$N] FAILED -- see $L"; fi
done
echo "regen done"
