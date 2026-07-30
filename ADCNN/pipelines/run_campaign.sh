#!/bin/bash
# Process a list of nights end-to-end, unattended, surviving dead GPU shards.
#
# Each night is one `./adcnn night`: manifest -> detect (GPU) -> mask -> static catalogue ->
# frozen science link -> pixel vet -> MPC -> alert stream (link, P(real) re-rank, cutouts,
# per-alert images, sheets, summary).
#
# Three things this adds over calling ./adcnn night by hand:
#
#  1. PER-NIGHT STREAM OP, chosen by PANEL COUNT. The op is cadence-dependent: a full-cadence
#     night has ~33 linkable pointing groups at ~42 min gaps against a sparse night's ~4 at ~20
#     min, and chance links scale as dt^2, so one fixed op gives wildly different volume AND
#     runtime (the sparse op did not finish a single pointing group of 20260629 in 42 minutes).
#     Selecting on VISIT count was wrong: 20260710 has only 88 visits but 8,870 panels and 2.24M
#     detections -- 5x night 20260630 -- and the sparse op it was handed ran >10 h without
#     finishing. Seeding cost goes as detection density, which tracks panels, not visits.
#
#  2. AUTOMATIC RESIDUAL TOP-UP. A per-GPU shard that dies does not fail the slurm job -- it just
#     omits its panels, and the job reports COMPLETED. `uncorrectable ECC error` hit FIVE separate
#     ada nodes during one campaign (20260706 lost two shards in one job), so this is the common
#     case, not the exception. run_night's coverage guard turns that into a loud failure plus a
#     manifest_residual.csv; this loop re-detects exactly those panels into a side directory,
#     merges, and retries -- up to --attempts times -- instead of halting the night.
#
#  3. BAD-NODE AVOIDANCE. Nodes whose detect log shows an ECC fault are accumulated into a
#     persistent exclude list, so later passes stop landing on known-faulty hardware.
#
# Detection dominates: ~24 panels/min on 4 GPUs, so a 15k-panel night is ~10 h, and a night that
# loses a shard needs ~1.3 passes. Expect days for a full campaign. Everything is resumable.
#
# Usage:  bash ADCNN/pipelines/run_campaign.sh 20260705 20260706 ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
REPO=$PWD
source ADCNN/pipelines/heliolinc/pipeline_config.sh >/dev/null 2>&1
adcnn_activate || { echo "conda env failed"; exit 1; }

NIGHTS=("$@")
[ ${#NIGHTS[@]} -eq 0 ] && { echo "usage: $0 <night> [night ...]"; exit 1; }
SPEC=${CAMPAIGN_SPEC:-/tmp/campaign_nights.json}
ATTEMPTS=${ATTEMPTS:-3}
LOGDIR=outputs/logs; mkdir -p $LOGDIR
BADNODES=$LOGDIR/bad_gpu_nodes.txt; touch $BADNODES
SUMMARY=$LOGDIR/campaign_$(date +%Y%m%d_%H%M%S).log

jq_get() { python -c "import json,sys;d=json.load(open('$SPEC'))['$1'];print(d.get('$2','') if '$2'!='visits' else ','.join(str(v) for v in d['visits']))" 2>/dev/null; }
excl() { tr '\n' ',' < $BADNODES | sed 's/,$//'; }

# Re-detect the panels a dead shard skipped, then fold them into the night's catalogue.
topup() {
  local N=$1 R=outputs/runs/run_night_$N
  local RES=$R/manifest_residual.csv
  [ -s "$RES" ] || return 1
  local NP; NP=$(( $(wc -l < "$RES") - 1 ))
  local F=${R}_fill$(date +%s)
  mkdir -p "$F"; cp "$RES" "$F/manifest.csv"
  local E; E=$(excl)
  echo "    top-up: $NP panels -> $(basename $F) ${E:+(excluding $E)}" | tee -a $SUMMARY
  ADCNN_REPO=$REPO RUN=$PWD/$F sbatch ${E:+--exclude=$E} --wait \
      --export=ALL,RUN,ADCNN_REPO ADCNN/pipelines/heliolinc/sn_detect.slurm \
      >> $LOGDIR/campaign_night_$N.log 2>&1
  # any node that threw ECC in this pass is remembered so later passes avoid it
  for L in $LOGDIR/sn_detect_*.log; do
    if grep -qi "uncorrectable ECC" "$L" 2>/dev/null; then
      grep -oE "sdfada[0-9]+" "$L" 2>/dev/null | sort -u >> $BADNODES
    fi
  done
  sort -u -o $BADNODES $BADNODES
  [ -s "$F/adcnn_dets.csv" ] || { echo "    top-up produced nothing" | tee -a $SUMMARY; return 1; }
  python -m ADCNN.pipelines.heliolinc.merge_dets --out "$R/adcnn_dets.csv" \
      "$R/adcnn_dets.csv" "$F/adcnn_dets.csv" >> $LOGDIR/campaign_night_$N.log 2>&1
  rm -f "$R/adcnn_dets_masked.csv"     # force re-mask over the enlarged catalogue
  return 0
}

echo "campaign: ${NIGHTS[*]}  (attempts=$ATTEMPTS)" | tee -a $SUMMARY
for N in "${NIGHTS[@]}"; do
  COLL=$(jq_get "$N" collection); VIS=$(jq_get "$N" visits); NV=$(jq_get "$N" n)
  if [ -z "$COLL" ] || [ -z "$VIS" ]; then
    echo "[$N] SKIP: not in $SPEC" | tee -a $SUMMARY; continue; fi
  NP=$(jq_get "$N" panels)
  if [ "${NP:-0}" -ge 3000 ]; then OP=ADCNN/pipelines/heliolinc/op_2v_stream_fullcadence.json
  else OP=ADCNN/pipelines/heliolinc/op_2v_stream.json; fi
  echo "[$N] $NV visits, ${NP:-?} panels -> $(basename $OP)" | tee -a $SUMMARY
  t0=$SECONDS
  for try in $(seq 1 $ATTEMPTS); do
    ./adcnn night --butler-repo embargo --collection "$COLL" --night "$N" --no-known \
        --visits "$VIS" --stream-op-point "$OP" >> $LOGDIR/campaign_night_$N.log 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    if grep -q "MISSING" $LOGDIR/campaign_night_$N.log 2>/dev/null && [ $try -lt $ATTEMPTS ]; then
      echo "[$N] attempt $try: detection incomplete, topping up" | tee -a $SUMMARY
      topup "$N" || break
    else
      break
    fi
  done
  A=$(wc -l < outputs/runs/run_night_$N/stream/alerts.jsonl 2>/dev/null || echo 0)
  P=$(ls outputs/runs/run_night_$N/stream/pairs 2>/dev/null | wc -l)
  echo "[$N] rc=$rc  $((SECONDS-t0))s  alerts=$A  images=$P" | tee -a $SUMMARY
  [ $rc -ne 0 ] && echo "[$N] see $LOGDIR/campaign_night_$N.log" | tee -a $SUMMARY
done
echo "campaign done -> $SUMMARY" | tee -a $SUMMARY
echo "known-bad GPU nodes: $(tr '\n' ' ' < $BADNODES)" | tee -a $SUMMARY
