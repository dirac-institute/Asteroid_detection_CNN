#!/bin/bash
# Process a list of nights end-to-end, one after another, unattended.
#
# Each night is an independent `./adcnn night` run: manifest -> detect (GPU) -> mask -> static
# catalogue -> frozen science link -> pixel vet -> MPC -> alert stream (link, P(real) re-rank,
# cutouts, per-alert images, contact sheets, summary).
#
# Two things this wrapper adds over calling ./adcnn night by hand:
#   * per-night STREAM OP. The stream op is cadence-dependent -- a full-cadence night has ~33
#     linkable pointing groups at ~42 min gaps against a sparse night's ~4 at ~20 min, and chance
#     links scale as dt^2, so one fixed op gives wildly different volume and runtime. Nights are
#     classified by visit count and given the matching op.
#   * it never stops the campaign on one night's failure. A night that dies (dead GPU shard, no
#     DRP coverage, whatever) is logged and skipped; the rest still run.
#
# Detection dominates: ~24 panels/min on 4 GPUs, so a 15k-panel night is ~10 h. Expect days, not
# hours, for a full campaign. Everything is resumable -- re-running skips completed stages.
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
LOGDIR=outputs/logs; mkdir -p $LOGDIR
SUMMARY=$LOGDIR/campaign_$(date +%Y%m%d_%H%M%S).log

echo "campaign: ${NIGHTS[*]}" | tee -a $SUMMARY
for N in "${NIGHTS[@]}"; do
  COLL=$(python -c "import json;print(json.load(open('$SPEC'))['$N']['collection'])" 2>/dev/null)
  VIS=$(python -c "import json;d=json.load(open('$SPEC'))['$N'];print(','.join(str(v) for v in d['visits']))" 2>/dev/null)
  NV=$(python -c "import json;print(json.load(open('$SPEC'))['$N']['n'])" 2>/dev/null)
  if [ -z "$COLL" ] || [ -z "$VIS" ]; then
    echo "[$N] SKIP: no collection/visits in $SPEC" | tee -a $SUMMARY; continue
  fi
  # >=100 visits => the telescope slewed through a sequence: many co-pointed groups, wide gaps,
  # high density. Anything looser than the full-cadence op does not finish in reasonable time.
  if [ "$NV" -ge 100 ]; then OP=ADCNN/pipelines/heliolinc/op_2v_stream_fullcadence.json
  else OP=ADCNN/pipelines/heliolinc/op_2v_stream.json; fi
  echo "[$N] $NV visits -> $(basename $OP)" | tee -a $SUMMARY
  t0=$SECONDS
  ./adcnn night --butler-repo embargo --collection "$COLL" --night "$N" --no-known \
      --visits "$VIS" --stream-op-point "$OP" > $LOGDIR/campaign_night_$N.log 2>&1
  rc=$?
  A=$(wc -l < outputs/runs/run_night_$N/stream/alerts.jsonl 2>/dev/null || echo 0)
  P=$(ls outputs/runs/run_night_$N/stream/pairs 2>/dev/null | wc -l)
  echo "[$N] rc=$rc  $((SECONDS-t0))s  alerts=$A  images=$P" | tee -a $SUMMARY
  [ $rc -ne 0 ] && echo "[$N] see $LOGDIR/campaign_night_$N.log" | tee -a $SUMMARY
done
echo "campaign done -> $SUMMARY" | tee -a $SUMMARY
