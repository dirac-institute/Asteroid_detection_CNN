#!/usr/bin/env bash
# Regenerate every embargo night end to end with the post-audit pipeline.
#
# WHY A REGENERATION IS NEEDED (all three fixed since the delivered products were made):
#   * six nights' contact sheets show the WRONG alert's pixels -- the cutout cache is keyed by alert
#     POSITION and rerank_alerts permuted alerts.jsonl after the cut (see SHEETS_INVALID.txt);
#     run_night now ranks BEFORE it cuts and the cache carries a sequence fingerprint
#   * the merge never ring-cleaned the stack side (61.2% ring-positioned vs 10.4% chance) and
#     re-imported deleted rings; cleaning both sides measured 9.26% -> 9.75% delivered completeness
#     at the 1k budget (gained 20, lost 1, p=2.1e-05)
#   * filter_op dropped the ENTIRE 3+visit discovery tier (chi2 is None there, and the default was
#     99 against an 8.0 gate) -- 52 alerts across nine nights, every delivered
#     multi_epoch_fraction = 0.0
#
# WHAT IS REUSED: detection. adcnn_dets.csv / adcnn_dets_masked.csv / manifest.csv / known.csv /
# bright_refcat.parquet are kept, so no GPU time is spent. Only the stale stages are deleted, and
# run_night's own existence checks then rebuild exactly those. That is what makes this cheap and
# what makes it resumable: re-running this script after any failure picks up where it stopped.
#
# ROBUSTNESS: every night is independent and wrapped, so one failure cannot stop the campaign; each
# night is verified by night_status before being marked done; a .regen_complete sentinel makes
# re-entry a no-op for finished nights.
#
#   ADCNN/pipelines/regen_campaign.sh [night ...]     (default: all nine)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source ADCNN/pipelines/heliolinc/pipeline_config.sh
adcnn_activate

OUT_ROOT="${OUT_ROOT:-outputs/runs/10k_cadence}"
LOG_DIR="outputs/logs"; mkdir -p "$LOG_DIR"
BUDGET="${BUDGET:-1000}"
OP1K="ADCNN/pipelines/heliolinc/op_2v_stream_1k.json"

collection_for() {   # the prompt-processing collection differs by night; wrong one = zero DIA sources
  case "$1" in
    20260629|20260630|20260705|20260706) echo "LSSTCam/runs/prompt/$1/ApPipe/pipelines-b856041-config-8f017ea" ;;
    20260708)                            echo "LSSTCam/runs/prompt/$1/ApPipe/pipelines-87bbc9a-config-8f017ea" ;;
    20260710|20260711|20260712|20260713) echo "LSSTCam/runs/prompt/$1/ApPipe/pipelines-69e0100-config-8f017ea" ;;
    *) echo "" ;;
  esac
}

NIGHTS=("$@")
[ ${#NIGHTS[@]} -eq 0 ] && NIGHTS=(20260629 20260630 20260705 20260706 20260708 20260710 20260711 20260712 20260713)

for N in "${NIGHTS[@]}"; do
  D="$OUT_ROOT/run_night_$N"
  LOG="$LOG_DIR/regen_$N.log"
  if [ -f "$D/.regen_complete" ]; then echo "[$N] already regenerated -- skip"; continue; fi
  COLL="$(collection_for "$N")"
  if [ -z "$COLL" ]; then echo "[$N] NO COLLECTION MAPPED -- skipping (add it to collection_for)"; continue; fi
  if [ ! -s "$D/adcnn_dets_masked.csv" ]; then
    echo "[$N] no adcnn_dets_masked.csv -- detection artifacts missing, skipping (this script does "\
"NOT re-run detection; run ./adcnn night for that)"; continue
  fi
  echo "=== [$N] regenerating -> $LOG ==="
  {
    echo "### $(date -Is) regen $N  collection=$COLL"
    # Delete ONLY what the fixes invalidate. Detection stays.
    rm -f  "$D/dets_merged.csv" "$D/stack_dets.csv" "$D/.complete" "$D/SHEETS_INVALID.txt"
    rm -rf "$D/stream" "$D/stream_1k"
    # --stream-workers 32: the wide stage now chunks by (visit, detector) rather than by visit, so
    # its parallelism is bounded by chunk count (hundreds) instead of visit count (tens) and the old
    # 4-worker RAM cap is gone. 32 of 128 cores leaves headroom for the linker's own pool.
    python -m ADCNN.pipelines.run_night --night "$N" --collection "$COLL" \
        --visits auto --out "$D" --keep-cutouts --stream-workers 32
    RC=$?
    echo "### run_night rc=$RC"
    # ---- the ~1k clean deliverable, rebuilt from the corrected stream ----
    if [ $RC -eq 0 ] && [ -s "$D/stream/alerts.jsonl" ]; then
      K="$D/stream_1k"; mkdir -p "$K"
      python -m ADCNN.qa.filter_op --alerts "$D/stream/alerts.jsonl" --dets "$D/dets_merged.csv" \
          --op "$OP1K" --out "$K/surv.jsonl" --refcat "$D/bright_refcat.parquet" \
        && head -n "$BUDGET" "$K/surv.jsonl" > "$K/topk.jsonl" \
        && python -m ADCNN.qa.alert_cutouts --alerts "$K/topk.jsonl" --dets "$D/dets_merged.csv" \
              --out "$K/_cut.npz" --stamp-px 96 --workers 32 \
        && python -m ADCNN.qa.alert_morphology --alerts "$K/topk.jsonl" --cutouts "$K/_cut.npz" \
              --out "$K/_morph.npz" \
        && python -m ADCNN.qa.select_clean --alerts "$K/topk.jsonl" --morph "$K/_morph.npz" \
              --cutouts "$K/_cut.npz" --n "$BUDGET" --mode rings \
              --out-alerts "$K/alerts.jsonl" --out-cutouts "$K/cutouts.npz" \
        && python -m ADCNN.qa.alert_sheets --alerts "$K/alerts.jsonl" --cutouts "$K/cutouts.npz" \
              --out-dir "$K/sheets" --per-sheet 48 \
        && python -m ADCNN.qa.alert_pairs --alerts "$K/alerts.jsonl" --cutouts "$K/cutouts.npz" \
              --out-dir "$K/pairs" --workers 12 \
        && python -m ADCNN.qa.stream_summary --alerts "$K/alerts.jsonl" --out "$K/stream_summary.json"
      echo "### stream_1k rc=$?"
      rm -f "$K/_cut.npz" "$K/_morph.npz"
    fi
    # ---- verify before declaring the night done ----
    python -m ADCNN.pipelines.night_status --json "$D" > "$D/regen_status.json" 2>&1
    # night_status --json emits a LIST of night records, not a dict
    if python -c "import json,sys; r=json.load(open('$D/regen_status.json')); r=r[0] if isinstance(r,list) else r; sys.exit(0 if r.get('complete') else 1)" 2>/dev/null; then
      : > "$D/.regen_complete"; echo "### $N VERIFIED COMPLETE"
    else
      echo "### $N NOT verified -- see regen_status.json (re-run this script to retry)"
    fi
  } >> "$LOG" 2>&1
  echo "[$N] done ($(grep -c '###' "$LOG" 2>/dev/null) markers) -- $(tail -1 "$LOG" | cut -c1-70)"
done
echo "=== campaign regen pass finished $(date -Is) ==="
