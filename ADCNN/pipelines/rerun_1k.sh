#!/bin/bash
# Build the 1k-cadence product for already-processed nights: ~1000 ripple-CLEAN, P(real)-ranked
# alerts per full-cadence night, reusing the existing linking (no GPU re-detect, no re-link).
#
# Per night, reading from outputs/runs/10k_cadence/run_night_<N> and writing 1k_cadence/run_night_<N>:
#   1. take the top-K alerts by P(real) from the existing (already-ranked) stream
#   2. cut mosaicked cutouts for those K            (ADCNN.qa.alert_cutouts -- multi-detector wide view)
#   3. score each by SHAPE                          (ADCNN.qa.alert_morphology -- elongation + dipole)
#   4. drop bright-star dipole RINGS, keep top-N clean by P(real), reindex the cache (ADCNN.qa.select_clean)
#   5. render per-alert pairs + contact sheets + summary for the N
#
# TWO complementary ring filters (the earlier "proximity does not separate rings from trails" claim was
# measured CIRCULARLY -- against product alerts that are themselves ring-contaminated -- and is WRONG):
#   * PROXIMITY to a DEEP all-sky refcat (--refcat, mag<21 @ 2.5"): the primary lever. Measured on
#     20260706 with an OFFSET NULL (members shifted 20-60": preserves footprint/density, decorrelates
#     from stars, so the null rate IS the chance cost a real mover pays) -> flags 55.8% of the product
#     at 2.7% mover cost (~20:1). DEPTH IS EVERYTHING: the residual rings sit on mag 19-21 stars, so a
#     mag<19 refcat catches 0% of them (their nearest CATALOGUED star was 26" away). Refcats MUST be
#     built with --mag-max 21 (build_refcats.sh).
#   * MORPHOLOGY / dipole SHAPE (alert_morphology): catches BRIGHT rings regardless of catalog coverage.
#     It cannot see the faint residuals (flags 0-4% of them while costing 8-11% of injected faint
#     trails -- at those flux levels the negative-lobe test fires on noise), so it complements, not
#     replaces, proximity.
#
# Usage:  bash ADCNN/pipelines/rerun_1k.sh 20260705 20260706 ...
#         N_CLEAN=1000 TOPK=2000 bash ADCNN/pipelines/rerun_1k.sh 20260705
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source ADCNN/pipelines/heliolinc/pipeline_config.sh >/dev/null 2>&1
adcnn_activate || { echo "conda env failed"; exit 1; }
OP=${OP:-ADCNN/pipelines/heliolinc/op_2v_stream_1k.json}   # threshold op-point (chi2<=5 + drops)
WORKERS=${WORKERS:-16}
LOG=outputs/logs; mkdir -p $LOG

for N in "$@"; do
  SRC=outputs/runs/10k_cadence/run_night_$N
  # ALWAYS prefer the ADCNN+stack merged catalogue when the night has one (run_night's stack_merge
  # stage): the stack finds ~3.4% of real movers ADCNN misses (short trails, 1-2 deg/day), and the
  # union is the reported product. Falls back to ADCNN-only for nights processed before the merge.
  DETS=$SRC/adcnn_dets_masked.csv
  [ -s "$SRC/dets_merged.csv" ] && DETS=$SRC/dets_merged.csv
  DST=outputs/runs/1k_cadence/run_night_$N; sd=$DST/stream
  L=$LOG/rerun_1k_$N.log
  if [ ! -s "$SRC/stream/alerts.jsonl" ] || [ ! -s "$DETS" ]; then
    echo "[$N] SKIP: missing src alerts or masked dets"; continue; fi
  # skip a night already built (resumable/unattended): needs both the alert list and its sheets
  if [ -s "$sd/alerts.jsonl" ] && [ -f "$sd/sheets/index.html" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$N] already built ($(wc -l < "$sd/alerts.jsonl") alerts) -- skip"; continue; fi
  echo "[$N] 1k rerun -> $L"
  {
    mkdir -p "$sd"
    # The refcat drives the bright-star PROXIMITY veto. It was previously NOT passed, so that veto was
    # silently inactive in the product (op flag on, no catalog = no-op) -- which is why rings survived.
    RC=""
    if [ -s "$SRC/bright_refcat.parquet" ]; then RC="--refcat $SRC/bright_refcat.parquet"
    else echo "WARNING: no $SRC/bright_refcat.parquet -- PROXIMITY VETO OFF (build with build_refcats.sh)"; fi
    echo "=== $N threshold filter ($OP) $RC ==="
    python -m ADCNN.qa.filter_op --alerts "$SRC/stream/alerts.jsonl" \
        --dets "$DETS" --op "$OP" --out "$sd/_surv.jsonl" $RC \
        || { echo "FILTER FAILED"; exit 1; }
    echo "=== $N cutouts (mosaic) for survivors ==="
    python -m ADCNN.qa.alert_cutouts --alerts "$sd/_surv.jsonl" \
        --dets "$DETS" --out "$sd/_surv_cutouts.npz" \
        --stamp-px 96 --workers "$WORKERS" || { echo "CUTOUTS FAILED"; exit 1; }
    echo "=== $N morphology ==="
    python -m ADCNN.qa.alert_morphology --alerts "$sd/_surv.jsonl" \
        --cutouts "$sd/_surv_cutouts.npz" --out "$sd/_morph.npz" || { echo "MORPH FAILED"; exit 1; }
    echo "=== $N select (drop residual dipole rings; keep ALL survivors) ==="
    python -m ADCNN.qa.select_clean --alerts "$sd/_surv.jsonl" --morph "$sd/_morph.npz" \
        --cutouts "$sd/_surv_cutouts.npz" --n 999999 --mode rings \
        --out-alerts "$sd/alerts.jsonl" --out-cutouts "$sd/cutouts.npz" || { echo "SELECT FAILED"; exit 1; }
    NA=$(wc -l < "$sd/alerts.jsonl")
    echo "=== $N render $NA (sheets + pairs + summary) ==="
    python -m ADCNN.qa.alert_sheets --alerts "$sd/alerts.jsonl" --cutouts "$sd/cutouts.npz" \
        --out-dir "$sd/sheets" --per-sheet 48 --limit "$NA" || { echo "SHEETS FAILED"; exit 1; }
    python -m ADCNN.qa.alert_pairs --alerts "$sd/alerts.jsonl" --cutouts "$sd/cutouts.npz" \
        --out-dir "$sd/pairs" --top-n "$NA" || { echo "PAIRS FAILED"; exit 1; }
    STATIC=$SRC/static_catalog.parquet
    python -m ADCNN.qa.stream_summary --alerts "$sd/alerts.jsonl" --out "$sd/stream_summary.json" \
        $( [ -f "$STATIC" ] && echo "--static-catalog $STATIC" ) || true
    rm -f "$sd/_surv.jsonl" "$sd/_surv_cutouts.npz" "$sd/_surv_cutouts_meta.json" "$sd/_morph.npz" "$sd/cutouts.npz" "$sd/cutouts_meta.json"
    echo "=== $N DONE ($NA clean alerts) ==="
  } > "$L" 2>&1
  if grep -q "$N DONE" "$L"; then echo "[$N] ok: $(grep -oE 'DONE \([0-9]+ clean' $L | grep -oE '[0-9]+') clean alerts"; else echo "[$N] FAILED -- see $L"; fi
done
echo "1k rerun done"
