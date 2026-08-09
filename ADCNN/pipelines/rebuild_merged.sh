#!/bin/bash
# FULL unattended rebuild of the nightly product under the CURRENT pipeline strategy:
#   ADCNN detections (rings dropped) UNION stack DIA sources -> relink -> 1k product.
#
# Per night, all four steps, each resumable (a completed step is skipped on re-run):
#   1. ingest  stack DIA sources for the night           (Butler/LSST stack env, ~30-60 min)
#   2. merge   ADCNN + stack, RINGS DROPPED FIRST        (deep refcat; chance links go as n1*n2)
#   3. link    the merged catalogue at score_min 0.70    (0.5 is the O(N^2) wall: 1.5M linkable
#              dets and a 5.3h non-finishing link, vs 208k and ~21 min at 0.70 -- and the
#              delivered stream's own weakest-member score minimum is already 0.700)
#   4. product filter_op (chi2<=8, deep-refcat ring veto) -> cutouts -> morphology -> render
#
# The Butler COLLECTION is derived per night from the manifest's fits_path, because the prompt
# pipeline hash is NOT constant across nights (e.g. 20260713 uses pipelines-69e0100, not -b856041).
#
# Usage:  bash ADCNN/pipelines/rebuild_merged.sh 20260629 20260630 ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source ADCNN/pipelines/heliolinc/pipeline_config.sh >/dev/null 2>&1
adcnn_activate || { echo "conda env failed"; exit 1; }
export PYTHONPATH="$PWD"
OP=${OP:-ADCNN/pipelines/heliolinc/op_2v_stream_1k.json}
STREAM_OP=${STREAM_OP:-ADCNN/pipelines/heliolinc/op_2v_stream.json}
SCORE_MIN=${SCORE_MIN:-0.7}
WORKERS=${WORKERS:-24}
LINK_WORKERS=${LINK_WORKERS:-40}
LOG=outputs/logs; mkdir -p $LOG

for N in "$@"; do
  SRC=outputs/runs/10k_cadence/run_night_$N
  DST=outputs/runs/1k_cadence/run_night_$N; sd=$DST/stream
  L=$LOG/rebuild_$N.log
  [ -s "$SRC/adcnn_dets_masked.csv" ] || { echo "[$N] SKIP: no detections"; continue; }
  echo "[$N] rebuild -> $L"
  {
    echo "===== NIGHT $N  $(date '+%F %T') ====="
    COLL=$(head -2 "$SRC/manifest.csv" | tail -1 | cut -d, -f5 | sed -E 's|^s3://[^/]+/||; s|/difference_image/.*||')
    echo "collection: $COLL"

    # --- 1. stack DIA sources -------------------------------------------------------------
    if [ ! -s "$SRC/stack_dets.csv" ]; then
      echo "--- [1/4] ingest DIA sources $(date '+%T') ---"
      bash -c "$LSST_STACK_SETUP; cd $PWD; export PYTHONPATH=$PWD:\$PYTHONPATH; \
        BUTLER_REPO=embargo python -m ADCNN.linking.ingest_diasource --butler-repo embargo \
        --collection '$COLL' --out $SRC/stack_dets.csv" \
        || echo "WARN: DIA ingest failed -- night will be ADCNN-only"
    else echo "--- [1/4] stack_dets.csv exists, reuse ---"; fi

    # --- 2. merge (rings dropped FIRST) --------------------------------------------------
    if [ ! -s "$SRC/dets_merged.csv" ]; then
      echo "--- [2/4] merge $(date '+%T') ---"
      if [ -s "$SRC/stack_dets.csv" ] && [ -s "$SRC/bright_refcat.parquet" ]; then
        python -m ADCNN.linking.merge_dets --adcnn "$SRC/adcnn_dets_masked.csv" \
          --stack "$SRC/stack_dets.csv" --out "$SRC/dets_merged.csv" \
          --refcat "$SRC/bright_refcat.parquet" || { echo "MERGE FAILED"; exit 1; }
      else
        echo "WARN: no stack catalogue or no refcat -- ADCNN-only for this night"
      fi
    else echo "--- [2/4] dets_merged.csv exists, reuse ---"; fi
    DETS=$SRC/adcnn_dets_masked.csv
    [ -s "$SRC/dets_merged.csv" ] && DETS=$SRC/dets_merged.csv
    echo "dets: $DETS"

    # --- 3. relink ------------------------------------------------------------------------
    if [ ! -s "$SRC/stream_merged/alerts.jsonl" ]; then
      echo "--- [3/4] link (score>=$SCORE_MIN) $(date '+%T') ---"
      mkdir -p "$SRC/stream_merged"
      SC=""; [ -s "$SRC/static_catalog.parquet" ] && SC="--static-catalog $SRC/static_catalog.parquet"
      python -u -m ADCNN.linking.link_2visit --dets "$DETS" --known "$SRC/known.csv" \
        --op-point "$STREAM_OP" --score-min "$SCORE_MIN" --npt 2 --min-epochs 2 --seed-2v chord \
        $SC --train-veto --claim-order preal --rank-by chi2 --link-workers "$LINK_WORKERS" \
        --out "$SRC/stream_merged/tracks.csv" --alerts-out "$SRC/stream_merged/alerts.jsonl" \
        || { echo "LINK FAILED"; exit 1; }
    else echo "--- [3/4] stream_merged/alerts.jsonl exists, reuse ---"; fi

    # --- 4. product -----------------------------------------------------------------------
    echo "--- [4/4] product $(date '+%T') ---"
    mkdir -p "$sd"
    RC=""; [ -s "$SRC/bright_refcat.parquet" ] && RC="--refcat $SRC/bright_refcat.parquet" \
        || echo "WARNING: no refcat -- PROXIMITY RING VETO OFF"
    python -m ADCNN.qa.filter_op --alerts "$SRC/stream_merged/alerts.jsonl" --dets "$DETS" \
      --op "$OP" --out "$sd/_surv.jsonl" $RC || { echo "FILTER FAILED"; exit 1; }
    python -m ADCNN.qa.alert_cutouts --alerts "$sd/_surv.jsonl" --dets "$DETS" \
      --out "$sd/_surv_cutouts.npz" --stamp-px 96 --workers "$WORKERS" || { echo "CUTOUTS FAILED"; exit 1; }
    python -m ADCNN.qa.alert_morphology --alerts "$sd/_surv.jsonl" --cutouts "$sd/_surv_cutouts.npz" \
      --out "$sd/_morph.npz" || { echo "MORPH FAILED"; exit 1; }
    python -m ADCNN.qa.select_clean --alerts "$sd/_surv.jsonl" --morph "$sd/_morph.npz" \
      --cutouts "$sd/_surv_cutouts.npz" --n 999999 --mode rings \
      --out-alerts "$sd/alerts.jsonl" --out-cutouts "$sd/cutouts.npz" || { echo "SELECT FAILED"; exit 1; }
    NA=$(wc -l < "$sd/alerts.jsonl")
    python -m ADCNN.qa.alert_sheets --alerts "$sd/alerts.jsonl" --cutouts "$sd/cutouts.npz" \
      --out-dir "$sd/sheets" --per-sheet 48 --limit "$NA" || { echo "SHEETS FAILED"; exit 1; }
    python -m ADCNN.qa.alert_pairs --alerts "$sd/alerts.jsonl" --cutouts "$sd/cutouts.npz" \
      --out-dir "$sd/pairs" --top-n "$NA" || { echo "PAIRS FAILED"; exit 1; }
    STATIC=$SRC/static_catalog.parquet
    python -m ADCNN.qa.stream_summary --alerts "$sd/alerts.jsonl" --out "$sd/stream_summary.json" \
      $( [ -f "$STATIC" ] && echo "--static-catalog $STATIC" ) || true
    # provenance: how many delivered alerts contain a STACK detection
    python - "$sd/alerts.jsonl" "$DETS" <<'PY' || true
import sys, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
A=[json.loads(l) for l in open(sys.argv[1])]
d=pd.read_csv(sys.argv[2], usecols=lambda c: c in ("ra","dec","visit","src"))
if "src" not in d.columns or not len(A): print("provenance: n/a"); raise SystemExit
S=d[d.src=="stack"]
def r2u(ra,dec):
    r=np.radians(np.asarray(ra,float)); q=np.radians(np.asarray(dec,float))
    return np.stack([np.cos(q)*np.cos(r),np.cos(q)*np.sin(r),np.sin(q)],-1)
t={int(v):cKDTree(r2u(g.ra.values,g.dec.values)) for v,g in S.groupby("visit")}
tol=2*np.sin(np.radians(1.0/3600)/2); n=0
for a in A:
    for e in a["epochs"]:
        k=t.get(int(e["visit"]))
        if k is None: continue
        if k.query(r2u(np.array([e["ra"]]),np.array([e["dec"]])),k=1)[0][0]<tol: n+=1; break
print(f"provenance: {n}/{len(A)} delivered alerts contain a STACK detection ({100*n/len(A):.1f}%)")
PY
    rm -f "$sd/_surv.jsonl" "$sd/_surv_cutouts.npz" "$sd/_surv_cutouts_meta.json" "$sd/_morph.npz" \
          "$sd/cutouts.npz" "$sd/cutouts_meta.json"
    echo "===== $N DONE ($NA clean alerts) $(date '+%T') ====="
  } > "$L" 2>&1
  if grep -q "$N DONE" "$L"; then
    echo "[$N] ok: $(grep -oE 'DONE \([0-9]+ clean' $L | grep -oE '[0-9]+') alerts | $(grep -oE 'provenance: [0-9]+/[0-9]+[^)]*\)' $L | tail -1)"
  else echo "[$N] FAILED -- see $L"; fi
done
echo "rebuild_merged done $(date '+%F %T')"
