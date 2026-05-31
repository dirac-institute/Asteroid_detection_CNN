#!/bin/bash
# STANDARD-PAIRING hunt: seed tracklets with Rubin's make_tracklets (pairs detections ACROSS visits
# within a night; velocity from the minutes-to-hours time baseline) instead of trail endpoints.
# Trail-endpoint seeding (trail_tracklets.py) fails on this field's marginally-trailed (~3-6px,
# ~0.5 deg/day) objects: the endpoint separation is at the noise floor -> garbage velocity -> no
# valid orbit seed (only spurious high-RMS clusters survive). make_tracklets is the proven path
# (run_diasrc -> 314 confirmed). Same grid-parallel heliolinc + link_refine + crossmatch tail.
# Usage: hunt_standard.sh <run_dir>
set -eo pipefail
RUN=${1:?usage: hunt_standard.sh <run_dir>}
DETS=${DETS:-$RUN/adcnn_dets_clean.csv}
HELIODIST=${HELIODIST:-heliohypo_all.txt}     # relative to RUN
NSHARD=${NSHARD:-96}
MINNIGHTS=${MINNIGHTS:-2}
NPT=${NPT:-3}
MINTIMESPAN=${MINTIMESPAN:-0.05}
# make_tracklets velocity gates (deg/day, GCR arcsec, hours) -- match run_diasrc proven config.
MAXVEL=${MAXVEL:-2.0}
MAXGCR=${MAXGCR:-2.0}
MAXTIME=${MAXTIME:-3.0}
CLUSTRAD=${CLUSTRAD:-16000}    # heliolinc 6D clustering radius (km); tighter -> fewer spurious clusters
CLUSTERS_DIR=${CLUSTERS_DIR:-$RUN}   # where to write bulky per-shard clusters (point at node-local scratch)
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc
BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cp -n "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" \
      "$HL/run_disco/heliohypo_all.txt" "$RUN/" 2>/dev/null || true

# 0) slim detection file with a colformat that matches it exactly (the cleaned catalog inserts
#    ra0/dec0/ra1/dec1, which shifts mag/band/obscode -> a fixed 1..7 colformat would be wrong).
python - "$DETS" "$RUN/std_dets.csv" "$RUN/std_colformat.txt" <<'PY'
import sys, pandas as pd
src, out, cf = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(src)
d = d[["detid","mjd","ra","dec","mag","band","obscode"]].copy()
d.to_csv(out, index=False)
open(cf,"w").write("IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
print(f"[std] {len(d)} detections over {d.mjd.astype(int).nunique()} nights -> {out}")
PY

cd "$RUN"
# 1) make_tracklets: cross-visit pairing (robust velocity from the time baseline).
#    SKIP_TRACKLETS=1 reuses existing pairdets.csv/pairs.txt (e.g. a rebalanced re-run).
if [ "${SKIP_TRACKLETS:-0}" = "1" ] && [ -s pairdets.csv ] && [ -s pairs.txt ]; then
  echo "SKIP_TRACKLETS: reusing existing pairdets.csv/pairs.txt"
else
  "$BIN/make_tracklets" -dets std_dets.csv -earth Earth1day2020s_02a.txt -obscode ObsCodes.txt \
    -colformat std_colformat.txt -pairdets pairdets.csv -pairs pairs.txt -outimgs imgs.txt \
    -maxtime "$MAXTIME" -mintime 0.0 -maxGCR "$MAXGCR" -mintrkpts 2 -maxvel "$MAXVEL" -minvel 0.0
fi
MJDREF=$(python -c "import pandas as pd;print(round(pd.read_csv('std_dets.csv').mjd.median(),3))")
NG=$(($(wc -l < "$HELIODIST")-1)); NT=$(grep -c '^T' pairs.txt || echo 0)
echo "=== grid-parallel heliolinc | grid=$NG pts x $NSHARD shards | tracklets=$NT | mjdref=$MJDREF ==="

# 2) shard the grid ROUND-ROBIN (interleave grid points across shards) so the expensive low-r/NEO
#    hypotheses are spread evenly -> all shards finish together (contiguous split piled them into a
#    few shards that then timed out, losing their clusters).
mkdir -p "$CLUSTERS_DIR"
rm -f "$CLUSTERS_DIR"/grid_chunk_* "$CLUSTERS_DIR"/hl_clusters_*.csv "$CLUSTERS_DIR"/hl_summary_*.csv
HDR=$(head -1 "$HELIODIST")
tail -n +2 "$HELIODIST" | awk -v n="$NSHARD" -v dir="$CLUSTERS_DIR" -v hdr="$HDR" '
  BEGIN{for(i=0;i<n;i++){f=sprintf("%s/grid_chunk_%03d",dir,i); print hdr > f}}
  {f=sprintf("%s/grid_chunk_%03d",dir,(NR-1)%n); print >> f}'
PD=$PWD/pairdets.csv; PR=$PWD/pairs.txt; OB=$PWD/Earth1day2020s_02a.txt

# 3) parallel heliolinc, one per grid chunk (same tracklets); -clustrad controls spurious-cluster volume
echo "heliolinc clustrad=$CLUSTRAD | clusters -> $CLUSTERS_DIR"
pids=()
for f in "$CLUSTERS_DIR"/grid_chunk_*; do
  s=${f##*grid_chunk_}
  "$BIN/heliolinc" -dets "$PD" -pairs "$PR" -mjd "$MJDREF" \
    -obspos "$OB" -heliodist "$f" -clustrad "$CLUSTRAD" \
    -npt "$NPT" -minobsnights "$MINNIGHTS" -mintimespan "$MINTIMESPAN" \
    -out "$CLUSTERS_DIR/hl_clusters_${s}.csv" -outsum "$CLUSTERS_DIR/hl_summary_${s}.csv" >"$CLUSTERS_DIR/hl_${s}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} heliolinc shards; waiting..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "shards done (failures=$fail); cluster volume: $(du -sh "$CLUSTERS_DIR"/hl_clusters_*.csv 2>/dev/null | tail -1 | cut -f1)"

# 4) link_refine over all shards
: > lflist.txt
for s in $(ls "$CLUSTERS_DIR"/hl_summary_*.csv 2>/dev/null | sed 's#.*hl_summary_##;s/.csv//'); do
  [ -s "$CLUSTERS_DIR/hl_clusters_${s}.csv" ] && echo "$CLUSTERS_DIR/hl_clusters_${s}.csv $CLUSTERS_DIR/hl_summary_${s}.csv" >> lflist.txt
done
echo "link_refine over $(wc -l < lflist.txt) shard files"
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"

# 5) crossmatch vs catalogued knowns AND vs Rubin-MISSED objects (the discovery target)
if [ -f "$RUN/known.csv" ]; then
  echo "--- crossmatch vs KNOWN (catalogued) ---"
  python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/known.csv" --tol-arcsec 3.0 --tol-day 0.02
fi
if [ -f "$RUN/missed_truth.csv" ]; then
  echo "--- crossmatch vs Rubin-MISSED (ss_object_unassociated) ---"
  python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/missed_truth.csv" --tol-arcsec 3.0 --tol-day 0.02
  cp -f "$RUN/confirmed.csv" "$RUN/recovered_missed.csv" 2>/dev/null || true
fi
echo "PARALLEL HUNT DONE -> $RUN"
