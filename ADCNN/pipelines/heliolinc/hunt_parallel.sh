#!/bin/bash
# GRID-PARALLEL trail-tracklet hunt: each hypothesis grid point is independent, so shard the
# heliocentric grid across N cores, run N heliolinc processes in parallel on the SAME tracklets,
# then let link_refine merge all shards (-lflist accepts many cluster/summary pairs). Makes the
# full fine grid tractable. Usage: hunt_parallel.sh <run_dir>
set -euo pipefail
RUN=${1:?usage: hunt_parallel.sh <run_dir>}
DETS=${DETS:-$RUN/adcnn_dets_veres.csv}
LENDB_MIN=${LENDB_MIN:-6}
HELIODIST=${HELIODIST:-heliohypo_all.txt}     # relative to RUN
NSHARD=${NSHARD:-96}
MINNIGHTS=${MINNIGHTS:-3}    # fast NEOs cross a single field in ~2 nights -> use 2 for NEO hunts
NPT=${NPT:-3}
MINTIMESPAN=${MINTIMESPAN:-0.5}
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc
BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cp -n "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" \
      "$HL/run_disco/heliohypo_all.txt" "$HL/run_disco/colformat.txt" "$RUN/" 2>/dev/null || true
cp -n "$HL/NEO_large/heliohypo_neo.txt" "$RUN/" 2>/dev/null || true   # NEO-targeted grid (default)

# 1) trail -> tracklets
python "$HL/trail_tracklets.py" --dets "$DETS" --earth "$RUN/Earth1day2020s_02a.txt" \
  --out "$RUN" --lendb-min "$LENDB_MIN"
MJDREF=$(python -c "import pandas as pd;print(round(pd.read_csv('$DETS').mjd.median(),3))")
cd "$RUN"
NG=$(($(wc -l < "$HELIODIST")-1)); NT=$(grep -c '^T' pairs.txt)
echo "=== grid-parallel heliolinc | grid=$NG pts x $NSHARD shards | tracklets=$NT | mjdref=$MJDREF ==="

# 2) shard the grid (header prepended to each chunk)
rm -f grid_chunk_* grid_body.tmp hl_clusters_*.csv hl_summary_*.csv
HDR=$(head -1 "$HELIODIST")
tail -n +2 "$HELIODIST" > grid_body.tmp          # real file so split -n can size it
split -n l/$NSHARD -d -a 3 grid_body.tmp grid_chunk_
for f in grid_chunk_*; do sed -i "1i $HDR" "$f"; done
rm -f grid_body.tmp

# 3) run N heliolinc in parallel, one per grid chunk
pids=()
for f in grid_chunk_*; do
  s=${f#grid_chunk_}
  "$BIN/heliolinc" -dets pairdets.csv -pairs pairs.txt -mjd "$MJDREF" \
    -obspos Earth1day2020s_02a.txt -heliodist "$f" \
    -clustrad "${CLUSTRAD:-100000}" \
    -npt "$NPT" -minobsnights "$MINNIGHTS" -mintimespan "$MINTIMESPAN" \
    -out "hl_clusters_${s}.csv" -outsum "hl_summary_${s}.csv" >"hl_${s}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} heliolinc shards; waiting..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
# A failed shard means part of the hypothesis grid was never searched -> link_refine would report a
# partial result as if complete, silently losing any object whose grid point was in the dead shard.
if [ "$fail" -gt 0 ]; then echo "ABORT: $fail/${#pids[@]} heliolinc shards failed -> partial grid, refusing to link"; exit 1; fi
echo "shards done (failures=$fail); clusters across shards: $(cat hl_clusters_*.csv 2>/dev/null | grep -vc '^#ptct' || echo 0)"

# 4) link_refine over ALL shard outputs (merges) -> lr.csv
: > lflist.txt
for s in $(ls hl_summary_*.csv | sed 's/hl_summary_//;s/.csv//'); do
  [ -s "hl_clusters_${s}.csv" ] && echo "hl_clusters_${s}.csv hl_summary_${s}.csv" >> lflist.txt
done
echo "link_refine over $(wc -l < lflist.txt) shard files"
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms "${MAXRMS:-100000}" -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"

# 5) crossmatch -> CONFIRMED + NEW
if [ -f "$RUN/known.csv" ]; then
  python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/known.csv" --tol-arcsec 3.0 --tol-day 0.02
fi
echo "PARALLEL HUNT DONE -> $RUN"
