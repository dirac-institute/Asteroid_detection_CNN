#!/bin/bash
# Hunt for asteroids via TRAIL-TRACKLETS: each trailed detection -> one tracklet (no >=2/night
# pairing) -> heliolinc -> link_refine -> crossmatch to known -> CONFIRMED + NEW candidates.
# Usage: hunt_new.sh <run_dir>   (expects <run_dir>/adcnn_dets.csv with ra0,dec0,ra1,dec1; known.csv optional)
set -euo pipefail
RUN=${1:?usage: hunt_new.sh <run_dir>}
DETS=${DETS:-$RUN/adcnn_dets_veres.csv}   # Veres-measured detections (precise endpoints)
LENDB_MIN=${LENDB_MIN:-6}     # de-biased trail length cut (px); ~6px ≈ 1 deg/day -> trailed/fast movers
HELIODIST=${HELIODIST:-heliohypo_all.txt}  # hypothesis grid (use a coarse one for tractable runtime)
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cp -n "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" \
      "$HL/run_disco/heliohypo_all.txt" "$HL/run_disco/colformat.txt" "$RUN/" 2>/dev/null || true

# 1) trail -> tracklets (both head/tail orderings; pairdets.csv + pairs.txt)
python "$HL/trail_tracklets.py" --dets "$DETS" --earth "$RUN/Earth1day2020s_02a.txt" \
  --out "$RUN" --lendb-min "$LENDB_MIN"
MJDREF=$(python -c "import pandas as pd;print(round(pd.read_csv('$DETS').mjd.median(),3))")
cd "$RUN"
echo "=== heliolinc on trail-tracklets | mjdref=$MJDREF ==="
echo "grid points: $(($(wc -l < "$HELIODIST")-1)) | tracklets: $(grep -c '^T' pairs.txt)"
"$BIN/heliolinc" -dets pairdets.csv -pairs pairs.txt -mjd "$MJDREF" \
  -obspos Earth1day2020s_02a.txt -heliodist "$HELIODIST" \
  -npt 3 -minobsnights 3 -mintimespan 0.5 -out hl_clusters.csv -outsum hl_summary.csv
printf "hl_clusters.csv hl_summary.csv\n" > lflist.txt
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"

# 2) crossmatch to known -> CONFIRMED + NEW
if [ -f "$RUN/known.csv" ]; then
  python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/known.csv" --tol-arcsec 3.0 --tol-day 0.02
fi
echo "HUNT DONE -> $RUN"
