#!/bin/bash
# Link an ADCNN detection catalog with HelioLinC and crossmatch to the known-object catalog.
# Usage: link_and_match.sh <run_dir>   (expects <run_dir>/adcnn_dets.csv ; uses known.csv if present)
# Uses the linking config validated on truth (full grid + mjd at window midpoint).
set -euo pipefail
RUN=${1:?usage: link_and_match.sh <run_dir>}
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cp -n "$HL/run_truth/colformat.txt" "$HL/run_truth/Earth1day2020s_02a.txt" "$HL/run_truth/ObsCodes.txt" "$RUN/" 2>/dev/null || true
cp -n "$HL/run_2wk/heliohypo_all.txt" "$RUN/" 2>/dev/null || true

# reference MJD = median detection epoch (HelioLinC clusters propagate to this)
MJDREF=$(python -c "import pandas as pd;print(round(pd.read_csv('$RUN/adcnn_dets.csv').mjd.median(),3))")
echo "=== linking $(wc -l < $RUN/adcnn_dets.csv) detections | mjdref=$MJDREF ==="
cd "$RUN"
"$BIN/make_tracklets" -dets adcnn_dets.csv -earth Earth1day2020s_02a.txt -obscode ObsCodes.txt \
  -colformat colformat.txt -pairdets pairdets.csv -pairs pairs.txt -outimgs imgs.txt \
  -maxtime 3.0 -mintime 0.0 -maxGCR 2.0 -mintrkpts 2 -maxvel 2.0 -minvel 0.0
echo "tracklets: $(($(wc -l < pairs.txt)))"
"$BIN/heliolinc" -dets pairdets.csv -pairs pairs.txt -mjd "$MJDREF" \
  -obspos Earth1day2020s_02a.txt -heliodist heliohypo_all.txt \
  -npt 3 -minobsnights 3 -mintimespan 0.5 -out hl_clusters.csv -outsum hl_summary.csv
printf "hl_clusters.csv hl_summary.csv\n" > lflist.txt
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"

if [ -f "$RUN/known.csv" ]; then
  python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/known.csv" --tol-arcsec 3.0 --tol-day 0.02
fi
echo "LINK+MATCH DONE -> $RUN"
