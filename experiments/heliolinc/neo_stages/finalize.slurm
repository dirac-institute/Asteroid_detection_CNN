#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J neo_finalize
#SBATCH -o %x_%j.log
set -eo pipefail
# Stage FINALIZE: link_refine over all nodes' clusters -> refined tracks; crossmatch vs neo_truth.csv
# (NEO recovery) and classify vs known.csv (KNOWN / RECOVERED / NEW-clean). Outputs recovered_neo.csv
# + classified.csv. Only feeds link_refine cluster files that actually contain clusters (>1 line).
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
eval "$TORCH_ENV"
BIN=$HL/heliolinc2/src
cd "$RUN"
: > lflist.txt
for c in clusters_mn/hl_clusters_*.csv; do
  s=$(basename "$c" .csv); s=${s#hl_clusters_}
  [ "$(wc -l < "$c")" -gt 1 ] && [ -s "clusters_mn/hl_summary_${s}.csv" ] && echo "$c clusters_mn/hl_summary_${s}.csv" >> lflist.txt
done
echo "link_refine over $(wc -l < lflist.txt) non-empty shard files"
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"
echo "--- NEO recovery: crossmatch refined tracks vs the known NEOs (neo_truth.csv) ---"
python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/neo_truth.csv" \
  --tol-arcsec "$XM_TOL_ARCSEC" --tol-day "$XM_TOL_DAY" || true
cp -f "$RUN/confirmed.csv" "$RUN/recovered_neo.csv" 2>/dev/null || true
echo "--- classify (KNOWN / RECOVERED-NEO / NEW-clean) ---"
python "$HL/classify_tracks.py" --run "$RUN" --known "$RUN/known.csv" --missed "$RUN/neo_truth.csv" \
  --maxposrms "$MAXPOSRMS" --minnights 3 || true
echo "NEO FINALIZE DONE -> $RUN/recovered_neo.csv, classified.csv"
