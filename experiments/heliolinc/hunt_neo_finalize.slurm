#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J hl_neo_fin
#SBATCH -o %x_%j.log
set -eo pipefail
# After the array finishes: link_refine over all nodes' clusters + crossmatch vs known NEOs (recovery)
# + classify (KNOWN / RECOVERED-NEO / NEW-clean).
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
RUN=$HL/NEO_large; BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$RUN"
: > lflist.txt
for c in clusters_mn/hl_clusters_*.csv; do
  s=$(basename "$c" .csv); s=${s#hl_clusters_}
  # only include shards whose cluster file actually has cluster rows (>1 line) -> link_refine skips empties cleanly
  [ "$(wc -l < "$c")" -gt 1 ] && [ -s "clusters_mn/hl_summary_${s}.csv" ] && echo "$c clusters_mn/hl_summary_${s}.csv" >> lflist.txt
done
echo "link_refine over $(wc -l < lflist.txt) shard files"
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"
echo "--- recovery: crossmatch vs known NEOs ---"
python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/neo_truth.csv" --tol-arcsec 3.0 --tol-day 0.02 || true
cp -f "$RUN/confirmed.csv" "$RUN/recovered_neo.csv" 2>/dev/null || true
echo "--- classify (KNOWN / RECOVERED-NEO / NEW-clean) ---"
python "$HL/classify_tracks.py" --run "$RUN" --known "$RUN/known.csv" --missed "$RUN/neo_truth.csv" \
  --maxposrms 2000 --minnights 3 || true
echo "HL_NEO_FINALIZE DONE"
