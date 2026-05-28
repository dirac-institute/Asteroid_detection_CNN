#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH -t 2:00:00
#SBATCH -J hl_neo_mn
#SBATCH -o %x_%j.log
set -eo pipefail
# MULTI-NODE NEO hunt: same computation as hunt_neo.slurm but the independent grid points are spread
# across 4 nodes (~4x faster, no accuracy change). Reuses existing pairdets/pairs. Each node writes
# small (clustrad-bounded) clusters to SHARED disk; head node then link_refines across all + crossmatch.
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
RUN=$HL/NEO_large
BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

rm -rf "$RUN/clusters_mn"; mkdir -p "$RUN/clusters_mn"
echo "=== launching 4-node heliolinc over NEO grid ($(($(wc -l < $RUN/heliohypo_neo.txt)-1)) pts), reuse tracklets ==="
NSHARD=90 CLUSTRAD=16000 NPT=3 MINNIGHTS=2 MINTIMESPAN=0.05 \
  srun --ntasks=4 --ntasks-per-node=1 bash "$HL/node_worker_neo.sh"
echo "=== all nodes done; link_refine over $(ls "$RUN"/clusters_mn/hl_clusters_*.csv 2>/dev/null | wc -l) shards ==="

cd "$RUN"
: > lflist.txt
for c in clusters_mn/hl_clusters_*.csv; do
  s=$(basename "$c" .csv); s=${s#hl_clusters_}
  [ -s "$c" ] && [ -s "clusters_mn/hl_summary_${s}.csv" ] && echo "$c clusters_mn/hl_summary_${s}.csv" >> lflist.txt
done
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"

echo "--- crossmatch vs known NEOs (recovery) ---"
python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/neo_truth.csv" --tol-arcsec 3.0 --tol-day 0.02 || true
cp -f "$RUN/confirmed.csv" "$RUN/recovered_neo.csv" 2>/dev/null || true
echo "--- classify (KNOWN / RECOVERED-NEO / NEW-clean) ---"
python "$HL/classify_tracks.py" --run "$RUN" --known "$RUN/known.csv" --missed "$RUN/neo_truth.csv" \
  --maxposrms 2000 --minnights 3 || true
echo "HL_NEO_MN DONE"
