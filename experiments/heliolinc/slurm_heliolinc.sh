#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -J hl_disco
#SBATCH -o %x_%j.log
set -euo pipefail
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
BIN=$HL/heliolinc2/src
RUN=$HL/run_disco
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$RUN"
MJDREF=$(python -c "import pandas as pd;print(round(pd.read_csv('adcnn_dets.csv').mjd.median(),3))")
echo "=== heliolinc: $(($(wc -l < pairs.txt))) tracklets, grid $(($(wc -l < heliohypo_mb.txt)-1)), mjd=$MJDREF ==="
time "$BIN/heliolinc" -dets pairdets.csv -pairs pairs.txt -mjd "$MJDREF" \
  -obspos Earth1day2020s_02a.txt -heliodist heliohypo_mb.txt \
  -npt 3 -minobsnights 3 -mintimespan 0.5 -out hl_clusters.csv -outsum hl_summary.csv 2>&1 | tail -3
printf "hl_clusters.csv hl_summary.csv\n" > lflist.txt
"$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv 2>&1 | tail -2
echo "refined tracks: $(($(wc -l < lr_rms.csv)-1))"
python "$HL/crossmatch.py" --run "$RUN" --known "$RUN/known.csv" --tol-arcsec 3.0 --tol-day 0.02
echo "HELIOLINC DISCO DONE"
