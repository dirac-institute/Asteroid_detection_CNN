#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J neo_tracklets
#SBATCH -o %x_%j.log
set -eo pipefail
# Stage TRACKLETS: ADCNN dets -> pairdets.csv + pairs.txt for the grid-parallel HelioLinC link stage.
#   TRACKMODE=pair  : make_tracklets cross-visit pairing (>=2 dets/night).
#   TRACKMODE=trail : trail_tracklets one-trail=one-tracklet (single-coverage nights; fast movers).
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
eval "$TORCH_ENV"
BIN=$HL/heliolinc2/src
cd "$RUN"
cp -nf "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" "$RUN/" 2>/dev/null || true

if [ "${TRACKMODE:-pair}" = trail ]; then
  echo "[tracklets] TRAIL mode (one trail=one tracklet, lendb_min=$LENDB_MIN) -- needs accurate endpoints (#4 Veres)"
  python "$HL/trail_tracklets.py" --dets "$RUN/adcnn_dets.csv" --earth Earth1day2020s_02a.txt \
    --out "$RUN" --lendb-min "$LENDB_MIN"
else
  echo "[tracklets] PAIR mode (make_tracklets cross-visit pairing)"
  python - <<PY
import pandas as pd
d=pd.read_csv("$RUN/adcnn_dets.csv")
d=d[["detid","mjd","ra","dec","mag","band","obscode"]]
d.to_csv("$RUN/std_dets.csv",index=False)
open("$RUN/std_colformat.txt","w").write("IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
print(f"[tracklets] {len(d)} ADCNN dets over {d.mjd.astype(int).nunique()} nights")
PY
  "$BIN/make_tracklets" -dets std_dets.csv -earth Earth1day2020s_02a.txt -obscode ObsCodes.txt \
    -colformat std_colformat.txt -pairdets pairdets.csv -pairs pairs.txt -outimgs imgs.txt \
    -maxtime "$MAXTIME" -mintime 0.0 -maxGCR "$MAXGCR" -mintrkpts 2 -maxvel "$MAXVEL" -minvel 0.0
fi
echo "NEO TRACKLETS DONE ($TRACKMODE) -> $RUN/pairdets.csv (tracklets=$(grep -c '^T' pairs.txt || echo ?))"
