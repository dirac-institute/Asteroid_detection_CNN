#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J neo_tracklets
#SBATCH -o %x_%j.log
set -eo pipefail
# Stage TRACKLETS: ADCNN-only slim catalog (score>=RF_THR, all trail lengths kept) -> make_tracklets
# (cross-visit pairing) -> pairdets.csv + pairs.txt for the grid-parallel HelioLinC link stage.
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
eval "$TORCH_ENV"
BIN=$HL/heliolinc2/src
cd "$RUN"
python - <<PY
import pandas as pd
d=pd.read_csv("$RUN/adcnn_dets.csv")
d=d[["detid","mjd","ra","dec","mag","band","obscode"]]
d.to_csv("$RUN/std_dets.csv",index=False)
open("$RUN/std_colformat.txt","w").write("IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n")
print(f"[tracklets] {len(d)} ADCNN dets over {d.mjd.astype(int).nunique()} nights")
PY
cp -nf "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" "$RUN/" 2>/dev/null || true
"$BIN/make_tracklets" -dets std_dets.csv -earth Earth1day2020s_02a.txt -obscode ObsCodes.txt \
  -colformat std_colformat.txt -pairdets pairdets.csv -pairs pairs.txt -outimgs imgs.txt \
  -maxtime "$MAXTIME" -mintime 0.0 -maxGCR "$MAXGCR" -mintrkpts 2 -maxvel "$MAXVEL" -minvel 0.0
echo "NEO TRACKLETS DONE -> $RUN/pairdets.csv (tracklets=$(grep -c '^T' pairs.txt || echo ?))"
