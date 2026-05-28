#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 96
#SBATCH --mem=256G
#SBATCH -t 4:00:00
#SBATCH -J hl_neo
#SBATCH -o %x_%j.log
set -eo pipefail
# NEO recovery: link ADCNN-only detections over the wide NEO strip and crossmatch vs the 86 known
# NEO-rate objects (neo_truth.csv). NEO-tuned: high maxvel + NEO-distance grid (r<1.6 AU, no main
# belt) + tight clustrad + node-local scratch.
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
RUN=$HL/NEO_large
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# ADCNN-only slim catalog (score>=0.5 already; keep ALL trail lengths — NEOs must not be cut)
python - <<PY
import pandas as pd
d=pd.read_csv("$RUN/adcnn_dets.csv")
d=d[["detid","mjd","ra","dec","mag","band","obscode"]]
d.to_csv("$RUN/adcnn_only.csv",index=False)
print(f"[adcnn-only] {len(d)} dets over {d.mjd.astype(int).nunique()} nights")
PY
cp -nf "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" "$RUN/" 2>/dev/null || true
# recovery target = the 86 known NEOs (hunt_standard crossmatches a file named missed_truth.csv)
cp -f "$RUN/neo_truth.csv" "$RUN/missed_truth.csv"

SCRATCH=/lscratch/$USER/hl_neo_$SLURM_JOB_ID
# reuse existing pairdets/pairs (make_tracklets already done); round-robin sharding balances load
DETS="$RUN/adcnn_only.csv" HELIODIST="heliohypo_neo.txt" CLUSTRAD=16000 SKIP_TRACKLETS=1 \
NSHARD=90 MINNIGHTS=2 NPT=3 MINTIMESPAN=0.05 MAXVEL=5.0 MAXGCR=2.5 MAXTIME=3.0 \
CLUSTERS_DIR="$SCRATCH" \
  bash "$HL/hunt_standard.sh" "$RUN"

# extra: full 3-way classify (KNOWN / RECOVERED-NEO / NEW-clean) for new-candidate flagging
python "$HL/classify_tracks.py" --run "$RUN" --known "$RUN/known.csv" --missed "$RUN/neo_truth.csv" \
  --maxposrms 2000 --minnights 3 || true
rm -rf "$SCRATCH" 2>/dev/null || true
echo "HL_NEO DONE"
