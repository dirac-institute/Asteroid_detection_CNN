#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 96
#SBATCH --mem=256G
#SBATCH -t 4:00:00
#SBATCH -J hl_missed
#SBATCH -o %x_%j.log
set -eo pipefail
# Can ADCNN+HelioLinC recover Rubin-MISSED objects (ss_object_unassociated)? Link ADCNN-ONLY
# detections in run_neo_field and crossmatch tracks vs missed_truth.csv (96 detected+linkable
# missed objects) AND known.csv. Main-belt grid + tight clustrad + node-local scratch keep the
# FP-driven cluster volume tractable. Reuses on-disk ADCNN detections (no GPU re-run).
HL=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc
RUN=$HL/run_neo_field
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# 1) ADCNN-ONLY detection catalog (score>=0.5, no trail cut) -> isolates ADCNN's own recovery
python - <<PY
import pandas as pd
d=pd.read_csv("$RUN/adcnn_dets_veres.csv")
d=d[d.score_rf>=0.5][["detid","mjd","ra","dec","mag","band","obscode"]]
d.to_csv("$RUN/adcnn_only.csv",index=False)
print(f"[adcnn-only] {len(d)} detections over {d.mjd.astype(int).nunique()} nights")
PY

# 2) support files: main-belt hypothesis grid (objects are main-belt distance) + ephem/obscode
cp -f "$HL/run_diasrc/heliohypo_mb.txt" "$RUN/heliohypo_mb.txt"
cp -nf "$HL/run_disco/Earth1day2020s_02a.txt" "$HL/run_disco/ObsCodes.txt" "$RUN/" 2>/dev/null || true

# 3) hunt: make_tracklets + grid-parallel heliolinc(mb grid, tight clustrad) + link_refine + crossmatch
SCRATCH=/lscratch/$USER/hl_missed_$SLURM_JOB_ID
DETS="$RUN/adcnn_only.csv" HELIODIST="heliohypo_mb.txt" CLUSTRAD=16000 \
NSHARD=90 MINNIGHTS=2 NPT=3 MINTIMESPAN=0.05 MAXVEL=2.0 MAXGCR=2.0 MAXTIME=3.0 \
CLUSTERS_DIR="$SCRATCH" \
  bash "$HL/hunt_standard.sh" "$RUN"
rm -rf "$SCRATCH" 2>/dev/null || true
echo "HL_MISSED DONE"
