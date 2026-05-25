#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 96
#SBATCH --mem=256G
#SBATCH -t 3:00:00
#SBATCH -J neo3_link
#SBATCH -o %x_%j.log
set -eo pipefail
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_pipeline/config.sh"
eval "$TORCH_ENV"
cd "$HL"

# Stage 3a: FP-cleaning (the step the machinery still needs; cheap cuts + hook in clean_fp.py)
python "$PIPE/clean_fp.py" --in "$RUN/adcnn_dets_veres.csv" --out "$RUN/adcnn_dets_clean.csv" \
  --score-min "$CLEAN_SCORE_MIN" --rchisq-max "$CLEAN_RCHISQ_MAX" --lendb-min "$CLEAN_LENDB_MIN"

# Stage 3b: trail->tracklets -> grid-parallel HelioLinC -> link_refine -> crossmatch -> CONFIRMED/NEW
DETS="$RUN/adcnn_dets_clean.csv" LENDB_MIN="$CLEAN_LENDB_MIN" \
HELIODIST="$HELIODIST" NSHARD="$NSHARD" MINNIGHTS="$MINNIGHTS" NPT="$NPT" MINTIMESPAN="$MINTIMESPAN" \
  bash "$HL/hunt_parallel.sh" "$RUN"
echo "NEO STAGE3 LINK DONE"
