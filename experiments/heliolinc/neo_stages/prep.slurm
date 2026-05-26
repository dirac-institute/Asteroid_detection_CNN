#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=kipac:kipac
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J neo_prep
#SBATCH -o %x_%j.log
set -eo pipefail
# Stage PREP (lsst env): NEO hypothesis grid + targeted diffim manifest (panels along known-NEO
# ephemeris arcs) + known-object catalog + neo_truth.csv (the recovery targets).
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
eval "$LSST_ENV"
cd "$HL"
python make_neo_grid.py --src "$GRID_SRC" --rmax "$RMAX" --out "$RUN/$HELIODIST"
python targeted_neo_manifest.py --ra0 "$RA0" --ra1 "$RA1" --dec0 "$DEC0" --dec1 "$DEC1" \
  --day-start "$DAY_START" --day-end "$DAY_END" --min-rate "$MIN_RATE" --run "$RUN"
python build_known_catalog.py --manifest "$RUN/manifest.csv" --out "$RUN/known.csv"
echo "NEO PREP DONE -> $RUN (manifest.csv, neo_truth.csv, known.csv, $HELIODIST)"
