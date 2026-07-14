#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 6:00:00
#SBATCH -J lam_sweep
#SBATCH -o outputs/logs/%x_%j.log
set -eo pipefail
# Final/preliminary lambda(S) sweep -> S*. Runs on a roma node (no harness time limit, 64 cores, 256G);
# sweep_S parallelises fields, each field's orbit-solve eval in its own worker. Reads whatever
# adcnn_dets_masked_*.csv exist (so it works for a partial/preliminary set too).
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
RUN=${RUN:?set RUN}
SCORES=${SCORES:-0.60 0.65 0.70 0.75 0.80 0.85 0.90}
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
python -m ADCNN.pipelines.heliolinc.sweep_S --dir "$RUN" --scores $SCORES --len-db-max ${LENMAX:-1000000000} --workers 60 --out "$RUN"
echo "SWEEP_DONE -> $RUN/lambda_vs_S.csv $RUN/s_star.csv"
