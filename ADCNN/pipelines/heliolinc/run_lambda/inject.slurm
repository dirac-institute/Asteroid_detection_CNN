#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J lam_inject
#SBATCH -o %x_%A_%a.log
set -eo pipefail
# Re-time + faint-SNR injection for one off-ecliptic field (array index = field k). Light CPU.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
RUN=${RUN:?set RUN}
NOBJ=${NOBJ:-300}
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
k=$SLURM_ARRAY_TASK_ID
M=$RUN/manifest_$k.csv
[ -f "$M" ] || { echo "no manifest $M -- skip"; exit 0; }
[ -f "$RUN/inject_$k.csv" ] && { echo "inject_$k exists -- skip"; exit 0; }
python ADCNN/pipelines/heliolinc/retime_cadence.py --manifest "$M" --out "$RUN/retime_$k.csv" --seed $((2026+k))
python ADCNN/pipelines/heliolinc/sim_orbits.py --manifest "$M" --retime-map "$RUN/retime_$k.csv" \
  --out-inject "$RUN/inject_$k.csv" --out-truth "$RUN/truth_$k.csv" --n-objects "$NOBJ" --seed $((7000+k))
echo "INJECT DONE $k"
