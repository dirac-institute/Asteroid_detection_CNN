#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH -c 64
#SBATCH --mem=192G
#SBATCH -t 6:00:00
#SBATCH -J lam_stack
#SBATCH -o %x_%A_%a.log
set -eo pipefail
# Stack 5-sigma SourceDetection on the inline-injected diffims for one field (array index = k). CPU, LSST
# stack env. Independent of ADCNN (only needs inject_$k.csv) -> runs in parallel with detect_ada.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
RUN=${RUN:?set RUN}
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "$REPO"
k=$SLURM_ARRAY_TASK_ID
M=$RUN/manifest_$k.csv
[ -f "$M" ] || { echo "no manifest $M -- skip"; exit 0; }
[ -f "$RUN/stack_dets_$k.csv" ] && { echo "stack_$k exists -- skip"; exit 0; }
python ADCNN/pipelines/heliolinc/stack_detect.py --manifest "$M" \
  --inject "$RUN/inject_$k.csv" --retime-map "$RUN/retime_$k.csv" \
  --inject-panels-only --out "$RUN/stack_dets_$k.csv" --workers 64
echo "STACK DONE $k"
