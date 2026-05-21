#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-gen-realistic
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_gen_realistic_%j.out
set -eo pipefail
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO/ADCNN/data/dataset_creation"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/group/rubin/sw/loadLSST.bash
setup lsst_distrib
OUT="$REPO/DATA_DIFFIM_realistic"
mkdir -p "$OUT"
echo "=== full realistic dataset generation === $(date -Is)"
srun python3 -u simulate_inject_diffim.py \
  --repo dp2_prep \
  --collections "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3" "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2" \
  --stage3-collection "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3" \
  --skymap lsst_cells_v2 \
  --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-64}" \
  --train-test-split 0.94117 \
  --random-subset 850 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 2 --mag-max 8 --mag-mode snr \
  --beta-min 0 --beta-max 180 \
  --number 20 \
  --stack-detection-threshold 5.0 \
  --chunks 128 \
  --realistic-trail \
  --where "instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y')"
echo "GEN REALISTIC DONE $(date -Is)"
