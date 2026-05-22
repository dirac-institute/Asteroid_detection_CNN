#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-gen-huge
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=350G
#SBATCH --time=1-12:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_gen_huge_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO/ADCNN/data/dataset_creation"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/group/rubin/sw/loadLSST.bash; setup lsst_distrib
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
OUT="$REPO/DATA_DIFFIM_realistic"; mkdir -p "$OUT"
echo "=== HUGE realistic gen (~4300 subset, test pairs excluded) === $(date -Is)"
srun python3 -u simulate_inject_diffim.py \
  --repo dp2_prep \
  --collections "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3" "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2" \
  --stage3-collection "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3" \
  --skymap lsst_cells_v2 --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-90}" \
  --train-test-split 0.99 --random-subset 4300 \
  --trail-length-min 6 --trail-length-max 60 --mag-min 2 --mag-max 8 --mag-mode snr \
  --beta-min 0 --beta-max 180 --number 20 --stack-detection-threshold 5.0 \
  --chunks 128 --seed 2025 --realistic-trail \
  --exclude-pairs-csv "$REPO/DATA_DIFFIM/test_5sigma/test.csv" "$REPO/DATA_DIFFIM/test_real/test.csv" \
  --where "instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y')"
echo "GEN HUGE DONE $(date -Is)"
