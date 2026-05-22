#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-gen-shard
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_gen_shard_%A_%a.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
cd "$REPO/ADCNN/data/dataset_creation"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/group/rubin/sw/loadLSST.bash; setup lsst_distrib
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
SH=${SLURM_ARRAY_TASK_ID}
OUT="$REPO/DATA_DIFFIM_realistic/shard_${SH}"; mkdir -p "$OUT"
SEED=$((1000 + SH*131))
echo "=== gen shard $SH (seed $SEED, k=1150, test excluded) === $(date -Is)"
srun python3 -u simulate_inject_diffim.py \
  --repo dp2_prep \
  --collections "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3" "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2" \
  --stage3-collection "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3" \
  --skymap lsst_cells_v2 --save-path "$OUT" \
  --parallel "${SLURM_CPUS_PER_TASK:-48}" \
  --train-test-split 0.99 --random-subset 2000 \
  --trail-length-min 6 --trail-length-max 60 --mag-min 2 --mag-max 8 --mag-mode snr \
  --beta-min 0 --beta-max 180 --number 20 --stack-detection-threshold 5.0 \
  --chunks 128 --seed $SEED --realistic-trail --skip-prevalidation --target-train-panels 1100 \
  --exclude-pairs-csv "$REPO/DATA_DIFFIM/test_5sigma/test.csv" "$REPO/DATA_DIFFIM/test_real/test.csv" \
  --where "instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band in ('u','g','r','i','z','y')"
echo "GEN SHARD $SH DONE $(date -Is)"
