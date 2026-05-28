#!/bin/bash
#SBATCH --job-name=adc-fpcurve
#SBATCH --account=kipac:kipac
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_fpcurve_%j.out
set -eo pipefail
export RUBIN_EUPS_PATH="${RUBIN_EUPS_PATH:-}"
REPO="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"; cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn
srun python3 -u experiments/explore_simreal_gap/fp_curve.py \
  "$REPO/experiments/diffim_runs/pilot_seg_reg2/ckpts/segmentation_reg2_best_scripted.pt::reg2" \
  "$REPO/experiments/diffim_runs/pilot_seg_realistic/ckpts/seg_realistic_scripted.pt::seg_model-realistic-baseline"
