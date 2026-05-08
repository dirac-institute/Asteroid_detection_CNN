#!/bin/bash
# Stage-1 diffim generator as a SLURM array.
#
# One array task = one (visit, detector) from the manifest.
#
# Usage:
#   sbatch --array=0-4 slurm_generate.sh \
#       experiments/diffim/stage1_generate/manifests/pilot_g_5.json \
#       experiments/diffim/stage1_generate/runs/pilot_g_5
#
#SBATCH --job-name=adc-diffim-gen
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/adc_diffim_gen_%A_%a.out

set -eo pipefail

MANIFEST="${1:?usage: slurm_generate.sh <manifest.json> <out_root>}"
OUT_ROOT="${2:?usage: slurm_generate.sh <manifest.json> <out_root>}"

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib

REPO_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN"
cd "${REPO_DIR}"

mkdir -p "${OUT_ROOT}"

srun python3 -u experiments/diffim/stage1_generate/driver.py \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --out-root "${OUT_ROOT}" \
  --n-trails-min 40 \
  --n-trails-max 100 \
  --mag-min 22.5 \
  --mag-max 26.0 \
  --trail-length-min 6 \
  --trail-length-max 60
