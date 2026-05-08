#!/bin/bash
# Run the single-visit diffim proof on one (visit, detector) pair.
# Usage:
#   sbatch slurm_proof.sh <visit> <detector>
#
# Example:
#   sbatch slurm_proof.sh 2025042400172 1
#
# Keep this small on purpose: the proof is a physical sanity check, not a
# production job. Do not scale this up before the proof looks correct.
#SBATCH --job-name=adc-diffim-proof
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/adc_diffim_proof_%j.out

set -eo pipefail

VISIT="${1:?usage: slurm_proof.sh <visit> <detector>}"
DETECTOR="${2:?usage: slurm_proof.sh <visit> <detector>}"

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib

REPO_DIR="/sdf/home/m/mrakovci/rubin-user/Projects/Asteroid_detection_CNN"
OUT_DIR="${REPO_DIR}/experiments/diffim/proof/runs/v${VISIT}_d${DETECTOR}"
mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"

srun python3 -u experiments/diffim/proof/single_visit_proof.py \
  --visit "${VISIT}" \
  --detector "${DETECTOR}" \
  --out-dir "${OUT_DIR}" \
  --n-trails 5 \
  --trail-mag 22.0 \
  --trail-length-px 30
