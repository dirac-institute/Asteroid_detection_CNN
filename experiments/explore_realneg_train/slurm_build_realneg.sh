#!/bin/bash
# EXPERIMENTAL realneg dataset build (CPU, LSST stack env, roma).
# Three leakage-disjoint builds (trail / empty / held-out empty) + merge.
# Outputs under the gitignored experiments/diffim_runs/seg_ft_realneg/.
# Usage: sbatch experiments/explore_realneg_train/slurm_build_realneg.sh
#
#SBATCH --job-name=adc-realneg-build
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=5G
#SBATCH --time=1-00:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_realneg_build_%j.out
set -eo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
# Heavy outputs go to the 80G scratch area — the 932G repo quota is full.
RUN="/sdf/scratch/users/m/mrakovci/realneg"
D="${RUN}/data"
S="${REPO_DIR}/experiments/explore_realneg_train"
WHERE="day_obs>=20250801 AND day_obs<=20250921"
PAR="${SLURM_CPUS_PER_TASK:-40}"
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${D}"
M=experiments/explore_realneg_train/build_realneg.py

# 1) TRAIL panels: faint streaks, snr bulk-faint <=10, wide length.
srun python "${M}" --mode trail  --out "${D}/trail"   --where "${WHERE}" \
  --n-panels 400 --parallel "${PAR}" --number 20 \
  --snr-min 3 --snr-max 10 --len-min 4 --len-max 200 --seed 20260519

# 2) EMPTY panels: number=0, disjoint from TRAIL.
srun python "${M}" --mode empty  --out "${D}/empty"   --where "${WHERE}" \
  --n-panels 200 --parallel "${PAR}" --seed 20260520 \
  --also-exclude "${D}/trail/chosen_pairs.csv"

# 3) HELD-OUT empty FP benchmark: disjoint from TRAIL and EMPTY.
srun python "${M}" --mode empty  --out "${D}/heldout" --where "${WHERE}" \
  --n-panels 120 --parallel "${PAR}" --seed 20260521 \
  --also-exclude "${D}/trail/chosen_pairs.csv" "${D}/empty/chosen_pairs.csv"

# 4) Merge TRAIL + EMPTY -> trainer-ready dataset (heldout kept separate).
python "${S}/merge_realneg.py" --trail "${D}/trail" --empty "${D}/empty" \
  --out "${RUN}/dataset"

echo "REALNEG-BUILD DONE $(date -Is)"
echo "  train: ${RUN}/dataset/{train.h5,train.csv,panels.csv}"
echo "  heldout-empty FP bench: ${D}/heldout/{data.h5,panels.csv}"
