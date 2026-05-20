#!/bin/bash
# Resume the realneg build: trail panels already valid on scratch
# (/sdf/scratch/users/m/mrakovci/realneg/data/trail/data.h5, 399 panels).
# This script rebuilds only the empty + heldout-empty steps with the fixed
# n_inject==0 worker path, then merges into the trainer-ready dataset.
#
#SBATCH --job-name=adc-realneg-resume
#SBATCH --account=rubin:developers
#SBATCH --partition=roma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output=/sdf/home/m/mrakovci/logs/ADCNN_realneg_resume_%j.out
set -eo pipefail
REPO_DIR="${REPO_DIR:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
RUN="/sdf/scratch/users/m/mrakovci/realneg"
D="${RUN}/data"
S="${REPO_DIR}/experiments/explore_realneg_train"
WHERE="day_obs>=20250801 AND day_obs<=20250921"
PAR="${SLURM_CPUS_PER_TASK:-40}"
source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh
setup lsst_distrib
cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
M=experiments/explore_realneg_train/build_realneg.py

# Sanity: trail data must already be valid.
test -f "${D}/trail/data.h5" || { echo "MISSING ${D}/trail/data.h5"; exit 2; }
test -f "${D}/trail/chosen_pairs.csv" || { echo "MISSING ${D}/trail/chosen_pairs.csv"; exit 2; }
echo "[resume] trail data already on disk: $(du -sh ${D}/trail/data.h5)"

# 1) EMPTY panels: number=0, disjoint from TRAIL (and existing exclude set).
if [ ! -f "${D}/empty/data.h5" ] || [ "$(stat -c%s ${D}/empty/data.h5)" -lt 1000000 ]; then
  rm -rf "${D}/empty"
  srun python "${M}" --mode empty --out "${D}/empty" --where "${WHERE}" \
    --n-panels 200 --parallel "${PAR}" --seed 20260520 \
    --also-exclude "${D}/trail/chosen_pairs.csv"
else
  echo "[resume] empty data already present: $(du -sh ${D}/empty/data.h5)"
fi

# 2) HELDOUT empty FP benchmark: disjoint from TRAIL and EMPTY.
if [ ! -f "${D}/heldout/data.h5" ] || [ "$(stat -c%s ${D}/heldout/data.h5)" -lt 1000000 ]; then
  rm -rf "${D}/heldout"
  srun python "${M}" --mode empty --out "${D}/heldout" --where "${WHERE}" \
    --n-panels 120 --parallel "${PAR}" --seed 20260521 \
    --also-exclude "${D}/trail/chosen_pairs.csv" "${D}/empty/chosen_pairs.csv"
else
  echo "[resume] heldout data already present: $(du -sh ${D}/heldout/data.h5)"
fi

# 3) Merge TRAIL + EMPTY -> trainer-ready dataset (heldout kept separate).
rm -rf "${RUN}/dataset"
python "${S}/merge_realneg.py" --trail "${D}/trail" --empty "${D}/empty" \
  --out "${RUN}/dataset"

echo "REALNEG-RESUME DONE $(date -Is)"
echo "  train: ${RUN}/dataset/{train.h5,train.csv,panels.csv}"
echo "  heldout-empty FP bench: ${D}/heldout/{data.h5,panels.csv}"
df -h /sdf/scratch | head -3
du -sh "${RUN}"
