#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-inject
#SBATCH --account=rubin:developers
#SBATCH --output=%x.%j.out
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00

set -euo pipefail

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2025_24/loadLSST.sh
setup lsst_distrib

cd $SLURM_SUBMIT_DIR

OUT=sim_dataset
rm -f $OUT/train.h5 $OUT/test.h5 $OUT/train.csv $OUT/test.csv
mkdir -p $OUT

srun python3 -u simulate_inject.py \
  --repo /repo/main \
  --collections LSSTComCam/runs/DRP/DP1/w_2025_24/DM-48478 \
  --save-path $OUT \
  --parallel 12 \
  --train-test-split 0.1 \
  --random-subset 0 \
  --trail-length-min 6 --trail-length-max 60 \
  --mag-min 19 --mag-max 24 \
  --beta-min 0 --beta-max 180 \
  --number 20
