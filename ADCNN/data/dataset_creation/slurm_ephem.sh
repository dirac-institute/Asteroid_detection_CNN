#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=adc-ephem
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

OUT=real_fast_trails_dataset
rm -f $OUT/train.h5 $OUT/test.h5 $OUT/train.csv $OUT/test.csv
mkdir -p $OUT

EPH=lsstcam_fast_trails_objects.csv  # from your ephemerides notebook

srun python3 -u real_ephemerides.py \
  --repo /repo/main \
  --collections LSSTCam/runs/nightlyValidation \
  --ephemerides-csv $EPH \
  --save-path $OUT \
  --speed-thr 0.5 \
  --parallel 1 \
  --train-test-split 0.1 \
  --random-subset 0
