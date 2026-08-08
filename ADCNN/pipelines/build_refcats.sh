#!/bin/bash
# Build the per-night all-sky bright-star refcat (the_monster, mag<19) for the 1.5k proximity veto.
# Writes 10k_cadence/run_night_<N>/bright_refcat.parquet. Runs under the LSST stack (Butler).
# Usage:  bash ADCNN/pipelines/build_refcats.sh 20260629 20260630 ...
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
source ADCNN/pipelines/heliolinc/pipeline_config.sh >/dev/null 2>&1
LSST="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; setup lsst_distrib"
for N in "$@"; do
  R=outputs/runs/10k_cadence/run_night_$N
  # MAG_MAX 19 -> 21 (MEASURED 2026-08-06 on 0706, offset-null): the residual bright-star DIPOLE/RINGS
  # in the delivered product sit on mag 19-21 stars, which a mag<19 refcat CANNOT see (the surviving
  # rings' nearest CATALOGUED star was 26" away = unrelated). the_monster is complete to G~21. Deepening
  # the cut takes the proximity veto from 21.7% of the product flagged (0% of the visible residual
  # rings) to 55.8% (42% of them), while the offset-null cost to REAL movers rises only 0.9% -> 2.7%
  # (~20:1). The null is the chance rate at random positions = exactly what a real mover suffers; the
  # earlier "12% trail cost" figure was measured circularly against ring-contaminated product alerts.
  MAG_MAX=${MAG_MAX:-21}
  OUT=$R/bright_refcat.parquet
  if [ -s "$OUT" ]; then echo "[$N] refcat exists -- skip"; continue; fi
  [ -s "$R/adcnn_dets_masked.csv" ] || { echo "[$N] no masked dets -- skip"; continue; }
  echo "[$N] building bright refcat (mag<$MAG_MAX) -> $OUT"
  bash -c "$LSST; cd $PWD; BUTLER_REPO=embargo python -m ADCNN.linking.build_static_refcat \
      --dets $R/adcnn_dets_masked.csv --out $OUT --refcat the_monster_20250219 --mag-max $MAG_MAX" \
      > outputs/logs/refcat_$N.log 2>&1 \
      && echo "[$N] ok ($(python3 -c "import pandas as pd;print(len(pd.read_parquet('$OUT')))" 2>/dev/null) stars)" \
      || echo "[$N] FAILED -- see outputs/logs/refcat_$N.log"
done
echo "refcats done"
