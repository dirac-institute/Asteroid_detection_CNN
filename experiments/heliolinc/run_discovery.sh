#!/bin/bash
# End-to-end ADCNN -> HelioLinC asteroid-discovery pipeline (one ~2-week window).
#
#   Stage A (asteroid_cnn, GPUs): ADCNN.inference.catalog  -> catalog.csv  (pixel x,y + score_rf)
#   Stage B (lsst_distrib, Butler): adcnn_wcs.py            -> adcnn_dets.csv (RA/Dec/MJD)
#   Link   (no env): make_tracklets -> heliolinc -> link_refine -> lr.csv (orbit-consistent tracks)
#   Match  (asteroid_cnn): crossmatch.py -> CONFIRMED (known) + NEW (unmatched) asteroids
#
# Each stage is toggleable (run only what you need); HelioLinC linking uses the config validated
# on the truth catalog (full grid + ~2-week window + mjd at midpoint -> see heliolinc-linking-fix).
set -euo pipefail
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/experiments/heliolinc
BIN=$HL/heliolinc2/src
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh

# ---- window / inputs (override via env) ----
RUN=${RUN:-$HL/run_disco}
H5=${H5:-$REPO/DATA_DIFFIM/test_real/test.h5}
PANELS=${PANELS:-$HL/window_panels.csv}      # image_id,visit,detector,band of the window
MJDREF=${MJDREF:-60873}                       # window midpoint
RF_THR=${RF_THR:-0.15}                         # recall-favoring (HelioLinC rejects FP via physics)
GRID=${GRID:-$HL/run_2wk/heliohypo_all.txt}    # full distance grid 1.05-6.5 AU
STAGES=${STAGES:-ABLMX}                         # which stages to run: A B L(ink) M(atch) X(=all)

mkdir -p "$RUN"
cp -n "$HL/run_truth/colformat.txt" "$HL/run_truth/Earth1day2020s_02a.txt" \
      "$HL/run_truth/ObsCodes.txt" "$RUN/" 2>/dev/null || true

run() { [[ "$STAGES" == *"$1"* || "$STAGES" == *X* ]]; }

if run A; then
  echo "=== Stage A: ADCNN catalog (rf_thr=$RF_THR) ==="
  conda activate asteroid_cnn
  cd "$REPO"
  python -m ADCNN.inference.catalog --h5 "$H5" --panels "$PANELS" \
    --panel-ids "$PANELS" --rf-thr "$RF_THR" --gate-pmax 0.10 \
    --n-gpus "$(nvidia-smi -L | wc -l)" --out "$RUN/catalog.csv"
  conda deactivate
fi

if run B; then
  echo "=== Stage B: WCS -> RA/Dec/MJD ==="
  conda activate lsst_distrib 2>/dev/null || source /sdf/group/rubin/sw/loadLSST.bash && setup lsst_distrib
  python "$HL/adcnn_wcs.py" --cands "$RUN/catalog.csv" --out "$RUN/adcnn_dets.csv" --validate
fi

if run L; then
  echo "=== Link: make_tracklets -> heliolinc -> link_refine ==="
  cd "$RUN"
  "$BIN/make_tracklets" -dets adcnn_dets.csv -earth Earth1day2020s_02a.txt -obscode ObsCodes.txt \
    -colformat colformat.txt -pairdets pairdets.csv -pairs pairs.txt -outimgs imgs.txt \
    -maxtime 3.0 -mintime 0.0 -maxGCR 2.0 -mintrkpts 2 -maxvel 2.0 -minvel 0.0
  "$BIN/heliolinc" -dets pairdets.csv -pairs pairs.txt -mjd "$MJDREF" \
    -obspos Earth1day2020s_02a.txt -heliodist "$GRID" \
    -npt 3 -minobsnights 3 -mintimespan 1.0 -out hl_clusters.csv -outsum hl_summary.csv
  printf "hl_clusters.csv hl_summary.csv\n" > lflist.txt
  "$BIN/link_refine" -pairdet pairdets.csv -lflist lflist.txt -maxrms 100000 -outfile lr.csv -outrms lr_rms.csv
  echo "refined tracks: $(($(wc -l < lr_rms.csv) - 1))"
fi

if run M; then
  echo "=== Match: crossmatch linked tracks to known objects ==="
  conda activate asteroid_cnn
  python "$HL/crossmatch.py" --run "$RUN"
fi
echo "DISCOVERY PIPELINE DONE -> $RUN"
