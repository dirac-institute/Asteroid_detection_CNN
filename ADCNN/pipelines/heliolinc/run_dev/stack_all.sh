#!/bin/bash
# Blind-test stack baselines: 5-sigma and 4-sigma SourceDetectionTask on the SAME injected pixels
# (inject-panels-only), retimed MJDs. Runs under the LSST stack env. afw reads the exact archive
# SkyWcs from the FITS, so this is unaffected by the DM-53195 header-WCS issue.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
HELIO="$(dirname "$HERE")"
cd "$HELIO"
ok=0; fail=0; failed=""
for thr in 5.0; do
  tag=${thr%.0}
  for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    out="$HERE/stack_dets_s${tag}_$k.csv"
    [ -s "$out" ] && { echo "skip s$tag field $k (exists)"; ok=$((ok+1)); continue; }
    python stack_detect.py --manifest "$HERE/manifest_$k.csv" --inject "$HERE/inject_$k.csv" \
      --retime-map "$HERE/retime_$k.csv" --threshold $thr --inject-panels-only --workers 32 \
      --out "$out" > "$HERE/stack_s${tag}_$k.log" 2>&1
    if [ $? -eq 0 ] && [ -s "$out" ]; then
      ok=$((ok+1)); echo "s$tag field $k OK: $(wc -l < "$out") rows"
    else
      fail=$((fail+1)); failed="$failed s$tag:$k"
      echo "s$tag field $k FAIL: $(tail -1 "$HERE/stack_s${tag}_$k.log")"
    fi
  done
done
echo "STACK_ALL_DONE ok=$ok fail=$fail failed:$failed"
