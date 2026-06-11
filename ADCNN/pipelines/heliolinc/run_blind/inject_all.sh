#!/bin/bash
# Blind-test round 5: inject all 26 fields against wcs_json-annotated manifests.
# Canaries per field: (a) >=70% of manifest panels usable (purge canary), (b) >0 sightings (sim_orbits
# now fails loud itself), (c) injected RA/Dec are degree-valued (footprint printed by sim_orbits).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
HELIO="$(dirname "$HERE")"
PY=/sdf/data/rubin/user/mrakovci/conda/envs/asteroid_cnn/bin/python3
cd "$HELIO"
ok=0; fail=0; failed_ks=""
for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 24 25 26 27 28 29; do
  log="$HERE/inject_$k.log"
  PYTHONPATH=. $PY sim_orbits.py \
    --manifest "$HERE/manifest_$k.csv" \
    --retime-map "$HERE/retime_$k.csv" \
    --out-inject "$HERE/inject_$k.csv" \
    --out-truth "$HERE/truth_$k.csv" \
    --n-objects 300 --seed $((3000+k)) > "$log" 2>&1
  rc=$?
  # purge canary: usable/manifest >= 70%
  frac_ok=$($PY - "$log" <<'EOF'
import re, sys
txt = open(sys.argv[1]).read()
m = re.search(r"\[orbits\] (\d+)/(\d+) manifest panels usable", txt)
print("yes" if m and int(m.group(1)) >= 0.7*int(m.group(2)) else "no")
EOF
)
  if [ $rc -eq 0 ] && [ -s "$HERE/inject_$k.csv" ] && [ "$frac_ok" = "yes" ]; then
    ok=$((ok+1)); echo "field $k OK: $(grep -m1 'objects ->' "$log")"
  else
    fail=$((fail+1)); failed_ks="$failed_ks $k"
    echo "field $k FAIL (rc=$rc frac_ok=$frac_ok): $(tail -2 "$log" | head -1)"
  fi
done
echo "INJECT_ALL_DONE ok=$ok fail=$fail failed:$failed_ks"
