#!/bin/bash
#SBATCH --partition=roma
#SBATCH --account=rubin:commissioning
#SBATCH -c 64
#SBATCH --mem=192G
#SBATCH -t 24:00:00
#SBATCH -J sn_run
#SBATCH -o %x_%j.log
set -eo pipefail
# CANONICAL same-night NEO pipeline orchestrator (run on CPU; drives the GPU detect + the rest).
#
# Morning run (build manifest from a night + tract(s) and link 2-visit tracks):
#   DAY=20250706 TRACTS=8489 RUN=<out dir>  sbatch --export=ALL,DAY,TRACTS,RUN sn_run.slurm
# Or with a pre-built manifest (DAY/TRACTS then optional):
#   RUN=<dir with manifest.csv>  sbatch --export=ALL,RUN sn_run.slurm
#
# Stages (this IS how we run it every morning):
#   0. manifest.csv for DAY + TRACTS (build_manifest, LSST stack env) if $RUN/manifest.csv is missing.
#   1. ADCNN detect (sn_detect.slurm on ampere) -> adcnn_dets.csv. ampere is PREEMPTIBLE; discover_stream
#      is resumable, so we resubmit via `sbatch --wait` in a loop until the catalog is complete.
#   2. known.csv (mpcorb SSObject ephemerides) for crossmatch, in the LSST stack env, if missing.
#   3. LSST diffim mask cut (mask_flags.py): TP-safe artifact removal (SPIKE/SAT/CR/STREAK/DETECTED_NEGATIVE).
#   4. trail_state_link.py: SINGLE-NIGHT linking in (position, trail-velocity) phase space -- NO heliocentric
#      hypothesis grid (the trail measures velocity directly), so NO candidate explosion; O(N log N).
#      Two tiers: 3+visit (full linear check) and 2visit (Heinze minimum; tighter pa_tol_2v + trail-vs-
#      trail velocity agreement). physical_check rejects false links.
#   -> tracks.csv : CONFIRMED (known recoveries) + NEW (uncatalogued same-night candidates), tagged by tier.
# ---- resolved config (single source of truth; every value overridable via env -- see pipeline_config.sh) ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/pipeline_config.sh"
: "${RUN:?set RUN=<out dir>}"
cd "$ADCNN_REPO"
mkdir -p "$RUN"
# record the exact configuration this run used (reproducibility / provenance)
cat > "$RUN/run_config.json" <<JSON
{"adcnn_repo":"$ADCNN_REPO","butler_repo":"$BUTLER_REPO","butler_collection":"$BUTLER_COLLECTION",
 "obscode":"$OBSCODE","link_op_point":"$LINK_OP_POINT","conda_env":"$ADCNN_CONDA_ENV",
 "day":"${DAY:-}","tracts":"${TRACTS:-}","score_min_override":"${SCORE_MIN:-}"}
JSON
echo "[sn_run] REPO=$ADCNN_REPO | COLLECTION=$BUTLER_COLLECTION | OBSCODE=$OBSCODE | OP_POINT=$LINK_OP_POINT"

# 0. manifest for DAY + TRACTS (LSST stack env) if missing
if [ ! -s "$RUN/manifest.csv" ]; then
  : "${DAY:?set DAY=<day_obs, e.g. 20250706> (or pre-build $RUN/manifest.csv)}"
  : "${TRACTS:?set TRACTS=<tract list/ranges, e.g. 8489> (or pre-build $RUN/manifest.csv)}"
  echo "[sn_run] building manifest for DAY=$DAY TRACTS=$TRACTS"
  bash -c "$LSST_STACK_SETUP; cd $ADCNN_REPO; python -m ADCNN.pipelines.heliolinc.build_manifest \
    --tracts $TRACTS --day-start $DAY --day-end $((DAY+1)) --butler-repo '$BUTLER_REPO' \
    --collection '$BUTLER_COLLECTION' --out $RUN/manifest.csv"
fi
# preflight gate: a manifest with only a header means no diffims resolved for this DAY/TRACTS/collection.
[ -s "$RUN/manifest.csv" ] && [ "$(wc -l < "$RUN/manifest.csv")" -gt 1 ] || {
  echo "[sn_run] ERROR: empty manifest -- no diffims for DAY=${DAY:-?} TRACTS=${TRACTS:-?} in $BUTLER_COLLECTION"; exit 1; }

# 1. ADCNN detect, resubmitting through preemptions (resumable via .done), exponential backoff
n=0
while [ ! -s "$RUN/adcnn_dets.csv" ] && [ "$n" -lt "${MAX_DETECT_ATTEMPTS:-30}" ]; do
  n=$((n+1)); echo "[sn_run] ADCNN detect attempt $n/${MAX_DETECT_ATTEMPTS:-30}"
  RUN="$RUN" sbatch --export=ALL,RUN --wait "$HERE/sn_detect.slurm" || true
  [ -s "$RUN/adcnn_dets.csv" ] && break
  sleep $(( n < 6 ? (1 << n) : 60 ))   # 2,4,8,16,32,60,60,... s
done
[ -s "$RUN/adcnn_dets.csv" ] || { echo "[sn_run] ERROR: detect produced no non-empty catalog after $n attempts"; exit 1; }

# 2. known.csv (LSST stack env) if missing
if [ ! -f "$RUN/known.csv" ]; then
  bash -c "$LSST_STACK_SETUP; cd $ADCNN_REPO; python -m ADCNN.pipelines.heliolinc.build_known_catalog \
    --manifest $RUN/manifest.csv --butler-repo '$BUTLER_REPO' --collection '$BUTLER_COLLECTION' --out $RUN/known.csv"
fi

# 3 + 4. mask cut + single-night trail-state linking (torch env)
adcnn_activate || exit 1
python -m ADCNN.pipelines.heliolinc.mask_flags --dets "$RUN/adcnn_dets.csv" \
  --manifest "$RUN/manifest.csv" --out "$RUN/adcnn_dets_masked.csv" --workers "${MASK_WORKERS:-64}"
[ -s "$RUN/adcnn_dets_masked.csv" ] || { echo "[sn_run] ERROR: mask stage produced empty output"; exit 1; }
# --score-min 0.80 = the 3-SIGMA discovery operating point (FPP Monte-Carlo, link_fpp.json): the 3+visit
# tier has lambda_FP <= 1.35e-3/night at this floor (recovers 2025 NY2). LINKER ARCHITECTURE (2026-06):
#   * 2-visit seeded by the precise position CHORD (--seed-2v chord) not the noisy trail-velocity -> ~4x
#     recall + ~10x lower FP vs the old tref-clustering (measured on real DP2 off-ecliptic FP).
#   * 2-visit purity is carried by the GEOMETRIC orbit-fit chi^2 (collinearity + trail-vs-motion PA & speed):
#     --chi2-2v-max 5 is the PRIMARY lever. Tightening it 10->5 lifts the 3-sigma completeness at the realistic
#     34-min WFD cadence from 0.070 to ~0.10 (+~45%%, validated on 80 off-ecliptic lambda fields) at no purity
#     cost (true median chi2 ~4 vs false ~39). The mf_snr floor (--mfsnr-min-2v 10) is a CADENCE-DEPENDENT
#     purity dial: ~10 holds 3-sigma at the 34-min gap; LOWER to ~5 at rapid (deep-drilling) cadence where FP
#     are sparse, to recover fast FAINT movers (a long trail has low matched-filter SNR ~ point_SNR*sqrt(PSF/
#     trail_area), so a high floor rejects exactly them). Epochs are now keyed by VISIT id (not a time gap),
#     and chord-seeding pairs ALL same-night visit pairs -- both fixes are required for rapid same-night cadences.
#   * 3+visit clustered with --pos-tol-3v 0.05 (~2x its recall at zero purity cost).
# 3+visit tracks at this floor are 3-sigma discoveries; 2visit at S0.80 + chi2<=5 + mfsnr>=10 are 3-sigma-PURITY
# at ~10%% completeness (detection ceiling -- BOTH same-night sightings must clear the score floor; reaching
# >=50%% requires a 3rd epoch / multi-night arc, NOT a looser 2-visit cut).
# The calibrated 2v/3v op-point (score floor, chi2<=5, mfsnr>=10, rate band, tolerances) lives in
# link_op_point.json (versioned config, $LINK_OP_POINT) -- NOT magic numbers here. SCORE_MIN env overrides
# the score floor for a given run; any individual param can also be overridden by adding its --flag below.
python -m ADCNN.pipelines.heliolinc.trail_state_link --dets "$RUN/adcnn_dets_masked.csv" \
  --known "$RUN/known.csv" --out "$RUN/tracks.csv" --op-point "$LINK_OP_POINT" \
  --npt 2 --min-epochs 2 --seed-2v chord ${SCORE_MIN:+--score-min "$SCORE_MIN"}
[ -f "$RUN/tracks.csv" ] || { echo "[sn_run] ERROR: linker did not write tracks.csv"; exit 1; }
echo "SN_RUN DONE -> $RUN/tracks.csv ($(( $(wc -l < "$RUN/tracks.csv") - 1 )) tracks)"
