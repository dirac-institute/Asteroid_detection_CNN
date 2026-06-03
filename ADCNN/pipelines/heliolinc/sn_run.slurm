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
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
: "${RUN:?set RUN=<out dir>}"
LSST="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; setup lsst_distrib"
cd "$REPO"
mkdir -p "$RUN"

# 0. manifest for DAY + TRACTS (LSST stack env) if missing
if [ ! -f "$RUN/manifest.csv" ]; then
  : "${DAY:?set DAY=<day_obs, e.g. 20250706> (or pre-build $RUN/manifest.csv)}"
  : "${TRACTS:?set TRACTS=<tract list/ranges, e.g. 8489> (or pre-build $RUN/manifest.csv)}"
  echo "[sn_run] building manifest for DAY=$DAY TRACTS=$TRACTS"
  bash -c "$LSST; cd $REPO; python -m ADCNN.pipelines.heliolinc.build_manifest --tracts $TRACTS --day-start $DAY --day-end $((DAY+1)) --out $RUN/manifest.csv"
fi

# 1. ADCNN detect, resubmitting through preemptions (resumable via .done)
n=0
while [ ! -f "$RUN/adcnn_dets.csv" ] && [ $n -lt 30 ]; do
  n=$((n+1)); echo "[sn_run] ADCNN detect attempt $n"
  RUN="$RUN" sbatch --export=ALL,RUN --wait ADCNN/pipelines/heliolinc/sn_detect.slurm || true
done
[ -f "$RUN/adcnn_dets.csv" ] || { echo "[sn_run] ERROR: detect did not complete after $n attempts"; exit 1; }

# 2. known.csv (LSST stack env) if missing
if [ ! -f "$RUN/known.csv" ]; then
  bash -c "$LSST; cd $REPO; python -m ADCNN.pipelines.heliolinc.build_known_catalog --manifest $RUN/manifest.csv --out $RUN/known.csv"
fi

# 3 + 4. mask cut + single-night trail-state linking (torch env)
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
python -m ADCNN.pipelines.heliolinc.mask_flags --dets "$RUN/adcnn_dets.csv" \
  --manifest "$RUN/manifest.csv" --out "$RUN/adcnn_dets_masked.csv" --workers 64
# --score-min 0.80 = the 3-SIGMA discovery operating point (FPP Monte-Carlo, link_fpp.json): the 3+visit
# tier has lambda_FP <= 1.35e-3/night at this floor (recovers 2025 NY2). LINKER ARCHITECTURE (2026-06):
#   * 2-visit seeded by the precise position CHORD (--seed-2v chord) not the noisy trail-velocity -> ~4x
#     recall + ~10x lower FP vs the old tref-clustering (measured on real DP2 off-ecliptic FP).
#   * 2-visit accepted by a single combined orbit-fit chi^2 (--chi2-2v-max 3.0) instead of independent
#     AND-thresholds -> +2.5x completeness at the SAME false rate (lambda~0.0023/pair). No ML.
#   * 3+visit clustered with --pos-tol-3v 0.05 (~2x its recall at zero purity cost).
# 3+visit tracks at this floor are 3-sigma discoveries; 2visit are candidate-grade (lambda~0.0023/pair,
# ~1.7x over 3-sigma point est. -- 2 same-night faint points cannot self-confirm; feed to follow-up).
python -m ADCNN.pipelines.heliolinc.trail_state_link --dets "$RUN/adcnn_dets_masked.csv" \
  --known "$RUN/known.csv" --out "$RUN/tracks.csv" \
  --npt 2 --min-epochs 2 --max-rms 1.0 --pa-tol 20 --pa-tol-2v 10 --score-min "${SCORE_MIN:-0.80}" \
  --seed-2v chord --chi2-2v-max 3.0 --pos-tol-3v 0.05 --max-arc-2v-min 40
echo "SN_RUN DONE -> $RUN/tracks.csv"
