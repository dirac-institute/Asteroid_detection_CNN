#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -J inj3v
#SBATCH -o outputs/logs/%x_%j.log
#SBATCH --open-mode=append
set -eo pipefail
# 3-EPOCH injection campaign: the truth set that makes the 3+visit gate tunable.
#
# Injects movers into COMMON-FOOTPRINT visit triples (INJ_EPOCHS=3; see inject_night.visit_groups
# for why boresight pairing finds zero triples) and records the per-epoch detection cascade
# (detA/B/C). Downstream: link the inj_dets catalogue, measure which truth triplets become 3+visit
# alerts and what chance triplets look like on the SAME geometry statistics the packet now publishes
# (linRmsArcsec / trailMotionDpaMaxDeg / speedRatio) -- that distribution is what turns the gate's
# untuned thresholds into measured ones.
#
# ONE GPU: this is a single sequential process (unlike sn_detect's 10-way shard). ~30 A-panels/group
# x (1 + 2 follow-up epochs x ~2 covering detectors) at ~4.6 s/panel => ~3 h for 30 groups.
#
#   INJ_RUN=outputs/runs/10k_cadence/run_night_20260706 INJ_TAG=tri_n20260706 \
#     sbatch ADCNN/analysis/truth_harness/inject_3v.slurm
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CFG="$HERE/../../pipelines/heliolinc/pipeline_config.sh"
if [ ! -f "$CFG" ]; then
  if [ -n "$ADCNN_REPO" ]; then CFG="$ADCNN_REPO/ADCNN/pipelines/heliolinc/pipeline_config.sh"
  else CFG="${SLURM_SUBMIT_DIR}/ADCNN/pipelines/heliolinc/pipeline_config.sh"; fi
fi
# shellcheck disable=SC1091
source "$CFG"
: "${INJ_RUN:?set INJ_RUN=<run_night dir with manifest.csv + adcnn_dets_masked.csv>}"
: "${INJ_TAG:?set INJ_TAG=<truth/dets suffix, e.g. tri_n20260706>}"
adcnn_activate || exit 1
cd "$ADCNN_REPO"
export INJ_EPOCHS=3
# 30 arc-stratified triples x 15 detectors x 25 movers/panel
PYTHONPATH="$PWD" python -m ADCNN.analysis.truth_harness.inject_night 30 15 25
echo "[inj3v] DONE tag=$INJ_TAG"
