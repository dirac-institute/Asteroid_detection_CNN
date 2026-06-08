# pipeline_config.sh -- single source of truth for the same-night NEO DISCOVERY pipeline.
#
# Source this from the SLURM drivers (sn_run.slurm, sn_detect.slurm). EVERY value is overridable by
# pre-setting the matching environment variable before sbatch; the defaults below are the SDF/DP2 deployment.
# To run at a different site / data release, override the env vars (no code edits needed):
#   ADCNN_REPO=/path/to/repo BUTLER_COLLECTION=... sbatch ... sn_run.slurm
#
# Resolve the repo root from THIS file's location (ADCNN/pipelines/heliolinc/pipeline_config.sh -> repo root),
# so the pipeline is portable; only override ADCNN_REPO if running from a relocated checkout.
_CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ADCNN_REPO="${ADCNN_REPO:-$(cd "$_CFG_DIR/../../.." && pwd)}"

# Python (torch) env for ADCNN detection + linking.
export ADCNN_CONDA_PROFILE="${ADCNN_CONDA_PROFILE:-/sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh}"
export ADCNN_CONDA_ENV="${ADCNN_CONDA_ENV:-asteroid_cnn}"

# LSST science-pipelines stack setup (for Butler reads: manifest + known catalogue). One shell command.
export LSST_STACK_SETUP="${LSST_STACK_SETUP:-source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; setup lsst_distrib}"

# Butler repo + the difference-image / SSObject collection (the data-release vintage).
export BUTLER_REPO="${BUTLER_REPO:-dp2_prep}"
export BUTLER_COLLECTION="${BUTLER_COLLECTION:-LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4}"

# Observatory MPC code carried into the detection catalogue.
export OBSCODE="${OBSCODE:-I11}"

# Calibrated linking operating point (JSON; see link_op_point.json). Read by trail_state_link --op-point.
export LINK_OP_POINT="${LINK_OP_POINT:-$ADCNN_REPO/ADCNN/pipelines/heliolinc/link_op_point.json}"

# Robustness knobs.
export MAX_DETECT_ATTEMPTS="${MAX_DETECT_ATTEMPTS:-30}"   # GPU-detect resubmits through preemption
export MAX_CORRUPT_FRAC="${MAX_CORRUPT_FRAC:-0.10}"       # fail a detect shard if > this fraction of FITS unreadable

# Convenience: activate the torch env with a clear error if missing.
adcnn_activate() {
  # shellcheck disable=SC1090
  source "$ADCNN_CONDA_PROFILE" || { echo "[config] ERROR: conda profile not found: $ADCNN_CONDA_PROFILE" >&2; return 1; }
  conda activate "$ADCNN_CONDA_ENV" || { echo "[config] ERROR: conda env '$ADCNN_CONDA_ENV' not found" >&2; return 1; }
}
