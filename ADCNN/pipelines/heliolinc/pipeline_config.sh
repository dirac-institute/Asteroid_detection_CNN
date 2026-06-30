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

# S3 access for a remote (object-store) datastore -- e.g. the `embargo` prompt-processing repo,
# whose `difference_image` URIs are s3:// (read in-memory by ADCNN.inference.diffim_io, no local
# clutter). These are inherited by the LSST stack env but NOT by the torch detect env, so export
# them here so detect_night's S3 reads work. Harmless for a local (POSIX) datastore. The two
# CHECKSUM knobs are mandatory: without them botocore thrashes the SDF gateway (~33 s/panel vs
# ~0.65 s) -- diffim_io also sets them defensively, this just makes the launcher self-contained.
export S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://s3dfrgw.slac.stanford.edu}"
export LSST_RESOURCES_S3_PROFILE_embargo="${LSST_RESOURCES_S3_PROFILE_embargo:-https://sdfembs3.sdf.slac.stanford.edu}"
export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.lsst/aws-credentials.ini}"
export AWS_REQUEST_CHECKSUM_CALCULATION="${AWS_REQUEST_CHECKSUM_CALCULATION:-WHEN_REQUIRED}"
export AWS_RESPONSE_CHECKSUM_VALIDATION="${AWS_RESPONSE_CHECKSUM_VALIDATION:-WHEN_REQUIRED}"

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
