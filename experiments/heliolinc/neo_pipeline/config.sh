# =============================================================================
# NEO trail-tracklet discovery pipeline — central config (sourced by every stage)
# =============================================================================
# One place to set paths, models, thresholds, parallelism and HelioLinC params.
# Tune the *_WORKERS / *_GPUS / NSHARD knobs to the node you run on (speed).
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/experiments/heliolinc
PIPE=$HL/neo_pipeline

# ---- run identity & I/O ----------------------------------------------------
RUN_NAME=${RUN_NAME:-run_wide_v2}            # output dir under $HL/<RUN_NAME>
RUN=$HL/$RUN_NAME
MANIFEST=${MANIFEST:-$HL/run_wide/manifest.csv}   # butler diffim FITS list (visit,detector,band,fits_path)
KNOWN=${KNOWN:-$HL/run_wide/known.csv}            # known-object sightings for crossmatch (ObjID,mjd,ra,dec)

# ---- models ----------------------------------------------------------------
V7=${V7:-$REPO/models/v7_diffim_scripted.pt}
RF=${RF:-$REPO/models/rf_postproc.pkl}

# ---- stage 1: detect (ADCNN, GPU) -----------------------------------------
FILTER=${FILTER:-cnn}                # stage-2 FP filter: cnn (focal-cutout CNN, GPU) or rf
CNN_THR=${CNN_THR:-0.63}             # CNN operating point (FP-matched to the old RF); same as eval
RF_THR=${RF_THR:-0.5}                # RF operating point (only used when FILTER=rf)
N_GPUS=${N_GPUS:-4}

# ---- stage 2: measure (Veres trailed fit, CPU) ----------------------------
# MEAS_LENGTH_MIN is a SPEED PRE-GATE ONLY (not a quality cut, not a score filter): which detections
# are worth the expensive Veres fit, judged on the ADCNN de-biased length (the accurate Veres length
# isn't computed yet). Keep it loose; the real >1 deg/day cut is applied later on the Veres length
# (CLEAN_LENDB_MIN). WAS 40 -> demanded ~7 deg/day and dropped ALL fast movers (their ADCNN length is
# median ~10px); 6px ≈ 1 deg/day.
MEAS_LENGTH_MIN=${MEAS_LENGTH_MIN:-6}
MEAS_WORKERS=${MEAS_WORKERS:-60}

# ---- stage 3: clean FP (dual-stream, validated on truth) -------------------
# diaSources (stack): Rubin real/bogus RELIABILITY cut (TP-safe: 95.7% TP / 93.6% FP removed).
# ADCNN: NO score cut here -- the stage-2 FP/score cut already happened ONCE at detect (CNN); a
#        second score threshold here would be a redundant, contradictory filter. NO real/bogus either
#        (its SNR-floor + trail-ceiling drop faint/fast trails) -> LINKING rejects residual FP. See
#        [[realbogus-fp-filter-limits]].
DIA_RELIABILITY_MIN=${DIA_RELIABILITY_MIN:-0.5}   # real/bogus, diaSources ONLY
CLEAN_LENDB_MIN=${CLEAN_LENDB_MIN:-6}        # VERES-measured trail length px (accurate); ~6px ≈ 1 deg/day = the real fast-mover cut
DIASRC=${DIASRC:-$RUN/diasources.csv}        # stack diaSource catalog (reliability + trailLength)

# ---- stage 4: link (grid-parallel HelioLinC) ------------------------------
# NEO-targeted grid: r 1.05-1.58 AU, 49,479 pts (~3x finer in the near-Earth band than heliohypo_all
# and ~2x faster). This is the right grid for >1 deg/day hunting; use heliohypo_all only for a
# general/mixed (incl. main-belt) recovery. See [[neo-pipeline-corrections]] HYPOTHESIS GRID.
HELIODIST=${HELIODIST:-heliohypo_neo.txt}   # hypothesis grid (relative to RUN)
NSHARD=${NSHARD:-96}                        # grid shards == parallel heliolinc processes
MINNIGHTS=${MINNIGHTS:-2}                   # fast NEOs cross a single field in ~2 nights -> 2
NPT=${NPT:-3}
MINTIMESPAN=${MINTIMESPAN:-0.05}
EXPTIME=${EXPTIME:-30}                       # exposure (s) -> trail spans MJD ± EXPTIME/2

# ---- stage 5: crossmatch ---------------------------------------------------
XM_TOL_ARCSEC=${XM_TOL_ARCSEC:-3.0}
XM_TOL_DAY=${XM_TOL_DAY:-0.02}

# ---- environments ----------------------------------------------------------
TORCH_ENV="source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn"
LSST_ENV="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; setup lsst_distrib"
mkdir -p "$RUN"
