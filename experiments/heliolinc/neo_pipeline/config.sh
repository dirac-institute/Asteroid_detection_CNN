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
MEAS_LENGTH_MIN=${MEAS_LENGTH_MIN:-6}    # de-biased trail length px (the `length` col == len_db); 6px ≈ 1 deg/day,
# the fast-mover floor we target + matches CLEAN_LENDB_MIN. WAS 40 -> dropped ALL fast movers (their length
# is median ~10px, 95th ~29px; <0.1% reach 40) -> 0 known recovered. At 6 -> ~224k dets measured, the full
# >1 deg/day population (144 linkable known fast objects available vs 0).
MEAS_WORKERS=${MEAS_WORKERS:-60}

# ---- stage 3: clean FP (dual-stream, validated on truth) -------------------
# diaSources (stack): Rubin real/bogus RELIABILITY cut (TP-safe: 95.7% TP / 93.6% FP removed).
# ADCNN: NO real/bogus (its SNR-floor + trail-ceiling would drop faint/fast trails) -> keep at a
#        LOW score threshold and let LINKING reject FP. See [[realbogus-fp-filter-limits]].
ADCNN_SCORE_MIN=${ADCNN_SCORE_MIN:-0.3}      # low -> preserve low-SNR / faint trails
DIA_RELIABILITY_MIN=${DIA_RELIABILITY_MIN:-0.5}   # real/bogus, diaSources ONLY
CLEAN_LENDB_MIN=${CLEAN_LENDB_MIN:-6}        # de-biased length px; ~6px ≈ 1 deg/day (fast movers)
DIASRC=${DIASRC:-$RUN/diasources.csv}        # stack diaSource catalog (reliability + trailLength)

# ---- stage 4: link (grid-parallel HelioLinC) ------------------------------
HELIODIST=${HELIODIST:-heliohypo_all.txt}   # hypothesis grid (relative to RUN); 109,983 pts
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
