# =============================================================================
# NEO RECOVERY pipeline — central config (sourced by the orchestrator + stages)
# Reproduces the run that recovered 4 real NEOs from real Rubin DP2 data (NEO_large).
# Every tunable lives here; change RUN_NAME to start a fresh run.
# =============================================================================
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/experiments/heliolinc

# ---- run identity --------------------------------------------------------
RUN_NAME=${RUN_NAME:-NEO_large}
RUN=$HL/$RUN_NAME

# ---- discovery field: the DP2 NEO ecliptic-opposition strip --------------
# (found from ss_source: 88 NEO-rate known objects cluster here; scout_ss.py)
RA0=${RA0:-295};  RA1=${RA1:-320}
DEC0=${DEC0:--25}; DEC1=${DEC1:--15}
DAY_START=${DAY_START:-20250620}   # day_obs >= (inclusive)
DAY_END=${DAY_END:-20250720}       # day_obs <  (exclusive)
MIN_RATE=${MIN_RATE:-0.5}          # NEO-rate cut (deg/day) for the truth/target set

# ---- models --------------------------------------------------------------
SEG_MODEL=${SEG_MODEL:-$REPO/models/segmentation_model.pt}

# ---- stage: NEO hypothesis grid -----------------------------------------
GRID_SRC=${GRID_SRC:-$HL/run_disco/heliohypo_all.txt}   # full grid to filter
RMAX=${RMAX:-1.6}                                        # NEO distance cap (AU) -> excludes belt
HELIODIST=heliohypo_neo.txt                             # produced under $RUN

# ---- stage: detect (ADCNN+RF, GPU) --------------------------------------
N_GPUS=${N_GPUS:-4}

# ---- stage: tracklet construction ---------------------------------------
# TRACKMODE=pair : make_tracklets cross-visit pairing (>=2 dets/night; robust to noisy endpoints) [default]
# TRACKMODE=trail: trail_tracklets one-trail=one-tracklet (unlocks single-coverage nights; best for
#   fast movers >=6px). NEEDS accurate trail endpoints -> Veres trail-centroid astrometry (#4);
#   current ADCNN endpoints have ~50% rate / ~25 deg direction error, so keep pair as default until #4.
TRACKMODE=${TRACKMODE:-pair}
LENDB_MIN=${LENDB_MIN:-6}    # trail mode: min trail length (px ~= 1 deg/day) to admit as a tracklet

# ---- stage: link (grid-parallel HelioLinC) ------------------------------
# clustrad=100000 km is ESSENTIAL for NEOs (close objects spread in state-space);
# 16000 km clustered ZERO NEOs. NEO grid keeps the volume bounded despite the looser radius.
CLUSTRAD=${CLUSTRAD:-100000}
MAXVEL=${MAXVEL:-5.0}        # deg/day — admit fast movers (default belt value 2.0 rejects NEOs)
MAXGCR=${MAXGCR:-2.5}        # tracklet great-circle residual (arcsec)
MAXTIME=${MAXTIME:-3.0}      # max hours between paired detections
MINNIGHTS=${MINNIGHTS:-2}    # link finds >=2-night clusters; NEW-candidate classify still requires >=3
NPT=${NPT:-3}
MINTIMESPAN=${MINTIMESPAN:-0.05}
NNODE=${NNODE:-4}            # job-array tasks (one node each) — multi-node srun is flaky here
NSHARD=${NSHARD:-90}         # local heliolinc shards per node

# ---- crossmatch / NEW-candidate gate ------------------------------------
XM_TOL_ARCSEC=${XM_TOL_ARCSEC:-3.0}
XM_TOL_DAY=${XM_TOL_DAY:-0.02}
# MAXPOSRMS: orbit-quality gate (km) for NEW-candidate classification.
# *** CORRECTED 2026-05-27: was 2000 (a SLOW-main-belt value). REAL fast NEOs link with posRMS
# ~9,000-54,000 km (NEO_large production tracks incl. the 4 recovered NEOs; median 39k); synthetic
# >=1 deg/day NEOs ~6,000-26,000 km. A 2000 km cut classifies EVERY real NEO as SPURIOUS -> silently
# kills the undiscovered-NEO (NEW-CANDIDATE) output. posRMS is a WEAK discriminator for fast NEOs
# (real 10-50k overlaps trash chance-clusters at 40-50k), so the gate is loose; real false-link
# rejection is downstream vetting (self-confirmation / arc-extension / MPC, tasks #2,#3). ***
MAXPOSRMS=${MAXPOSRMS:-60000}

# ---- environments --------------------------------------------------------
TORCH_ENV="source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn"
LSST_ENV="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; setup lsst_distrib"
mkdir -p "$RUN"
