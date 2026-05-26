# =============================================================================
# NEO RECOVERY pipeline — central config (sourced by the orchestrator + stages)
# Reproduces the run that recovered 4 real NEOs from real Rubin DP2 data (run_neo_wide).
# Every tunable lives here; change RUN_NAME to start a fresh run.
# =============================================================================
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/experiments/heliolinc

# ---- run identity --------------------------------------------------------
RUN_NAME=${RUN_NAME:-run_neo_wide}
RUN=$HL/$RUN_NAME

# ---- discovery field: the DP2 NEO ecliptic-opposition strip --------------
# (found from ss_source: 88 NEO-rate known objects cluster here; scout_ss.py)
RA0=${RA0:-295};  RA1=${RA1:-320}
DEC0=${DEC0:--25}; DEC1=${DEC1:--15}
DAY_START=${DAY_START:-20250620}   # day_obs >= (inclusive)
DAY_END=${DAY_END:-20250720}       # day_obs <  (exclusive)
MIN_RATE=${MIN_RATE:-0.5}          # NEO-rate cut (deg/day) for the truth/target set

# ---- models --------------------------------------------------------------
V7=${V7:-$REPO/models/v7_diffim_scripted.pt}
RF=${RF:-$REPO/models/rf_postproc.pkl}

# ---- stage: NEO hypothesis grid -----------------------------------------
GRID_SRC=${GRID_SRC:-$HL/run_disco/heliohypo_all.txt}   # full grid to filter
RMAX=${RMAX:-1.6}                                        # NEO distance cap (AU) -> excludes belt
HELIODIST=heliohypo_neo.txt                             # produced under $RUN

# ---- stage: detect (ADCNN+RF, GPU) --------------------------------------
RF_THR=${RF_THR:-0.5}
N_GPUS=${N_GPUS:-4}

# ---- stage: link (make_tracklets + grid-parallel HelioLinC) -------------
# clustrad=100000 km is ESSENTIAL for NEOs (close objects spread in state-space);
# 16000 km clustered ZERO NEOs. NEO grid keeps the volume bounded despite the looser radius.
CLUSTRAD=${CLUSTRAD:-100000}
MAXVEL=${MAXVEL:-5.0}        # deg/day — admit fast movers (default belt value 2.0 rejects NEOs)
MAXGCR=${MAXGCR:-2.5}        # tracklet great-circle residual (arcsec)
MAXTIME=${MAXTIME:-3.0}      # max hours between paired detections
MINNIGHTS=${MINNIGHTS:-2}
NPT=${NPT:-3}
MINTIMESPAN=${MINTIMESPAN:-0.05}
NNODE=${NNODE:-4}            # job-array tasks (one node each) — multi-node srun is flaky here
NSHARD=${NSHARD:-90}         # local heliolinc shards per node

# ---- crossmatch ----------------------------------------------------------
XM_TOL_ARCSEC=${XM_TOL_ARCSEC:-3.0}
XM_TOL_DAY=${XM_TOL_DAY:-0.02}
MAXPOSRMS=${MAXPOSRMS:-2000}   # orbit-quality gate (km) for NEW-candidate classification

# ---- environments --------------------------------------------------------
TORCH_ENV="source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh; conda activate asteroid_cnn"
LSST_ENV="source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2026_09/loadLSST.sh; setup lsst_distrib"
mkdir -p "$RUN"
