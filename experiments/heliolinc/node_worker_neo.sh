#!/bin/bash
# One node of the multi-node NEO hunt: process this node's round-robin slice of the NEO grid with
# 90 local heliolinc shards, then copy the (small, clustrad-bounded) cluster files to SHARED disk so
# the head node can link_refine across all nodes. Identical math to the single-node hunt -> no
# accuracy change, just N-node parallelism. Driven by srun ($SLURM_PROCID = node index).
set -eo pipefail
source "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/heliolinc/neo_recovery_config.sh"
BIN=$HL/heliolinc2/src
eval "$TORCH_ENV"
# node index + count: works under srun (SLURM_PROCID/NTASKS) or a job array (NODE_IDX/NNODE env)
PROCID=${NODE_IDX:-${SLURM_PROCID:-0}}; NNODE=${NNODE:-${SLURM_NTASKS:-4}}; NSHARD=${NSHARD:-90}
GRID=$RUN/$HELIODIST
SHARED=$RUN/clusters_mn
SCRATCH=/lscratch/$USER/hlmn_${SLURM_JOB_ID}_${PROCID}
mkdir -p "$SHARED" "$SCRATCH"
HDR=$(head -1 "$GRID")
# this node takes global grid lines where (line %% NNODE == PROCID), then splits ITS slice
# round-robin into NSHARD local shards (balanced load within the node too)
tail -n +2 "$GRID" | awk -v node="$PROCID" -v nn="$NNODE" -v ns="$NSHARD" -v dir="$SCRATCH" -v hdr="$HDR" '
  (NR-1)%nn==node { c++; sid=(c-1)%ns; f=sprintf("%s/gc_%03d",dir,sid);
                    if(!(sid in seen)){print hdr > f; seen[sid]=1} print >> f }'
MJDREF=$(python -c "import pandas as pd;print(round(pd.read_csv('$RUN/std_dets.csv').mjd.median(),3))")
cd "$RUN"
pids=()
for f in "$SCRATCH"/gc_*; do
  s=${f##*gc_}
  "$BIN/heliolinc" -dets pairdets.csv -pairs pairs.txt -mjd "$MJDREF" -obspos Earth1day2020s_02a.txt \
    -heliodist "$f" -clustrad "${CLUSTRAD:-100000}" -npt "${NPT:-3}" -minobsnights "${MINNIGHTS:-2}" \
    -mintimespan "${MINTIMESPAN:-0.05}" \
    -out "$SCRATCH/hl_clusters_${s}.csv" -outsum "$SCRATCH/hl_summary_${s}.csv" >"$SCRATCH/hl_${s}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
# publish this node's shards to shared disk (prefix with node id to avoid cross-node name clash)
for c in "$SCRATCH"/hl_clusters_*.csv; do
  s=${c##*hl_clusters_}; s=${s%.csv}
  [ -s "$c" ] && cp "$c" "$SHARED/hl_clusters_${PROCID}_${s}.csv" && cp "$SCRATCH/hl_summary_${s}.csv" "$SHARED/hl_summary_${PROCID}_${s}.csv"
done
rm -rf "$SCRATCH"
echo "NODE $PROCID DONE (shard failures=$fail, $(ls "$SHARED"/hl_clusters_${PROCID}_*.csv 2>/dev/null | wc -l) shards published)"
