#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=kipac:kipac
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -J freshnight
#SBATCH -o %x_%j.log
set -eo pipefail
# FRESH full-night end-to-end: real LSST cadence (one real night) + Granvik-realistic NEO injection
# restricted to the TRAINING RANGE (detection-SNR 2-10, rate 1-8 deg/day -> trail ~6-50 px), the REAL
# cadence (NO retiming), fresh GPU detect, frozen alert op link, then purity/completeness/wall-time.
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
RUN=$REPO/ADCNN/pipelines/heliolinc/run_freshnight
HL=$REPO/ADCNN/pipelines/heliolinc
# A real held-out night's manifest (real LSST visits/cadence, real diffim FITS, wcs_json-annotated).
# Default: blind field 2 (held-out night 20250723, off-ecliptic, 12 real visits, 768-panel footprint).
SRCMAN=${SRCMAN:-$HL/run_blind/manifest_2.csv}
NOBJ=${NOBJ:-600}
mkdir -p "$RUN"; cd "$REPO"
ts() { date +%s; }
declare -A T

# 0. stage the real-night manifest (real cadence comes from the FITS MJDs; no retiming)
t=$(ts)
[ -s "$RUN/manifest_0.csv" ] || cp "$SRCMAN" "$RUN/manifest_0.csv"
T[manifest]=$(( $(ts) - t ))
NPANEL=$(( $(wc -l < "$RUN/manifest_0.csv") - 1 ))
NVIS=$(tail -n +2 "$RUN/manifest_0.csv" | cut -d, -f2 | sort -u | wc -l)
echo "[freshnight] manifest: $NPANEL panels, $NVIS visits"

source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn

# 1. inject Granvik-realistic NEOs, TRAINING RANGE only, REAL cadence (no --retime-map)
t=$(ts)
python "$HL/sim_orbits.py" --manifest "$RUN/manifest_0.csv" \
  --out-inject "$RUN/inject_0.csv" --out-truth "$RUN/truth_0.csv" \
  --n-objects "$NOBJ" --rate-min 1.0 --rate-max 8.0 --snr-min 2.0 --snr-max 10.0 \
  --m5 24.0 --seed 20250703
T[inject]=$(( $(ts) - t ))

# 2. ADCNN detect with inline injection (FRESH, 4 GPU) -- the timed detection stage
t=$(ts)
python ADCNN/inference/detect_night.py --n-gpus "${NGPU:-4}" \
  --manifest "$RUN/manifest_0.csv" --out "$RUN/adcnn_dets_0.csv" --inject "$RUN/inject_0.csv"
T[detect]=$(( $(ts) - t ))

# 3. LSST diffim mask cut
t=$(ts)
python "$HL/mask_flags.py" --dets "$RUN/adcnn_dets_0.csv" --manifest "$RUN/manifest_0.csv" \
  --out "$RUN/adcnn_dets_masked_0.csv" --workers 32
T[mask]=$(( $(ts) - t ))

# 4. same-night link at the FROZEN ALERT OP (op_2v_alert.json: S>=0.80, mfsnr>=5, chi2<=5, rate[1,8])
t=$(ts)
python -m ADCNN.linking.link_2visit --dets "$RUN/adcnn_dets_masked_0.csv" \
  --known "$HL/run_night8731/known.csv" --out "$RUN/tracks.csv" \
  --op-point "$HL/op_2v_alert.json" --npt 2 --min-epochs 2 --seed-2v chord \
  --alerts-out "$RUN/alerts.jsonl" || true
T[link]=$(( $(ts) - t ))

# 5. rigorous purity/completeness at the op on the training-range truth (validation definition)
t=$(ts)
python "$RUN/measure_freshnight.py" --dir "$RUN" --k 0 --out "$RUN/metrics.json"
T[measure]=$(( $(ts) - t ))

# 6. runtime report (per-stage + per-visit / per-detector-pass)
python - "$RUN" "$NVIS" "$NPANEL" "${T[manifest]}" "${T[inject]}" "${T[detect]}" "${T[mask]}" "${T[link]}" "${T[measure]}" <<'PY'
import json, sys
run, nvis, npanel = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
man, inj, det, mask, link, meas = [int(x) for x in sys.argv[4:10]]
rep = {"tract_night": "see manifest", "n_visits": nvis, "n_detector_panels": npanel,
       "seconds": {"build_manifest": man, "inject": inj, "detect": det, "mask": mask,
                   "link": link, "measure": meas},
       "pipeline_seconds_detect_to_link": det + mask + link,
       "per_visit_detect_s": round(det / nvis, 3) if nvis else None,
       "per_panel_detect_s": round(det / npanel, 4) if npanel else None,
       "total_seconds": man + inj + det + mask + link + meas}
json.dump(rep, open(f"{run}/runtime.json", "w"), indent=2)
print(json.dumps(rep, indent=2))
PY
echo "FRESHNIGHT DONE -> $RUN/{metrics.json,runtime.json,tracks.csv,alerts.jsonl}"
