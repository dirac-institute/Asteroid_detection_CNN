#!/bin/bash
#SBATCH --partition=ada
#SBATCH --account=rubin:commissioning
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH -t 8:00:00
#SBATCH -J v2D_s2refit
#SBATCH -o outputs/logs/%x_%j.log
#SBATCH --exclude=sdfada006
set -eo pipefail
# ADCNN v2_D stage-2 refit (ADCNN_V2_SPRINT.md Phase 3, APPROVED). The REQUIRED end-to-end
# recalibration after stage-1 changed: cutout channels recomputed from v2_D's scripted seg model;
# focal CNN trained on leakage-clean dev panels (run_ft_cnn, disjoint from the 1,429 stage-1 panels);
# FP-budget threshold calibrated on the held-out cnn val split. Blind fields untouched.
REPO="${ADCNN_REPO:-/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN}"
HL=$REPO/ADCNN/pipelines/heliolinc
source /sdf/data/rubin/user/mrakovci/conda/etc/profile.d/conda.sh
conda activate asteroid_cnn
cd "$REPO"
python - <<'PY'
import os
import pandas as pd
from ADCNN.training.cnn_postproc import train_cnn_with_calibration, FPP_BUDGET
from ADCNN.config import OUTPUTS
RUNS = f"{OUTPUTS}/runs"
seg = f"{RUNS}/run_ft/v2_D_segmentation_scripted.pt"         # v2_D stage-1, FIXED
tr_h5, tr_csv = f"{RUNS}/run_ft_cnn/train.h5", f"{RUNS}/run_ft_cnn/train.csv"
vl_h5, vl_csv = f"{RUNS}/run_ft_cnn/val.h5",   f"{RUNS}/run_ft_cnn/val.csv"
out = f"{RUNS}/run_ft/v2_D_cnn_postproc.pt"
tr_ids = sorted(pd.read_csv(tr_csv)["image_id"].unique())
vl_ids = sorted(pd.read_csv(vl_csv)["image_id"].unique())
print(f"[s2refit] seg={seg}\n[s2refit] train {len(tr_ids)} panels, calib {len(vl_ids)} panels, budget={FPP_BUDGET}", flush=True)
_, info = train_cnn_with_calibration(seg, tr_h5, tr_csv, tr_ids, out,
                                     thr_h5=vl_h5, thr_csv=vl_csv, thr_panel_ids=vl_ids,
                                     fpp_budget=FPP_BUDGET, gpus=1)
import json; json.dump(info, open(out.replace(".pt",".json"),"w"), indent=2)
print(f"[s2refit] DONE -> {out} | threshold={info.get('threshold')} "
      f"combined_recall={info.get('combined_recall')} @ {info.get('combined_fp_per_panel')} FP/panel", flush=True)
PY
echo "V2D_S2REFIT_DONE"
