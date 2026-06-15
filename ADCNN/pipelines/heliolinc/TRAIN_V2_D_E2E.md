> **Provenance / historical record.** `v2_D` is the development name of what is now the **current** default pipeline (`models/current/`, frozen `adcnn-v2_D-rc1`). For the active workflow use the repo-root `REPRODUCE.md` / `TRAINING_PROTOCOL.md` / `EVALUATION_PROTOCOL.md` and `python -m ADCNN.pipelines.run_experiment`. This doc is kept for the development record.

# ADCNN v2_D — end-to-end training that recreates the headline

This is the full chain from the v1 detector to the v2_D blind headline
(faint-fast 2v alert completeness 3.64% → 10.33%, clean-24 +192%). Stages 2–8 are SLURM (ada,
`rubin:commissioning`); the whole chain is ~1–1.5 days wall-clock on the 1-node cap. The frozen
result artifact is `models/v2_D/` + `ADCNN_V2_RESULT.md`; this doc lets you rebuild it from scratch.

`REPO=/sdf/.../Asteroid_detection_CNN`, `HL=$REPO/ADCNN/pipelines/heliolinc`. Driver:
`bash $HL/train_v2_D_e2e.sh` sequences the SLURM steps with dependencies (edit field lists / accounts
as needed); the steps are also runnable by hand below.

## 0. Trainable v1 init (one-time; training ckpts were purged)
```
# Reconstruct a trainable state-dict from the deployed TorchScript export (strict-load verified):
python - <<'PY'
import torch; from ADCNN.core.detector import UNetResSEOrientHough
j=torch.jit.load("models/segmentation_model.pt",map_location="cpu").state_dict()
m=UNetResSEOrientHough(widths=[24,48,96,192,384],kernel_lens=[11,21,41],n_angles=12)
m.load_state_dict(j,strict=True)
torch.save({"model":m.state_dict(),"epoch":0},"models/seg_v1_trainable_init.pt")
PY
```

## 1. Non-blind DM-53195 dev set (training substrate; MUST be blind-disjoint)
```
# Off-ecliptic + ecliptic DM-53195 field-nights, tract-disjoint from run_blind. NOTE/FIX: enforce
# (visit,detector)-EXPOSURE disjointness, not just tracts (rc1 leaked 12 boundary-CCD panels into 2
# blind fields via shared night 20250723 -- non-inflating, but exclude for a clean build):
python build_realfp_manifests.py --collection LSSTCam/runs/DRP/20250421_20250921/d_2025_11_10/DM-53195 \
  --out-dir run_dev --from-diffim-cadence cadence_diffim.csv --exclude-fields-from run_blind/fields.csv \
  --exclude-mode tract  ...   # + (recommended) drop any (visit,detector) shared with run_blind
# then per field: retime_cadence.py ; annotate_manifest_wcs.py --run run_dev ; sim_orbits.py (--retime-map)
```

## 2. Stage-1 fine-tune (the win: hard-positive domain adaptation)
```
# v2_D = oversample the stack-found/ADCNN-missed pool (stk-balance 0.85), init from v1, low LR.
# First build the canonical-contract fine-tune H5 (build_ft_dataset.py: catalog/detect[stack-env]/assemble),
# then:
cd $HL/run_ft && sbatch --export=ALL,RUN_NAME=v2_D,LR=5e-5,STKBAL=0.85 variant.slurm
#  == python -m ADCNN.training.train --init-from models/seg_v1_trainable_init.pt --skip-cnn-equiv
#       --data-sources run_ft/train.h5:run_ft/train.csv --data-h5 run_ft/val.h5 --data-csv run_ft/val.csv
#       --epochs 10 --lr 5e-5 --stk-balance 0.85 --intensity-aug
python -m ADCNN.inference.export --ckpt experiments/diffim_runs/v2_D/ckpts/best.pt \
  --out $HL/run_ft/v2_D_segmentation_scripted.pt --no-optimize
```
Detector ladder check (`run_dev/v2_detector_ladder.md`): faint-fast per-sighting recall 22.9→27.6%.

## 3. Stage-2 refit (REQUIRED after stage-1 changes — leakage-clean panels)
```
python build_ft_dataset.py --stage catalog --run run_dev --out run_ft_cnn \
  --exclude-catalog run_ft/ft_catalog.csv --panels-train 500 --panels-val 150   # disjoint from stage-1
# (then --stage detect [stack env] and --stage assemble [asteroid_cnn])
cd $HL/run_ft && sbatch refit_stage2.slurm     # train_cnn_with_calibration on the v2_D scripted seg
#   -> run_ft/v2_D_cnn_postproc.pt  (256G; smaller fp_cap if mem-bound)
```

## 4. MF_LEN trail-length de-bias re-derivation (REQUIRED — v2_D ends-bloom differs)
```
# Fit raw_mf_length ~= slope*L_true + offset on dev injections (match 10px, faint-fast L 6-60px),
# field-held-out. v2_D result: offset 7.67, slope 0.9425 (v1 was 33.4/0.887).
# (recompute uses these; or set ADCNN_MF_LEN_OFFSET/SLOPE at detection.)
```

## 5. Blind shot (single, pre-registered; run_blind WRITE-PROTECTED)
```
cd $HL/run_ft && sbatch --export=ALL,RUN=$HL/run_blind,\
  SEGMODEL=$REPO/models/v2_D/segmentation_scripted.pt,CNNMODEL=$REPO/models/v2_D/cnn_postproc.pt,\
  OUTDIR=$HL/run_blind_v2eval -J det_v2blind --array=0-19,24-29 detect_v2full.slurm
python run_dev/recompute_lendb.py --src run_blind_v2eval --manifests run_blind \
  --out run_blind_v2eval_cal --offset 7.67 --slope 0.9425 --fields 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 24 25 26 27 28 29
# symlink run_blind inject/truth/manifest/retime into run_blind_v2eval_cal (so the scorer can label tp/fp)
PYTHONPATH=$REPO python -m ADCNN.evaluation.summarize_results     # -> the headline table (v1 vs v2_D, all/off-ecl/ecliptic)
```

## Headline reproduced (release reference)
```
ALL    v1 C_ff 3.64% -> v2_D 10.33% (+184%)   purity 86.1->88.6%
clean-24 (leaked 0,1 excluded)   3.68% -> 10.74% (+192%)   purity 88.5%
```
See `ADCNN_V2_RESULT.md` (arc + caveats), `ADCNN_V2_MFLEN_DECISION.md`, `models/v2_D/v2_D_release.json`.
For evaluation on the *old* DM-53881 test set (cross-domain detector metrics), see
`Evaluation/Evaluation_v2D_test.*` (built with `RUN_NAME=v2_D` + the MF_LEN env via `eval_end_to_end.slurm`).
