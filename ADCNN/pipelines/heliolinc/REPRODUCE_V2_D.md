> **Provenance / historical record.** `v2_D` is the development name of what is now the **current** default pipeline (`models/current/`, frozen `adcnn-v2_D-rc1`). For the active workflow use the repo-root `REPRODUCE.md` / `TRAINING_PROTOCOL.md` / `EVALUATION_PROTOCOL.md` and `python -m ADCNN.pipelines.run_experiment`. This doc is kept for the development record.

# Reproducing the ADCNN v2_D release-candidate result

Frozen artifacts (md5 in `models/v2_D/v2_D_release.json`):
- Stage-1 (scripted): `models/v2_D/segmentation_scripted.pt` · trainable: `models/v2_D/stage1_best_trainable.pt`
- Stage-2 (cutout CNN): `models/v2_D/cnn_postproc.pt`
- Trail-length de-bias (v2_D-specific): MF_LEN **offset 7.67, slope 0.9425**
- Frozen alert op (UNCHANGED from v1): `op_2v_alert.json` (S≥0.80, mf_snr≥5, chi2≤5, len_db≥6, rate[1,8], top-50)
- Blind set: `run_blind/` (26 DM-53195 fields, write-protected) + v1 caches `run_blind/_nomfsnr_cache`
- v2_D blind dets: `run_blind_v2eval/` (raw) → `run_blind_v2eval_cal/` (MF_LEN-recomputed)
- Verdict caches (durable): `run_blind_v2eval_cal/_nomfsnr_cache/*_smin0.8_v3exact.json`

Env: `source /sdf/group/rubin/sw/loadLSST.sh && setup lsst_distrib` for the stack stages;
`asteroid_cnn` conda for detection/scoring. `REPO=/sdf/.../Asteroid_detection_CNN`, `HL=$REPO/ADCNN/pipelines/heliolinc`.

## 0. One-command verdict regeneration (instant, from saved caches)
```
PYTHONPATH=$REPO python $HL/regen_v2_report.py
```
Prints the frozen-op v1-vs-v2_D table (ALL / off-ecl / ecliptic). Reads only the small pair caches;
no GPU, no re-detection. First run regenerates any missing v2_D cache via `eval_field_exact` (smin 0.80).

## 1. Full reproduction from the frozen models (GPU; ~hours, 1-node serial)
```
# (a) detect the 26 blind fields with the v2_D detector (writes run_blind_v2eval/, run_blind untouched)
cd $HL/run_ft && sbatch --export=ALL,RUN=$HL/run_blind,\
  SEGMODEL=$REPO/models/v2_D/segmentation_scripted.pt,\
  CNNMODEL=$REPO/models/v2_D/cnn_postproc.pt,\
  OUTDIR=$HL/run_blind_v2eval -J det_v2blind --array=0-19,24-29 detect_v2full.slurm
#     (detect_v2full.slurm: discover_stream --seg-model $SEGMODEL --cnn $CNNMODEL --cnn-thr 0.50, then mask_flags)
#     length_raw is stored in the output (catalog.py emits it; ADCNN_MF_LEN_* env optional).

# (b) apply v2_D trail-length de-bias (recompute len_db + endpoints; no re-detect)
PYTHONPATH=$REPO python $HL/run_dev/recompute_lendb.py \
  --src $HL/run_blind_v2eval --manifests $HL/run_blind --out $HL/run_blind_v2eval_cal \
  --offset 7.67 --slope 0.9425 --fields 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 24 25 26 27 28 29

# (c) symlink injection truth into the scoring dir (so eval_field_exact can label tp/fp)
cd $HL/run_blind_v2eval_cal && for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 24 25 26 27 28 29; do \
  for p in inject truth retime manifest; do ln -sf ../run_blind/${p}_$k.csv ${p}_$k.csv; done; done

# (d) verdict
PYTHONPATH=$REPO python $HL/regen_v2_report.py
```

## 2. Expected blind verdict (release reference)
```
split        model   tp   fp  purity   C_ff  alerts/fn
ALL          v1     279   45  86.1%   3.64%       12.5
ALL          v2_D   753   97  88.6%  10.33%       32.7
off-ecl      v1     226    2  99.1%   3.55%       11.4
off-ecl      v2_D   588   15  97.5%   9.04%       30.1
ecliptic     v1      53   43  55.2%   4.19%       16.0
ecliptic     v2_D   165   82  66.8%  18.32%       41.2
```
v1 must reproduce its `BLIND_TEST_REPORT.md` numbers exactly (harness check).

## 3. Trail-length de-bias re-derivation (how 7.67/0.9425 was obtained)
Fit `raw_mf_length ≈ slope·L_true + offset` on non-blind dev injections (match dets↔inject 10px,
faint-fast L 6–60px), field-held-out. Re-run on `run_dev/v2_D_s2` vs `run_dev/inject_*.csv`/`truth_*`.
Stage-2 was refit (canonical `train_cnn_with_calibration` on the leakage-clean `run_ft_cnn` H5,
disjoint from the 1,429 stage-1 panels) — see `run_ft/refit_stage2.slurm`.

## Provenance docs
`ADCNN_V2_SPRINT.md` (charter) · `ADCNN_V2_MFLEN_DECISION.md` (recalibration decision) ·
`ADCNN_V2_RESULT.md` (result + arc) · `run_dev/v2_detector_ladder.md` · `run_dev/v2_D_dev_gate.md`.
