#!/bin/bash
# ADCNN v2_D end-to-end TRAINING -> blind headline (the model half: stages 2-5 of TRAIN_V2_D_E2E.md).
# Sequences the SLURM steps with afterok dependencies. PRECONDITION: the dev + fine-tune datasets
# already exist (run_dev/, run_ft/{train,val}.h5, run_ft_cnn/{train,val}.h5 -- built per stages 1&3
# of TRAIN_V2_D_E2E.md, which need the LSST stack + Butler). This driver does NOT rebuild datasets.
# Usage: bash train_v2_D_e2e.sh   (review job IDs; edit MF_LEN if a fresh fit differs from 7.67/0.9425)
set -euo pipefail
REPO=/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN
HL=$REPO/ADCNN/pipelines/heliolinc
ACCT=rubin:commissioning
OFF=7.67; SLOPE=0.9425        # v2_D MF_LEN de-bias (re-fit on dev injections; see stage 4)
cd "$HL/run_ft"

for f in train.h5 val.h5 "$HL/run_ft_cnn/train.h5" "$HL/run_ft_cnn/val.h5" "$REPO/models/seg_v1_trainable_init.pt"; do
  [[ -s "$f" ]] || { echo "MISSING precondition: $f (build datasets per TRAIN_V2_D_E2E.md stages 1&3)"; exit 1; }
done

echo "[1] stage-1 fine-tune (v2_D, hard-positive)";  J1=$(sbatch --parsable --account=$ACCT \
  --export=ALL,RUN_NAME=v2_D,LR=5e-5,STKBAL=0.85 variant.slurm)
echo "    job $J1  (on done: export ckpt -> v2_D_segmentation_scripted.pt; the variant.slurm does this)"
echo "[2] stage-2 refit (after stage-1)";            J2=$(sbatch --parsable --account=$ACCT \
  --dependency=afterok:$J1 refit_stage2.slurm)
echo "    job $J2"
echo "[3] blind detect (after stage-2)";             J3=$(sbatch --parsable --account=$ACCT --dependency=afterok:$J2 \
  --export=ALL,RUN=$HL/run_blind,SEGMODEL=$HL/run_ft/v2_D_segmentation_scripted.pt,\
CNNMODEL=$HL/run_ft/v2_D_cnn_postproc.pt,OUTDIR=$HL/run_blind_v2eval -J det_v2blind --array=0-19,24-29 detect_v2full.slurm)
echo "    job $J3"
cat <<EOF

After job $J3 completes, finish on the login node (fast, CPU):
  python $HL/run_dev/recompute_lendb.py --src $HL/run_blind_v2eval --manifests $HL/run_blind \\
    --out $HL/run_blind_v2eval_cal --offset $OFF --slope $SLOPE \\
    --fields 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 24 25 26 27 28 29
  ( cd $HL/run_blind_v2eval_cal && for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 24 25 26 27 28 29; do
      for p in inject truth retime manifest; do ln -sf ../run_blind/\${p}_\$k.csv \${p}_\$k.csv; done; done )
  PYTHONPATH=$REPO python $HL/regen_v2_report.py     # -> the +184% headline table
EOF
echo "E2E_CHAIN_SUBMITTED J1=$J1 J2=$J2 J3=$J3"
