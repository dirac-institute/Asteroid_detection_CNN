# Reproducing the real-asteroid (test_real) result from zero

End-to-end pipeline. Heavy steps run on SLURM; analysis/plots are in the
notebooks (`Evaluation.ipynb` synthetic, `Evaluation_Real.ipynb` real). All
logic is tracked in `ADCNN/` with `python -m` CLIs; these scripts are thin
SLURM wrappers. Two conda envs are used:

* **Butler / dataset build** — LSST stack:
  `source /cvmfs/sw.lsst.eu/.../loadLSST.sh && setup lsst_distrib`
* **NN inference / training / analysis** — `conda activate asteroid_cnn`

Set `REPO_DIR` (defaults to the repo root these scripts live in) and provide
the real fast-mover catalog CSV (`FieldID,detector,RA_deg,Dec_deg,
RARateCosDec_deg_day,DecRate_deg_day,exposure_time,x,y,angle,trail_length,
ObjID,speed_deg_day,...`).

## Order

0. **Leakage check** (no `(visit,detector)` of the real catalog may appear in
   `DATA/`/`DATA_DIFFIM/` train+test) — see `Evaluation_Real.ipynb` §0.

0b. **Pipeline self-check** (CPU, ~20 s, no DATA/GPU) — asserts the lazy
   `ADCNN.inference` API resolves, `DEFAULT_THR==0.50`, the 72-feature RF
   contract is in lock-step with the promoted RF, and the promoted scripted
   v7 runs Stage-1→Stage-2 end to end. `SKIP` (exit 0) in a fresh checkout
   with no ckpts; `PASS`/`FAIL` once artifacts exist. Re-run it right after
   step 8 to confirm the promotion didn't break the contract.
   `python ADCNN/scripts/test_real/validate_pipeline.py`

1. **Scan Butler availability** → `manifest.csv`
   `sbatch ADCNN/scripts/test_real/slurm_build.sh scan  /path/real.csv`

2. **Build the real diffims** → `DATA_DIFFIM/test_real/{test.h5,test.csv,panels.csv}`
   `sbatch ADCNN/scripts/test_real/slurm_build.sh build /path/real.csv`

3. **Score v7+V2 vs the stack** (20-shard GPU array, auto-merges)
   `sbatch ADCNN/scripts/test_real/slurm_real_eval.sh`
   → `experiments/diffim_runs/test_real/results/summary.txt`

4. **FP analysis** (genuine-vs-stack-overlap, SNR-gain, threshold sweep)
   `sbatch ADCNN/scripts/test_real/slurm_fp_pipeline.sh`

5. **Fine-tune v7** (precision-tilt hard-neg, resumes shipped weights)
   `sbatch ADCNN/scripts/test_real/slurm_finetune.sh`

6. **FT eval + RF retrain** (dump-empty/-syn with FT model → `fp-fix`)
   `sbatch ADCNN/scripts/test_real/slurm_ft_eval.sh`

7. **Synthetic bar gate** (promote only if it holds ≈840 cTP / ≤~10k cFP)
   `sbatch ADCNN/scripts/test_real/slurm_bar.sh`

8. **Promote** (if bar holds): back up + replace the shipped ckpts
   `bash ADCNN/scripts/test_real/promote.sh`

Model/RF/data live outside git by repo convention (`DATA_DIFFIM/`,
`checkpoints/`, `experiments/` are gitignored); only ADCNN code + notebooks
are tracked. Results land in `experiments/diffim_runs/test_real/results/`.
