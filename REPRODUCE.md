# Reproducing the current ADCNN result

The **current** pipeline (default, frozen `adcnn-v2_D-rc1`) is the promoted detector. Its
scientific headline is the faint-fast same-night 2-visit **alert completeness 3.64% → 10.33%**
(+184%, clean-24 +192%) at held purity, on the DM-53195 **blind** fields.

Everything routes through one driver:
```bash
python -m ADCNN.pipelines.run_experiment --stage <stage>     # one stage
python -m ADCNN.pipelines.run_experiment --stages all --dry-run   # the full ordered plan
```
The active pipeline (models + MF_LEN de-bias) is resolved by `ADCNN/config.py`
(`ADCNN_PIPELINE` selects; default `current`). Env: `asteroid_cnn` conda for torch/CPU stages;
the LSST stack (`loadLSST.sh` + `setup lsst_distrib`) only for the Butler data/detect stages.

## 1. One-command verdict (CPU, instant) — the headline
```bash
python -m ADCNN.pipelines.run_experiment --stage report
```
Reads the committed per-field pair caches and prints the frozen-op blind table (ALL / off-ecl /
ecliptic) for the prior baseline vs current. Expected (ALL): `3.64% → 10.33% (+184%)`, purity
`86.1 → 88.6%`. This is the durable evidence; it does **not** re-run detection.

## 2. Cross-domain detector diagnostic (CPU) — the Evaluation notebooks
```bash
# current detector on the legacy DM-53881 test set (out-of-domain by design; recall is lower here):
jupyter nbconvert --to notebook --execute --inplace Evaluation/Evaluation.ipynb
# prior baseline, for comparison:
jupyter nbconvert --to notebook --execute --inplace Evaluation/Evaluation_legacy_v1.ipynb
```
Catalog-based (no NN inference); deterministic from the committed catalogs
(`Evaluation/catalogs_current/` and `Evaluation/catalogs/`). No manual path/version edits —
defaults resolve from the active pipeline. The notebook also surfaces the product headline,
sourced from `models/v2_D/v2_D_release.json` (not from the cross-domain recall).

## 3. Rebuild the frozen models from scratch (GPU, ~1–1.5 days on the 1-node cap)
The model half (stage-1 fine-tune → stage-2 refit → MF_LEN → blind detect → verdict) is the
SLURM chain in **`ADCNN/pipelines/heliolinc/TRAIN_V2_D_E2E.md`** (driver
`train_v2_D_e2e.sh`). `run_experiment` prints the exact `sbatch` commands:
```bash
python -m ADCNN.pipelines.run_experiment --stages all --dry-run    # shows every stage command
python -m ADCNN.pipelines.run_experiment --stage train-stage1 --submit   # actually submit (GPU)
```
After blind detection, apply the de-bias and regenerate the verdict:
```bash
python -m ADCNN.pipelines.run_experiment --stage calibrate-mflen \
  --run-dir .../run_blind_eval --manifests .../run_blind --out .../run_blind_eval_cal --fields 0 1 ... 29
python -m ADCNN.pipelines.run_experiment --stage report
```

## Guarantees (this is a release candidate, not an experiment)
- Frozen op-points (`op_2v_alert.json`, `link_op_point.json`, `op_multinight_discovery.json`)
  are byte-unchanged; a golden-value test pins them.
- The blind set is eval-only; the verdict reads committed caches — no retune after the blind shot.
- `models/v2_D/` is the immutable release (md5s in `v2_D_release.json`); `models/current/` points into it.
- Linker regression anchor: `run_night8731` with the op-point still recovers exactly **2025 NY2**.
