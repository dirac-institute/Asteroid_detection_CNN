# outputs/ — the ONE runtime output location

Everything the pipeline writes at runtime lands here (gitignored except this file
and logs/.gitkeep). Layout since 2026-07-14; override the root with `ADCNN_OUTPUTS`.

- `runs/` — night + campaign run directories (`run_embargo_*`, `run_lambda` bulk,
  `run_ft*`, `run_blind*` bulk, …). The tracked machinery of mixed runs (slurm
  scripts, validation caches, frozen reductions) stays in the package tree at
  `ADCNN/pipelines/heliolinc/run_{lambda,blind,blind_v2eval_cal,dev,ft,freshnight,truth}/`;
  only the bulky regenerable data lives here.
- `logs/` — Slurm job logs (`#SBATCH -o outputs/logs/…`; submit from the repo root).
- `training_runs/` — model-training run dirs (checkpoints, history). Contains the
  v2_D training provenance (`diffim_runs/v2_D`); the frozen release weights live
  in git at `models/v2_D/`.
- `query_snapshots/` — Butler query snapshots (cadence/field CSVs), regenerable.
- `attic/` — preserved-but-parked items (e.g. `nn_experimentation/`, an uncommitted
  scaffold for a future `nn-experimentation` branch).

Convention: code must never write into the package tree. `ADCNN/config.py`
exposes `OUTPUTS` / `outputs_dir()` for scripts.
