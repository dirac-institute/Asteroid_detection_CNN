# outputs/ — the ONE runtime output location

Everything the pipeline writes at runtime lands here (gitignored except this file
and `logs/.gitkeep`). Layout since 2026-07-14; override the root with `ADCNN_OUTPUTS`.
Code must never write into the package tree — `ADCNN/config.py` exposes
`OUTPUTS` / `outputs_dir()` for scripts.

```
outputs/
├── runs/             night + campaign run directories (index below)
│   └── _archive/     closed campaigns, kept for provenance (index below)
├── logs/             slurm logs (#SBATCH -o outputs/logs/…; SUBMIT FROM THE REPO ROOT)
├── training_runs/    model-training provenance (diffim_runs/v2_D ckpts; release weights live in git at models/v2_D/)
├── query_snapshots/  Butler query snapshots (cadence*.csv, field_*.csv) — regenerable
└── attic/            preserved-but-parked (nn_experimentation/ scaffold)
```

The tracked *machinery* of mixed runs (slurm scripts, the 82 gzipped validation
caches, frozen reductions) stays in the package tree at
`ADCNN/pipelines/heliolinc/run_{lambda,blind,blind_v2eval_cal,dev,ft,freshnight,truth}/`;
only the bulky regenerable data lives here.

## runs/ index (live)

New nights land as `runs/run_night_<night>/` (the `./adcnn night` default).

| dir | what it is | why it stays |
|---|---|---|
| `run_embargo_0625` … `run_embargo_0701` | the six delivered 2026-06 embargo science nights (+0701 stub) | delivered science; alert provenance |
| `run_embargo_night`, `run_embargo_2v` | embargo end-to-end + 2v-linking working dirs | same campaign |
| `run_night8731` | canonical DRP tract-night run | `link_2visit` CLI defaults point here |
| `run_freshnight` | fresh full-night end-to-end benchmark (commit 59c8daa) | documented headline; tracked slurm writes here |
| `run_lambda` | 82-field validation sweep bulk (caches promoted → package tree) | cache regeneration source |
| `run_blind`, `run_blind_v2eval`, `run_blind_v2eval_cal` | blind-test protocol runs (v1 + v2_D eval/calibration) | blind set = eval-only forever |
| `run_blind_v1_purged` | quarantined v1 blind field set | referenced by BLIND_TEST_REPORT.md |
| `run_ft`, `run_ft_cnn` | v2_D stage-1/stage-2 fine-tune working dirs | training provenance |
| `run_dev` | development/scratch run referenced across QA scripts | live defaults |
| `run_band` | 14-tract × 10-night band campaign | `validate_candidate` default; FPP calib basis |
| `run_2v_0706` | night-60862 2v run | FPP calibration basis (`link_fpp.json`) |
| `run_test2` | multi-night test incl. `sorcha/baseline_v2.0_1yr.db` | `retime_cadence`/`recovery_metrics` defaults |
| `run_realfp` | real-FP manifest runs | `build_realfp_manifests`/`count_realfp` defaults |
| `alert_sweep` | 2v alert-op sweep grids | `summarize_alert_sweep` default |

## runs/_archive/ index (closed campaigns — conclusions recorded, dirs unreferenced by code)

| dir(s) | campaign | outcome |
|---|---|---|
| `run_camp1`…`run_camp5`, `run_p0`…`run_p3`, `run_pair`, `run_box` | 2026-05/06 same-night pilot + sweep campaigns | fed the λ-campaign op-point work |
| `run_night` | first one-night pipeline run (2026-05-31) | superseded by `run_night8731` + `run_night_<night>` convention |
| `run_lambda_pilot` | λ-campaign pilot | superseded by `run_lambda` |
| `run_neo5658_0717`, `run_neo6328_0713`, `run_neoband_0717` | 2026-06-08 known-NEO recovery runs | recoveries documented (2014 HR161 etc.) |
| `run_2v_clean`, `run_2v_test` | 2v confidence-veto development | veto shipped 2026-07-02 (schema 1.4/1.5) |
| `run_h2h`, `run_h2h_sn`, `run_h2h_sn2` | ADCNN-vs-stack head-to-head | headline recorded; `h2h_metrics.py` takes explicit paths |
| `run_real_main`, `run_valid_main`, `run_sweep_neo` | 2026-06-28/29 COSMOS + ecliptic discovery runs | 0 NEOs, chance-wall re-confirmed |

Nothing in `_archive/` is deleted — move a dir back up if a campaign reopens.
