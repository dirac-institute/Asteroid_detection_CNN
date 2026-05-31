"""ENTRY POINT — train the full two-stage detector end to end: NN segmentation, then cutout CNN.

This pipeline is the single authoritative source of the deployed training recipe: every
training decision lives in :class:`TrainingRecipe` below and is passed explicitly to the
trainer, so the result does NOT depend on ``ADCNN.training.train``'s argparse defaults.

Stage 1 (NN): trains the segmentation model (UNetResSE + orientation + Hough aggregator)
  on the training shards (``--data-sources``), then exports the best checkpoint to TorchScript.
Stage 2 (CNN): trains the focal-loss cutout CNN second-stage false-positive filter on
  cutouts from a held-out training set; sets the operating threshold by the combined-FPP
  budget on a separate calibration set.

Default datasets follow the layout produced by ``ADCNN/pipelines/slurm/make_datasets.slurm``:
the canonical roles are ``train`` / ``val`` (stage-1 train/val) and ``cnn_train`` / ``cnn_val``
(stage-2 train/calibration), with ``test`` held out for evaluation.

    python -m ADCNN.pipelines.train_end_to_end --run-name seg \\
        --data-sources DATA_DIFFIM/train.h5:DATA_DIFFIM/train.csv \\
        --val-h5  DATA_DIFFIM/val.h5  --val-csv  DATA_DIFFIM/val.csv \\
        --cnn-train-h5 DATA_DIFFIM/cnn_train.h5 --cnn-train-csv DATA_DIFFIM/cnn_train.csv \\
        --cnn-val-h5  DATA_DIFFIM/cnn_val.h5   --cnn-val-csv  DATA_DIFFIM/cnn_val.csv

Override ``--data-sources``/``--val-*`` to train on other data, ``--epochs`` for a quick
smoke run, or ``--skip-nn`` / ``--skip-cnn`` to run a single stage. The NN stage needs a GPU
+ the ``asteroid_cnn`` env.

Outputs: ``<out-root>/<run>/ckpts/`` + ``<models-dir>/<run>_{segmentation_scripted.pt,cnn_postproc.pt}``
plus a JSON sidecar carrying the calibrated CNN threshold.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_DATA = REPO / "DATA_DIFFIM"

# Default training data layout: 4 training shards (shards 0-2 carry their split CSV inside the
# shard dir; shard_3 uses a top-level train/val split CSV alongside the shard h5).
DEFAULT_DATA_SOURCES = [
    f"{_DATA}/shard_0/train.h5:{_DATA}/shard_0/train.csv",
    f"{_DATA}/shard_1/train.h5:{_DATA}/shard_1/train.csv",
    f"{_DATA}/shard_2/train.h5:{_DATA}/shard_2/train.csv",
    f"{_DATA}/shard_3/train.h5:{_DATA}/shard_3_train.csv",
]
DEFAULT_VAL_H5 = f"{_DATA}/shard_3/train.h5"
DEFAULT_VAL_CSV = f"{_DATA}/shard_3_val.csv"


@dataclass(frozen=True)
class TrainingRecipe:
    """Authoritative training recipe for the deployed detector — every training decision."""
    # optimisation
    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 30
    batch_size: int = 24
    seed: int = 2026
    # loss: asymmetric focal Tversky + small BCE anchor; orientation aux-loss off by default
    # (it competes with segmentation on the shared backbone).
    aftl_alpha: float = 0.3
    aftl_beta: float = 0.7
    aftl_gamma: float = 1.3
    aftl_bce_anchor: float = 0.1
    lambda_orient: float = 0.0
    # anchor sampling per (fixed-size) epoch
    n_pos_anchors_per_epoch: int = 3000
    n_neg_anchors_per_epoch: int = 900
    n_train_panels: int = 150
    n_val_panels: int = 64
    stk_balance: float = 0.6
    anchor_jitter: int = 48
    tile: int = 128
    # architecture
    widths: tuple[int, ...] = (24, 48, 96, 192, 384)
    kernel_lens: tuple[int, ...] = (11, 21, 41)
    n_angles: int = 12
    # regularisation + EMA
    dropout: float = 0.15
    ema_decay: float = 0.999
    ema_exclude: tuple[str, ...] = ("agg_alpha",)
    intensity_aug: bool = True
    augment: bool = True
    orient_cache_size: int = 24
    num_workers: int = 8
    # stage-2 cutout CNN (focal-loss FP filter)
    cnn_epochs: int = 30
    cnn_fp_cap: int = 600


def _nn_train_flags(r: TrainingRecipe) -> list[str]:
    """Render the recipe as explicit ``ADCNN.training.train`` CLI flags (so the pipeline,
    not the trainer's argparse defaults, defines the model)."""
    flags = [
        "--lr", str(r.lr), "--wd", str(r.wd), "--epochs", str(r.epochs),
        "--batch-size", str(r.batch_size), "--seed", str(r.seed),
        "--aftl-alpha", str(r.aftl_alpha), "--aftl-beta", str(r.aftl_beta),
        "--aftl-gamma", str(r.aftl_gamma), "--aftl-bce-anchor", str(r.aftl_bce_anchor),
        "--lambda-orient", str(r.lambda_orient),
        "--n-pos-anchors-per-epoch", str(r.n_pos_anchors_per_epoch),
        "--n-neg-anchors-per-epoch", str(r.n_neg_anchors_per_epoch),
        "--n-train-panels", str(r.n_train_panels), "--n-val-panels", str(r.n_val_panels),
        "--stk-balance", str(r.stk_balance), "--anchor-jitter", str(r.anchor_jitter),
        "--tile", str(r.tile), "--n-angles", str(r.n_angles),
        "--dropout", str(r.dropout), "--ema-decay", str(r.ema_decay),
        "--orient-cache-size", str(r.orient_cache_size), "--num-workers", str(r.num_workers),
        "--widths", *map(str, r.widths),
        "--kernel-lens", *map(str, r.kernel_lens),
    ]
    if r.ema_exclude:
        flags += ["--ema-exclude", *r.ema_exclude]
    if r.intensity_aug:
        flags.append("--intensity-aug")
    if r.augment:
        flags.append("--augment")
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-sources", nargs="*", default=DEFAULT_DATA_SOURCES,
                    help="'h5:csv' training shards (default = the training shards)")
    ap.add_argument("--val-h5", default=DEFAULT_VAL_H5, help="h5 holding the held-out val panels")
    ap.add_argument("--val-csv", default=DEFAULT_VAL_CSV, help="catalog for the val panels")
    ap.add_argument("--cnn-train-h5", default=None,
                    help="Dedicated stage-2 training set (e.g. the cnn_train dataset). When given, the "
                         "focal cutout-CNN is trained on ALL of its panels instead of the segmentation "
                         "model's held-out val panels. Must be disjoint from the stage-1 train set "
                         "(the make-datasets script guarantees this) so the CNN cutouts are leakage-free.")
    ap.add_argument("--cnn-train-csv", default=None, help="catalog for --cnn-train-h5")
    ap.add_argument("--cnn-val-h5", default=None,
                    help="held-out calibration set for the stage-2 CNN operating threshold "
                         "+ AUC. When given (with --cnn-train-h5), the CNN trains on ALL of "
                         "cnn_train and the threshold is set on cnn_val; otherwise a slice "
                         "of cnn_train is held out for it.")
    ap.add_argument("--cnn-val-csv", default=None, help="catalog for --cnn-val-h5")
    ap.add_argument("--fpp-budget", type=float, default=None,
                    help="combined 5sigma-stack + ADCNN false-positives-per-panel budget "
                         "for the operating point (default: FPP_BUDGET in "
                         "ADCNN.training.cnn_postproc). The CNN score cut is set on "
                         "--cnn-val-h5 so the deduplicated union (stack ∪ ADCNN) has "
                         "exactly this FP/panel. Requires cnn_val to carry "
                         "real_labels_5sigma + stack_detection_5sigma -- build it with "
                         "`make_sim_data --multi-sigma-sets cnn_val --test-sigmas 5`.")
    ap.add_argument("--gpus", type=int, default=1,
                    help="DataParallel across this many GPUs for the stage-2 CNN training step.")
    ap.add_argument("--out-root", default=str(REPO / "runs"))
    ap.add_argument("--models-dir", default=str(REPO / "models"))
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe epochs (e.g. smoke run)")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the NN training command and exit")
    a = ap.parse_args()

    recipe = TrainingRecipe()
    if a.epochs is not None:
        recipe = replace(recipe, epochs=a.epochs)

    run_dir = Path(a.out_root) / a.run_name
    best = run_dir / "ckpts" / "best.pt"
    scripted = Path(a.models_dir) / f"{a.run_name}_segmentation_scripted.pt"

    # --- Stage 1: NN  ---
    if not a.skip_nn:
        cmd = [sys.executable, "-m", "ADCNN.training.train", "--run-name", a.run_name,
               "--out-root", a.out_root, *_nn_train_flags(recipe),
               "--data-sources", *a.data_sources, "--data-h5", a.val_h5, "--data-csv", a.val_csv]
        print("[stage1-nn]", " ".join(cmd), flush=True)
        if a.dry_run:
            return
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, "-m", "ADCNN.inference.export",
                        "--ckpt", str(best), "--out", str(scripted), "--no-optimize"], check=True)
        print(f"[stage1-nn] scripted segmentation model -> {scripted}", flush=True)
    elif a.dry_run:
        return

    # --- Stage 2: cutout CNN (focal FP filter) ---
    if not a.skip_cnn:
        import json
        from ADCNN.training.cnn_postproc import train_cnn_with_calibration, FPP_BUDGET
        cnn_out = Path(a.models_dir) / f"{a.run_name}_cnn_postproc.pt"
        budget = a.fpp_budget if a.fpp_budget is not None else FPP_BUDGET
        if a.cnn_train_h5:
            # Dedicated stage-2 training set: train the FP-filter CNN on ALL of its panels.
            # The segmentation model never trained on these panels (disjoint by construction),
            # so its cutouts here are leakage-free.
            if not a.cnn_train_csv:
                raise SystemExit("--cnn-train-h5 requires --cnn-train-csv")
            cnn_ids = sorted(pd.read_csv(a.cnn_train_csv)["image_id"].unique())
            thr_kw = {}
            if a.cnn_val_h5:
                if not a.cnn_val_csv:
                    raise SystemExit("--cnn-val-h5 requires --cnn-val-csv")
                thr_kw = dict(thr_h5=a.cnn_val_h5, thr_csv=a.cnn_val_csv,
                              thr_panel_ids=sorted(pd.read_csv(a.cnn_val_csv)["image_id"].unique()))
                print(f"[stage2-cnn] calibration set = {a.cnn_val_h5}", flush=True)
            print(f"[stage2-cnn] training on all {len(cnn_ids)} panels of {a.cnn_train_h5}",
                  flush=True)
            _, cnn_info = train_cnn_with_calibration(
                scripted, a.cnn_train_h5, a.cnn_train_csv, cnn_ids, cnn_out,
                epochs=recipe.cnn_epochs, fp_cap=recipe.cnn_fp_cap,
                fpp_budget=budget, gpus=a.gpus, **thr_kw)
        else:
            # Default: the exact panels stage 1 held out (split.json) so the FP-filter CNN is
            # never built on panels the segmentation model trained on.
            split = run_dir / "split.json"
            if split.exists():
                val_ids = json.loads(split.read_text())["val_panels"]
            else:
                val_ids = sorted(pd.read_csv(a.val_csv)["image_id"].unique())[: recipe.n_val_panels]
            _, cnn_info = train_cnn_with_calibration(
                scripted, a.val_h5, a.val_csv, val_ids, cnn_out,
                epochs=recipe.cnn_epochs, fp_cap=recipe.cnn_fp_cap,
                fpp_budget=budget, gpus=a.gpus)
        # Persist the combined-budget operating threshold + diagnostics next to the weights so eval
        # and deployment score THIS model at the FP-budget op-point instead of the baked-in default.
        sidecar = cnn_out.with_suffix(".json")
        sidecar.write_text(json.dumps(cnn_info, indent=2))
        print(f"[stage2-cnn] done -> {cnn_out} "
              f"(threshold={cnn_info.get('threshold')} combined_recall={cnn_info.get('combined_recall')} "
              f"@ {cnn_info.get('combined_fp_per_panel')} FP/panel | budget={budget} "
              f"-> {sidecar.name})", flush=True)
    print("END-TO-END TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
