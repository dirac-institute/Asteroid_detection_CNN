"""ENTRY POINT — train the full two-stage detector end to end: segmentation model NN, then the cutout CNN.

This pipeline is the single authoritative source of the deployed **reg2** recipe: every
training parameter decision lives in :class:`Reg2Recipe` below and is passed explicitly to
the trainer, so the result does NOT depend on ``ADCNN.training.train``'s argparse defaults.
The recipe was captured from the deployed model's run config
(``experiments/diffim_runs/pilot_seg_reg2/config.json``).

Stage 1 (NN): trains segmentation model (UNetResSE + orientation + Hough aggregator) with the reg2 recipe on
  the realistic-trail diffim shards (``--data-sources``), then exports the best checkpoint to
  TorchScript.
Stage 2 (CNN): trains the focal-loss cutout CNN second-stage false-positive filter on cutouts
  from the held-out VALIDATION panels of the freshly trained segmentation model (leakage-safe; never sees the
  test set). For a stronger filter, build a large dedicated cutout set with
  ``ADCNN.training.cnn_postproc.build_cutout_dataset`` and train on that instead.

Defaults reproduce reg2 out of the box on the 4 realistic shards:

    python -m ADCNN.pipelines.train_end_to_end --run-name seg_repro

Override --data-sources / --val-* to train on other data, --epochs for a quick smoke, or
--skip-nn / --skip-cnn to run a single stage. The NN stage needs a GPU + the asteroid_cnn env.
Outputs: ``experiments/diffim_runs/<run>/ckpts/`` + ``<models-dir>/<run>_{segmentation_scripted.pt,cnn_postproc.pt}``.

NOTE: the deployed reg2 used ``init_agg_alpha=0.073``; that trainer flag was removed in the
consolidation, so the Hough aggregator's ``agg_alpha`` now initialises at 0.0 (model default).
``agg_alpha`` is learnable and EMA-excluded, so it converges regardless — this only changes the
optimisation path slightly, not the recipe.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_REALISTIC = REPO / "DATA_DIFFIM_realistic"

# Default training data = the 4 realistic-trail shards reg2 was trained on (shards 0-2 carry
# their split CSV inside the shard dir; shard_3 uses the top-level train/val split).
DEFAULT_DATA_SOURCES = [
    f"{_REALISTIC}/shard_0/train.h5:{_REALISTIC}/shard_0/train.csv",
    f"{_REALISTIC}/shard_1/train.h5:{_REALISTIC}/shard_1/train.csv",
    f"{_REALISTIC}/shard_2/train.h5:{_REALISTIC}/shard_2/train.csv",
    f"{_REALISTIC}/shard_3/train.h5:{_REALISTIC}/shard_3_train.csv",
]
DEFAULT_VAL_H5 = f"{_REALISTIC}/shard_3/train.h5"
DEFAULT_VAL_CSV = f"{_REALISTIC}/shard_3_val.csv"


@dataclass(frozen=True)
class Reg2Recipe:
    """The deployed reg2 detector recipe — the authoritative set of every training decision."""
    # optimisation
    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 30
    batch_size: int = 24
    seed: int = 2026
    # loss: asymmetric focal Tversky + small BCE anchor; orientation aux-loss OFF (the aux head
    # pulled the shared backbone off segmentation — dropping it lifted real fire@truth 71->77%).
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
    # architecture (half-width segmentation model backbone + Hough head)
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


def _nn_train_flags(r: Reg2Recipe) -> list[str]:
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
                    help="'h5:csv' training shards (default = the reg2 realistic shards)")
    ap.add_argument("--val-h5", default=DEFAULT_VAL_H5, help="h5 holding the held-out val panels")
    ap.add_argument("--val-csv", default=DEFAULT_VAL_CSV, help="catalog for the val panels")
    ap.add_argument("--out-root", default=str(REPO / "experiments/diffim_runs"))
    ap.add_argument("--models-dir", default=str(REPO / "models"))
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe epochs (e.g. smoke run)")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the NN training command and exit")
    a = ap.parse_args()

    recipe = Reg2Recipe()
    if a.epochs is not None:
        recipe = replace(recipe, epochs=a.epochs)

    run_dir = Path(a.out_root) / a.run_name
    best = run_dir / "ckpts" / "best.pt"
    scripted = Path(a.models_dir) / f"{a.run_name}_segmentation_scripted.pt"

    # --- Stage 1: NN (reg2 recipe, passed explicitly) ---
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

    # --- Stage 2: cutout CNN (focal FP filter, on the trained segmentation model's held-out val candidates) ---
    if not a.skip_cnn:
        import json
        from ADCNN.training.cnn_postproc import train_cnn_from_val
        # Use the EXACT panels stage 1 held out (split.json) so the FP-filter CNN is never built on
        # panels segmentation model trained on. Fall back to the deterministic multi-source selection only if absent.
        split = run_dir / "split.json"
        if split.exists():
            val_ids = json.loads(split.read_text())["val_panels"]
        else:
            val_ids = sorted(pd.read_csv(a.val_csv)["image_id"].unique())[: recipe.n_val_panels]
        cnn_out = Path(a.models_dir) / f"{a.run_name}_cnn_postproc.pt"
        train_cnn_from_val(scripted, a.val_h5, a.val_csv, val_ids, cnn_out,
                           epochs=recipe.cnn_epochs, fp_cap=recipe.cnn_fp_cap)
        print(f"[stage2-cnn] done -> {cnn_out}", flush=True)
    print("END-TO-END TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
