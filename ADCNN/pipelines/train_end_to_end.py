"""ENTRY POINT — train the full two-stage detector: v7 NN, then the RandomForest.

Stage 1 (NN): trains the v7 detector with the deployed **reg2** recipe
  lambda_orient=0, --dropout 0.15, --wd 1e-4, --intensity-aug, --augment,
  half-width backbone (24 48 96 192 384), EMA excluding agg_alpha,
  on the realistic-trail diffim set (multiple shards via --data-sources).
Then exports the best checkpoint to TorchScript.

Stage 2 (RF): trains the neg5 RandomForest second stage on candidate features from
the held-out VALIDATION panels of the trained v7 (leakage-safe).

    python -m ADCNN.pipelines.train_end_to_end --run-name v7_prod \\
        --data-sources DATA/shard_0/train.h5:DATA/shard_0/train.csv ... \\
        --val-h5 DATA/shard_v/train.h5 --val-csv DATA/shard_v/val.csv

Use --skip-nn / --skip-rf to run a single stage. The NN stage needs a GPU + the
asteroid_cnn env. Outputs: experiments/diffim_runs/<run>/ckpts/ + models/.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-sources", nargs="*", help="'h5:csv' shards for v7 training")
    ap.add_argument("--data-h5", help="single-h5 alternative to --data-sources")
    ap.add_argument("--data-csv")
    ap.add_argument("--val-h5", required=True, help="h5 holding the held-out val panels (RF training)")
    ap.add_argument("--val-csv", required=True, help="catalog for the val panels")
    ap.add_argument("--n-val-panels", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-root", default=str(REPO / "experiments/diffim_runs"))
    ap.add_argument("--models-dir", default=str(REPO / "models"))
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-rf", action="store_true")
    a = ap.parse_args()

    run_dir = Path(a.out_root) / a.run_name
    best = run_dir / "ckpts" / "best.pt"
    scripted = Path(a.models_dir) / f"{a.run_name}_v7_scripted.pt"

    # --- Stage 1: NN (reg2 recipe) ---
    if not a.skip_nn:
        cmd = [sys.executable, "-m", "ADCNN.training.train", "--run-name", a.run_name,
               "--out-root", a.out_root, "--epochs", str(a.epochs),
               "--lambda-orient", "0.0", "--dropout", "0.15", "--wd", "1e-4",
               "--intensity-aug", "--augment", "--ema-exclude", "agg_alpha",
               "--n-val-panels", str(a.n_val_panels)]
        if a.data_sources:
            cmd += ["--data-sources", *a.data_sources, "--data-h5", a.val_h5, "--data-csv", a.val_csv]
        else:
            cmd += ["--data-h5", a.data_h5, "--data-csv", a.data_csv]
        print("[stage1-nn]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        # export best checkpoint -> TorchScript
        subprocess.run([sys.executable, "-m", "ADCNN.inference.export",
                        "--ckpt", str(best), "--out", str(scripted), "--no-optimize"], check=True)
        print(f"[stage1-nn] scripted v7 -> {scripted}", flush=True)

    # --- Stage 2: RF (neg5, on held-out val candidates) ---
    if not a.skip_rf:
        from ADCNN.inference.rf_train import train_rf_from_val
        val_ids = sorted(pd.read_csv(a.val_csv)["image_id"].unique())[: a.n_val_panels]
        rf_out = Path(a.models_dir) / f"{a.run_name}_rf_postproc.pkl"
        train_rf_from_val(scripted, a.val_h5, a.val_csv, val_ids, rf_out, neg_ratio=a.neg_ratio)
        print(f"[stage2-rf] done -> {rf_out}", flush=True)
    print("END-TO-END TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
