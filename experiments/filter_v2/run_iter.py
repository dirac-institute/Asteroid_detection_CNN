"""Run one filter-v2 iteration end-to-end and append a row to results.tsv.

Wraps train_filter.py + eval_combined.py for a single config. The config is just CLI args
forwarded into the train step. Eval writes its metrics next to the .pt; the tsv carries the
condensed summary so progress is resumable + glanceable across iterations.

Usage:
    python -m experiments.filter_v2.run_iter --iter iter01_baseline -- \
        --width 40 --depth 3 --epochs 30
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CUTS = REPO / "experiments/filter_v2/cutouts"
RUNS = REPO / "experiments/filter_v2/runs"
TSV  = REPO / "experiments/filter_v2/results.tsv"

HDR = ("iter\tckpt\twidth\tdepth\tk\tin_ch\tepochs\thnm_from\tval_auc\t"
       "T_min\tcombined_recall\tcombined_fp_per_panel\tstack5_fp_per_panel\tmeets_budget\twall_s\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", required=True, help="iteration label (becomes runs/<iter>/cnn.pt)")
    ap.add_argument("--recall-target", type=float, default=0.81)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="args forwarded to train_filter.py after a '--' separator")
    a = ap.parse_args()

    if a.train_args and a.train_args[0] == "--":
        a.train_args = a.train_args[1:]

    run = RUNS / a.iter; run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "cnn.pt"
    metrics = run / "metrics.json"
    t0 = time.time()

    print(f"=== {a.iter} ===  ckpt={ckpt}", flush=True)
    train_cmd = [sys.executable, "-u", "-m", "experiments.filter_v2.train_filter",
                 "--train-cuts", str(CUTS / "train2"),
                 "--val-cuts", str(CUTS / "val"),
                 "--out", str(ckpt), "--device", a.device, *a.train_args]
    print("[run] train:", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True, cwd=str(REPO))

    eval_cmd = [sys.executable, "-u", "-m", "experiments.filter_v2.eval_combined",
                "--ckpt", str(ckpt), "--out", str(metrics),
                "--recall-target", str(a.recall_target), "--device", a.device]
    print("[run] eval:", " ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, check=True, cwd=str(REPO))

    info = json.loads(ckpt.with_suffix(".json").read_text())
    met = json.loads(metrics.read_text())
    if not TSV.exists():
        TSV.write_text(HDR)
    row = [a.iter, str(ckpt), info["width"], info["depth"], info["k"], info["in_ch"],
           info["epochs"],
           next((x for x in a.train_args if x.endswith(".pt")), ""),
           f"{info.get('best_val_auc', float('nan')):.4f}",
           f"{met['T_min']:.4f}", f"{met['combined_recall']:.4f}",
           f"{met['combined_fp_per_panel']:.2f}", f"{met['stack5_fp_per_panel']:.2f}",
           str(met["meets_budget"]).lower(), f"{time.time()-t0:.0f}"]
    with TSV.open("a") as f:
        f.write("\t".join(map(str, row)) + "\n")
    print(f"\nDONE in {time.time()-t0:.0f}s -> row appended to {TSV}", flush=True)


if __name__ == "__main__":
    main()
