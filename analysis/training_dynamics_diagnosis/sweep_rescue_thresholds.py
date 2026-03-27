from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADCNN.core.model import UNetResSEASPP
from ADCNN.training.rescue_validation import RescueValidator
from analysis.training_dynamics_diagnosis.rescue_subset_utils import (
    select_harder_val_panels,
    save_panel_subset,
)


def cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep rescue postprocess thresholds for a saved checkpoint.")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--train-h5", type=str, required=True)
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--subset-mode", choices=["missed_count", "missed_fraction"], default="missed_count")
    ap.add_argument("--thresholds", type=str, default="0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--budgets", type=str, default="50,200,1000,15000")
    ap.add_argument("--out-prefix", type=str, default="analysis/training_dynamics_diagnosis/threshold_sweep")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main() -> None:
    args = cli()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    panel_ids = select_harder_val_panels(
        train_h5=args.train_h5,
        train_csv=args.train_csv,
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        max_images=int(args.max_images),
        mode=str(args.subset_mode),
    )

    validator = RescueValidator(
        h5_path=args.train_h5,
        csv_path=args.train_csv,
        val_panel_ids=panel_ids,
        seed=int(args.seed),
        max_images=0,
        batch_size=16,
        num_workers=0,
        rescue_budget_primary=int(budgets[0]),
        rescue_budget_secondary=int(budgets[-1]),
        rescue_budget_grid=budgets,
        rescue_overlap_policy="ignore_baseline_duplicates",
        psf_width=40,
        threshold=0.2,
        pixel_gap=2,
        min_area=3,
        max_area=None,
        min_score=0.2,
        min_peak_probability=0.2,
        score_method="topk_mean",
        topk_fraction=0.05,
    )

    model = UNetResSEASPP(in_ch=1, out_ch=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state"], strict=True)
    preds = validator._predict_subset(model, device=device)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_panel_subset(
        str(out_prefix) + "_subset.json",
        panel_ids,
        meta={"subset_mode": args.subset_mode, "thresholds": thresholds, "budgets": budgets},
    )

    rows = []
    json_rows = []
    for thr in thresholds:
        result = validator.evaluate_predictions(preds, threshold=float(thr))
        json_rows.append(result)
        for rec in result["frontier"]:
            rows.append(
                {
                    "threshold": float(thr),
                    "budget": int(rec["budget"]),
                    "new_tp": int(rec["new_tp"]),
                    "missed_recall": float(rec["missed_recall"]),
                    "union_recall": float(rec["union_recall"]),
                    "added_fp": int(rec["added_fp"]),
                    "n_candidates": int(rec["n_candidates"]),
                }
            )

    with open(str(out_prefix) + ".json", "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)
    with open(str(out_prefix) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["threshold", "budget", "new_tp", "missed_recall", "union_recall", "added_fp", "n_candidates"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[threshold-sweep] wrote {out_prefix}.json and {out_prefix}.csv")


if __name__ == "__main__":
    main()
