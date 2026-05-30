"""Evaluate a trained filter v2 model with the COMBINED 5σ+NN FP/panel metric on val2.

Pipeline:
  1. Load the trained .pt (architecture comes from the matching sidecar .json).
  2. Score every cached val2 cutout -> the val2 ADCNN catalog (image_id, x, y, beta, length, score).
  3. Build the 5σ stack catalog on val2 (uses DATA/val2.h5 + DATA/val2.csv real_labels_5sigma plane).
  4. For each candidate ADCNN threshold, evaluate the cross-dedup'd union against val2 truth.
  5. Report:
       - threshold T_min where combined recall ≥ --recall-target (default 0.81),
       - combined FP/panel at T_min,
       - the 5σ-stack-alone baseline (recall + FP),
       - the recall ADCNN adds.

Goal (overnight mandate): combined recall ≥ 0.81 at FP/panel ≤ 1.5 × 5σ-stack ≈ 75 on val2.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CAT_COLS = ["x_centroid", "y_centroid", "mf_beta", "mf_length",
            "mf_flux", "mf_snr", "area", "max_p"]
MF_LEN_OFFSET = 33.4
MF_LEN_SLOPE = 0.887


def score_cuts(net, X, batch_size, device, aux=None):
    import torch
    n = len(X); s = np.zeros(n, np.float32)
    net.eval()
    Xt = torch.tensor(np.clip(X, -20, 20).astype(np.float32))
    At = torch.tensor(aux) if aux is not None else None
    with torch.no_grad():
        for k in range(0, n, batch_size):
            chunk = Xt[k:k + batch_size].to(device)
            if At is not None:
                s[k:k + batch_size] = torch.sigmoid(net(chunk, At[k:k + batch_size].to(device))).cpu().numpy()
            else:
                s[k:k + batch_size] = torch.sigmoid(net(chunk)).cpu().numpy()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".pt produced by train_filter.py")
    ap.add_argument("--val2-cuts", default=str(REPO / "experiments/filter_v2/cutouts/val2"))
    ap.add_argument("--val2-h5", default=str(REPO / "DATA/val2.h5"))
    ap.add_argument("--val2-csv", default=str(REPO / "DATA/val2.csv"))
    ap.add_argument("--recall-target", type=float, default=0.81,
                    help="find the threshold at which combined recall meets this target")
    ap.add_argument("--out", default=None, help="optional path to write metrics json")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from experiments.filter_v2.train_filter import build_net
    from ADCNN.evaluation.catalog_match import (evaluate_catalog, dedup_cross_catalog,
                                                  stack_sigma_catalog)
    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    info = json.loads(Path(a.ckpt).with_suffix(".json").read_text())
    aux_dim = int(info.get("aux_dim", 0))
    net = build_net(info["width"], info["depth"], in_ch=info["in_ch"], k=info["k"], aux_dim=aux_dim).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=True))
    print(f"[eval] loaded {a.ckpt}  arch w{info['width']}d{info['depth']} k{info['k']}  "
          f"best_val_auc={info.get('best_val_auc', float('nan')):.4f}", flush=True)

    # ---- score val2 cutouts ----
    parts = sorted(Path(a.val2_cuts).glob("part_*.npz"))
    Xs, ys, ps, cs = [], [], [], []
    for p in parts:
        z = np.load(p)
        Xs.append(z["X"]); ys.append(z["y"]); ps.append(z["panel"]); cs.append(z["cand"])
    X = np.concatenate(Xs); y = np.concatenate(ys); panel = np.concatenate(ps); cand = np.concatenate(cs)
    cache_k = X.shape[2]
    model_k = info["k"]
    if model_k != cache_k:
        if model_k > cache_k:
            raise SystemExit(f"model k={model_k} > cached k={cache_k}; rebuild cutouts at larger k")
        o = (cache_k - model_k) // 2
        X = X[:, :, o:o + model_k, o:o + model_k]
    aux = None
    if aux_dim > 0:
        idx = info["aux_idx"]; mean = np.array(info["aux_mean"], np.float32); std = np.array(info["aux_std"], np.float32)
        raw = cand[:, idx].astype(np.float32)
        aux = np.clip(((np.nan_to_num(raw, nan=0.0) - mean) / std), -10, 10).astype(np.float32)
        print(f"[eval] aux features: dim={aux_dim}  idx={idx}", flush=True)
    print(f"[eval] val2 cutouts: {len(y)} candidates over {len(np.unique(panel))} panels  "
          f"(cache_k={cache_k}, model_k={model_k})", flush=True)
    scores = score_cuts(net, X, a.batch_size, dev, aux=aux)

    # ---- val2 ADCNN catalog (with the inference-pipeline mf_length de-bias so segment matching is consistent) ----
    raw_len = cand[:, CAT_COLS.index("mf_length")]
    length = np.clip((raw_len - MF_LEN_OFFSET) / MF_LEN_SLOPE, 0.0, None)
    adcnn = pd.DataFrame({
        "image_id": panel.astype(int),
        "x":  cand[:, 0], "y":  cand[:, 1],
        "beta": cand[:, 2], "length": length,
        "score": scores,
    })

    # ---- 5σ stack catalog on val2 ----  (panel set = truth's panels, not just panels with candidates)
    truth_ids = sorted(pd.read_csv(a.val2_csv)["image_id"].unique())
    n_panels = len(truth_ids)
    stack = stack_sigma_catalog(a.val2_h5, a.val2_csv, truth_ids, sigma=5)
    m_stack, _ = evaluate_catalog(stack, a.val2_csv, tol_px=20.0, n_panels=n_panels)
    print(f"[eval] stack 5σ: recall={m_stack['recall']:.4f}  FP/panel={m_stack['fp_per_panel']:.2f}", flush=True)

    # ---- sweep thresholds + find T_min for the recall target ----
    cols = ["image_id", "x", "y", "beta", "length"]
    def union_metrics(T):
        a_kept = adcnn[adcnn["score"] >= T][cols]
        u = dedup_cross_catalog(stack[cols], a_kept, tol_px=20.0)
        m, _ = evaluate_catalog(u, a.val2_csv, tol_px=20.0, n_panels=n_panels)
        return m

    grid = sorted(set(np.round(np.concatenate([
        np.linspace(0.05, 0.95, 19),
        np.array([0.0, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]),
    ]), 3)))
    rows = []
    for T in grid:
        m = union_metrics(T)
        rows.append((T, m["recall"], m["fp_per_panel"], m["TP"], m["FP"], m["FN"]))
    cur = pd.DataFrame(rows, columns=["T", "recall", "fp_per_panel", "TP", "FP", "FN"])
    print("[eval] curve (T, recall, FP/panel):", flush=True)
    print(cur.to_string(index=False), flush=True)

    # binary search T_min so recall >= target
    lo, hi = 0.0, 1.0
    if union_metrics(lo)["recall"] < a.recall_target:
        T_min = lo; m_op = union_metrics(lo); ok = False
    elif union_metrics(hi)["recall"] >= a.recall_target:
        T_min = hi; m_op = union_metrics(hi); ok = True
    else:
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if union_metrics(mid)["recall"] >= a.recall_target:
                lo = mid                 # can still raise T
            else:
                hi = mid
        T_min = lo; m_op = union_metrics(lo); ok = True

    target_fp = 1.5 * m_stack["fp_per_panel"]
    print(f"\n[eval] OP @ recall≥{a.recall_target:.2f}: T={T_min:.4f}  "
          f"combined recall={m_op['recall']:.4f}  FP/panel={m_op['fp_per_panel']:.2f}  "
          f"(budget=1.5x stack={target_fp:.2f})  meets_budget={m_op['fp_per_panel'] <= target_fp}", flush=True)
    print(f"[eval] ADCNN adds {m_op['recall']-m_stack['recall']:+.4f} recall vs stack alone", flush=True)

    out = {
        "ckpt": a.ckpt,
        "recall_target": a.recall_target,
        "T_min": float(T_min),
        "combined_recall": float(m_op["recall"]),
        "combined_fp_per_panel": float(m_op["fp_per_panel"]),
        "combined_tp": int(m_op["TP"]), "combined_fp": int(m_op["FP"]), "combined_fn": int(m_op["FN"]),
        "stack5_recall": float(m_stack["recall"]),
        "stack5_fp_per_panel": float(m_stack["fp_per_panel"]),
        "fp_budget": float(target_fp),
        "meets_budget": bool(m_op["fp_per_panel"] <= target_fp),
        "curve": cur.to_dict(orient="records"),
        "n_panels": int(n_panels),
        "n_candidates": int(len(y)),
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"[eval] metrics -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
