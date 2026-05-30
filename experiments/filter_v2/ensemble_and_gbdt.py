"""Quick-win experiments WITHOUT retraining any CNN.

(a) Mean-ensemble of multiple iter*/cnn.pt scores on val2.
(b) Post-CNN GBDT: train an XGBoost / LightGBM-style classifier on
    (cnn_score_from_best_iter, mf_snr, mf_length, area, elongation, max_p)
    using train2 cutouts (CNN scores + cached cand columns) and evaluate on val2.

Both write a "scores" array sized to val2 cutouts; the combined-FP metric is computed via
the same path as eval_combined.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CAT_COLS = ["x_centroid", "y_centroid", "mf_beta", "mf_length",
            "mf_flux", "mf_snr", "area", "max_p"]
MF_LEN_OFFSET = 33.4
MF_LEN_SLOPE = 0.887


def load_set(cuts_dir):
    parts = sorted(Path(cuts_dir).glob("part_*.npz"))
    Xs, ys, ps, cs = [], [], [], []
    for p in parts:
        z = np.load(p)
        Xs.append(z["X"]); ys.append(z["y"]); ps.append(z["panel"]); cs.append(z["cand"])
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(ps), np.concatenate(cs))


def score_with_ckpt(ckpt: Path, X, cand, device):
    """Score `X` (and aux from `cand` if the ckpt's sidecar says aux_dim>0) using one CNN."""
    import torch
    from experiments.filter_v2.train_filter import build_net, DEFAULT_AUX_IDX
    info = json.loads(ckpt.with_suffix(".json").read_text())
    aux_dim = int(info.get("aux_dim", 0))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    net = build_net(info["width"], info["depth"], in_ch=info["in_ch"],
                    k=info["k"], aux_dim=aux_dim).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    net.eval()
    cache_k = X.shape[2]; model_k = info["k"]
    if model_k != cache_k:
        o = (cache_k - model_k) // 2
        X = X[:, :, o:o + model_k, o:o + model_k]
    aux = None
    if aux_dim > 0:
        idx = info["aux_idx"]; m = np.array(info["aux_mean"], np.float32); s = np.array(info["aux_std"], np.float32)
        raw = cand[:, idx].astype(np.float32)
        aux = np.clip(((np.nan_to_num(raw, nan=0.0) - m) / s), -10, 10).astype(np.float32)
    Xt = torch.tensor(np.clip(X, -20, 20).astype(np.float32))
    At = torch.tensor(aux) if aux is not None else None
    out = np.zeros(len(X), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            chunk = Xt[i:i + 1024].to(dev)
            if At is not None:
                out[i:i + 1024] = torch.sigmoid(net(chunk, At[i:i + 1024].to(dev))).cpu().numpy()
            else:
                out[i:i + 1024] = torch.sigmoid(net(chunk)).cpu().numpy()
    return out


def eval_scores(scores, panel, cand, val2_csv, val2_h5, recall_target):
    """Build the val2 ADCNN catalog with these scores; run the combined-FP metric."""
    from ADCNN.evaluation.catalog_match import (evaluate_catalog, dedup_cross_catalog,
                                                  stack_sigma_catalog)
    raw_len = cand[:, CAT_COLS.index("mf_length")]
    length = np.clip((raw_len - MF_LEN_OFFSET) / MF_LEN_SLOPE, 0.0, None)
    adcnn = pd.DataFrame({"image_id": panel.astype(int),
                          "x": cand[:, 0], "y": cand[:, 1],
                          "beta": cand[:, 2], "length": length,
                          "score": scores})
    truth_ids = sorted(pd.read_csv(val2_csv)["image_id"].unique())
    n_panels = len(truth_ids)
    stack = stack_sigma_catalog(val2_h5, val2_csv, truth_ids, sigma=5)
    m_stack, _ = evaluate_catalog(stack, val2_csv, tol_px=20.0, n_panels=n_panels)
    cols = ["image_id", "x", "y", "beta", "length"]

    def at(T):
        a = adcnn[adcnn["score"] >= T][cols]
        u = dedup_cross_catalog(stack[cols], a, tol_px=20.0)
        m, _ = evaluate_catalog(u, val2_csv, tol_px=20.0, n_panels=n_panels)
        return m

    # binary search recall_target
    lo, hi = 0.0, 1.0
    if at(lo)["recall"] < recall_target:
        T = lo; m = at(lo)
    elif at(hi)["recall"] >= recall_target:
        T = hi; m = at(hi)
    else:
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if at(mid)["recall"] >= recall_target:
                lo = mid
            else:
                hi = mid
        T = lo; m = at(lo)
    budget = 1.5 * m_stack["fp_per_panel"]
    return {"T_min": float(T), "combined_recall": float(m["recall"]),
            "combined_fp_per_panel": float(m["fp_per_panel"]),
            "stack5_recall": float(m_stack["recall"]),
            "stack5_fp_per_panel": float(m_stack["fp_per_panel"]),
            "fp_budget": float(budget),
            "meets_budget": bool(m["fp_per_panel"] <= budget)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="model .pt files to ensemble OR feed the GBDT as features")
    ap.add_argument("--mode", choices=["ensemble", "gbdt"], required=True)
    ap.add_argument("--train2-cuts", default=str(REPO / "experiments/filter_v2/cutouts/train2"))
    ap.add_argument("--val2-cuts", default=str(REPO / "experiments/filter_v2/cutouts/val2"))
    ap.add_argument("--val2-csv", default=str(REPO / "DATA/val2.csv"))
    ap.add_argument("--val2-h5", default=str(REPO / "DATA/val2.h5"))
    ap.add_argument("--recall-target", type=float, default=0.81)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # Always load val2
    Xv, yv, pv, cv = load_set(a.val2_cuts)
    print(f"val2: N={len(yv)}  TP={int((yv==1).sum())}", flush=True)

    if a.mode == "ensemble":
        scores = np.zeros(len(yv), np.float32)
        for c in a.ckpts:
            s = score_with_ckpt(Path(c), Xv, cv, a.device)
            print(f"  {Path(c).parent.name}: mean={s.mean():.3f}", flush=True)
            scores += s
        scores /= len(a.ckpts)
        m = eval_scores(scores, pv, cv, a.val2_csv, a.val2_h5, a.recall_target)
        print("\n== ENSEMBLE ==")
        print(json.dumps(m, indent=2))

    elif a.mode == "gbdt":
        # Use the first (best) ckpt for the CNN-score feature; stack with catalog features.
        primary = Path(a.ckpts[0])
        # Score train2 with the primary model (cache size is large -- streamed by score_with_ckpt).
        Xtr, ytr, _, ctr = load_set(a.train2_cuts)
        print(f"train2: N={len(ytr)}  TP={int((ytr==1).sum())}", flush=True)
        s_tr = score_with_ckpt(primary, Xtr, ctr, a.device)
        s_va = score_with_ckpt(primary, Xv, cv, a.device)
        # Features: [cnn_score, mf_length, mf_flux, mf_snr, area, max_p]
        # log-scale the heavy-tailed ones.
        def feats(s, cand):
            ml = cand[:, CAT_COLS.index("mf_length")]
            mf = cand[:, CAT_COLS.index("mf_flux")]
            ms = cand[:, CAT_COLS.index("mf_snr")]
            ar = cand[:, CAT_COLS.index("area")]
            mp = cand[:, CAT_COLS.index("max_p")]
            return np.stack([s,
                             np.log1p(np.clip(ml, 0, None)),
                             np.sign(mf) * np.log1p(np.abs(mf)),
                             np.sign(ms) * np.log1p(np.abs(ms)),
                             np.log1p(np.clip(ar, 0, None)),
                             mp], axis=1).astype(np.float32)
        Ftr = feats(s_tr, ctr); Fva = feats(s_va, cv)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                  max_depth=6, class_weight="balanced",
                                                  random_state=7)
            print("[gbdt] fitting HistGradientBoosting", flush=True)
            clf.fit(Ftr, ytr.astype(int))
            scores = clf.predict_proba(Fva)[:, 1].astype(np.float32)
        except Exception as e:
            print("sklearn HGB failed, fallback to LogReg:", e, flush=True)
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)
            clf.fit(Ftr, ytr.astype(int))
            scores = clf.predict_proba(Fva)[:, 1].astype(np.float32)
        m = eval_scores(scores, pv, cv, a.val2_csv, a.val2_h5, a.recall_target)
        print("\n== POST-CNN GBDT ==")
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
