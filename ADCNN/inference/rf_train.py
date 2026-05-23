"""Train the RandomForest second-stage post-processor (the reg2 "neg5" RF).

Pipeline stage 2: the v7 segmentation emits many candidate components per panel
(asteroid trails + residual/artefact false positives). The RandomForest scores each
candidate from 72 hand-built features (matched-filter SNR/length, morphology, local
PCA, orientation agreement, ...) and rejects false positives while keeping trails.

Training is leakage-safe: features are computed by running the trained v7 on the
held-out VALIDATION panels (never the test set, never the v7's own training panels),
candidates are labelled by overlap with the injected truth, and negatives are
subsampled `neg_ratio:1` (reg2 used 5:1 — the "neg5" RF).

Entry: ``train_rf_from_val(v7_ckpt, val_h5, val_csv, val_panel_ids, out_pkl)``.
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    RF_FEATURES_V2, compute_v2_features, label_candidates_by_injection_overlap, save_rf,
)


def infer_candidate_features(model, h5_path, panel_ids, catalog, device):
    """Run v7 over `panel_ids` of `h5_path`, extract candidates + 72 features, and
    label each candidate (1=overlaps an injected trail, 0=false positive)."""
    probs, sins, coss, aggs, diffims, reals = [], [], [], [], [], []
    with h5py.File(h5_path, "r") as f:
        for pid in panel_ids:
            img = f["images"][pid][:]
            rl = f["real_labels"][pid][:].astype(np.uint16)
            p, s, c, a = predict_panel_overlap_3ch_full(model, img, rl, device=device)
            probs.append(p.astype(np.float32)); sins.append(s); coss.append(c); aggs.append(a)
            diffims.append(img.astype(np.float32)); reals.append(rl)
    stk = lambda L: np.stack(L, 0)
    prob = stk(probs)
    cand, _ = compute_v2_features(prob, stk(diffims), stk(sins), stk(coss), stk(aggs),
                                  real_labels=stk(reals), verbose=False)
    labels = label_candidates_by_injection_overlap(cand, catalog, prob)
    return cand, labels


def build_pool(cand, labels):
    """Select the trainable candidate pool: all positives, plus negatives that do NOT
    sit on a DIA/artefact mask (frac_real_label_overlap < 0.5). Returns (X, y, groups)."""
    fp = (labels == 0) & (cand["frac_real_label_overlap"].to_numpy() < 0.5)
    keep = (labels == 1) | fp
    X = cand.loc[keep, list(RF_FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
    return X, labels[keep], cand.loc[keep, "panel_id"].to_numpy()


def train_rf(X, y, *, neg_ratio: int = 5, seed: int = 0) -> RandomForestClassifier:
    """Fit the reg2 RandomForest: 500 trees, depth 14, balanced, negatives subsampled
    `neg_ratio:1` against positives (neg_ratio=5 = the deployed neg5 RF)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, min(len(neg), neg_ratio * len(pos)), replace=False)])
    return RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed,
    ).fit(X[keep], y[keep])


def train_rf_from_val(v7_ckpt, val_h5, val_csv, val_panel_ids, out_pkl, *,
                      neg_ratio: int = 5, device: str = "cuda") -> RandomForestClassifier:
    """Full RF training: load the TorchScript v7, compute candidate features on the
    held-out val panels, build the pool, fit the neg-subsampled RF, and save it."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(v7_ckpt), map_location=dev).eval()
    cat = pd.read_csv(val_csv)
    cand, labels = infer_candidate_features(model, val_h5, val_panel_ids, cat, dev)
    X, y, _ = build_pool(cand, labels)
    print(f"[rf-train] pool: {len(y)} candidates ({int(y.sum())} pos); neg_ratio={neg_ratio}", flush=True)
    rf = train_rf(X, y, neg_ratio=neg_ratio)
    save_rf(rf, str(out_pkl))
    print(f"[rf-train] saved -> {out_pkl}", flush=True)
    return rf
