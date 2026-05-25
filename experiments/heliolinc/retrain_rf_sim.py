"""Attack diffim false positives using ONLY simulated data (synthetic trails on real diffim
backgrounds) -- NO real-sky labels, to avoid the selection bias of the real (catalogued) objects.

Same v7, same RandomForest, same 72 features, same 0.5 threshold. The only change vs the deployed
neg5 RF: train on MANY more realistic-trail panels (the deployed RF saw only 64) so the RF sees
far more unbiased negatives (the noise + subtraction residuals in the real backgrounds).

Labels are purely simulated: positive = injected SYNTHETIC trail (a distribution we control, no
selection bias); negative = a background candidate that is NOT an injected trail and NOT on a real
DIA source (`frac_real_label_overlap < 0.5`). We deliberately do NOT label real DIA sources -- they
are a selection-biased sample of real objects (and include real asteroids), so using them would
both bias the model and teach it to reject real asteroids.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.inference.rf_postproc import RF_FEATURES_V2, save_rf
from ADCNN.inference.rf_train import infer_candidate_features, train_rf

REAL = REPO / "DATA_DIFFIM_realistic"
SHARDS = [(f"{REAL}/shard_0/train.h5", f"{REAL}/shard_0/train.csv"),
          (f"{REAL}/shard_1/train.h5", f"{REAL}/shard_1/train.csv"),
          (f"{REAL}/shard_2/train.h5", f"{REAL}/shard_2/train.csv"),
          (f"{REAL}/shard_3/train.h5", f"{REAL}/shard_3/train.csv")]


def pool(cand, labels):
    """X, y over the unbiased trainable candidates: injected-trail positives + negatives that are
    NOT on a real DIA source (real sources are excluded -> no selection bias from real objects)."""
    keep = (labels == 1) | ((labels == 0) & (cand["frac_real_label_overlap"].to_numpy() < 0.5))
    X = cand.loc[keep, list(RF_FEATURES_V2)].fillna(0.0).to_numpy(np.float32)
    return X, labels[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--panels-per-shard", type=int, default=300, help="how many panels per shard to use")
    ap.add_argument("--neg-ratio", type=int, default=12)
    ap.add_argument("--out", default=str(REPO / "models/rf_postproc_simhard.pkl"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(a.v7, map_location=dev).eval()
    Xs, ys = [], []
    for h5, csv in SHARDS:
        cat = pd.read_csv(csv)
        ids = sorted(cat.image_id.unique())[: a.panels_per_shard]
        remap = {o: i for i, o in enumerate(ids)}
        c = cat[cat.image_id.isin(ids)].copy(); c["image_id"] = c["image_id"].map(remap)
        cand, labels = infer_candidate_features(model, h5, ids, c, dev)
        X, y = pool(cand, labels)
        Xs.append(X); ys.append(y)
        print(f"[{Path(h5).parent.name}] {len(ids)} panels -> {len(y)} cand ({int(y.sum())} pos, {int((y==0).sum())} neg)", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"[pool] total {len(y)} candidates: {int(y.sum())} pos / {int((y==0).sum())} neg | neg_ratio={a.neg_ratio}", flush=True)
    rf = train_rf(X, y, neg_ratio=a.neg_ratio, seed=2026)
    save_rf(rf, a.out)
    s = rf.predict_proba(X)[:, 1]
    print(f"saved -> {a.out}")
    print(f"  train self-check @0.5: keeps {(s[y==1]>=.5).mean()*100:.0f}% of injected trails, "
          f"{(s[y==0]>=.5).mean()*100:.1f}% of negatives (incl. real artifacts)")
    print("DONE")


if __name__ == "__main__":
    main()
