"""Stage-2 FALSE-POSITIVE filter — focal-loss cutout CNN.

Stage 1 is the v7 NN (``predict`` + ``candidates``); stage 2 rejects false positives. For each
v7 candidate this scores a 48x48 3-channel cutout ``[diffim/sigma, v7_prob, v7_agg]`` centred on
the candidate with a small conv net, then keeps detections whose score >= ``CNN_DEFAULT_THR``.

Why a cutout CNN (and not the legacy 72-feature RandomForest): the CNN sees the raw local image
context the hand-built features summarised, so it separates faint trails from residual/artefact
false positives better. Trained on the dedicated train2 cutout dataset (focal loss, width-40,
30 epochs; see ``ADCNN.training.cnn_postproc``) -> shipped at ``models/cnn_postproc.pt``. On the
held-out test_5sigma it removes ~56% of false positives at 95% trail recall (AUC 0.93).

The score is written into the ``score`` column of the detection catalog (the operating cut is
applied by the caller, as with any reranker).
"""
from __future__ import annotations

import numpy as np

# Default operating point (override per call). Chosen on held-out test_5sigma to keep 95% of
# trails while roughly halving false positives; do NOT tune this on the eval set.
CNN_DEFAULT_THR = 0.63
CUTOUT_K = 48          # cutout side length (px); must match the training cutout size
NET_WIDTH = 40         # base conv width; must match the shipped checkpoint


def build_net(width: int = NET_WIDTH):
    """The stage-2 cutout classifier: 3 conv blocks (w, 2w, 4w) -> global-avg-pool -> linear.
    Shared by training (``ADCNN.training.cnn_postproc``) and inference so the architecture is
    defined in exactly one place."""
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(blk(3, width), blk(width, 2 * width), blk(2 * width, 4 * width),
                                nn.AdaptiveAvgPool2d(1))
            s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * width, 1))

        def forward(s, x):
            return s.h(s.f(x)).squeeze(1)

    return Net()


def load_cnn(path: str, device: str = "cpu"):
    """Load the focal cutout CNN (width-40 state_dict) in eval mode on `device`."""
    import torch
    net = build_net().to(device)
    net.load_state_dict(torch.load(str(path), map_location=device))
    net.eval()
    return net


def _cutout(arr: np.ndarray, x: float, y: float, k: int = CUTOUT_K) -> np.ndarray:
    """k x k patch of `arr` centred on (x, y), zero-padded where it runs off the panel edge."""
    h, w = arr.shape
    x, y = int(round(x)), int(round(y))
    hh = k // 2
    out = np.zeros((k, k), np.float32)
    x0, x1, y0, y1 = max(0, x - hh), min(w, x + hh), max(0, y - hh), min(h, y + hh)
    c = arr[y0:y1, x0:x1]
    out[:c.shape[0], :c.shape[1]] = c
    return out


def make_cutouts(cand_df, img, prob, agg, *, k: int = CUTOUT_K) -> np.ndarray:
    """Build the (N, 3, k, k) cutout stack [diffim/sigma, v7_prob, v7_agg] for each candidate.
    The diffim channel is normalised by the panel MAD-sigma and clipped to [-20, 20], exactly as
    in training — so a model trained by ``ADCNN.training.cnn_postproc`` scores them consistently."""
    img = np.asarray(img, np.float32)
    prob = np.asarray(prob, np.float32)
    agg = np.asarray(agg, np.float32)
    sig = float(np.median(np.abs(img - np.median(img))) * 1.4826) or 1.0
    if not len(cand_df):
        return np.zeros((0, 3, k, k), np.float32)
    X = np.stack([
        np.stack([_cutout(img, r.x_centroid, r.y_centroid, k) / sig,
                  _cutout(prob, r.x_centroid, r.y_centroid, k),
                  _cutout(agg, r.x_centroid, r.y_centroid, k)])
        for _, r in cand_df.iterrows()]).astype(np.float32)
    return np.clip(X, -20, 20)


def apply_cnn(cand_df, cnn, img, prob, agg, *, thr: float | None = None,
              score_col: str = "score", k: int = CUTOUT_K, device: str = "cpu"):
    """Score each candidate's cutout with the CNN and write `score_col`. Returns the dataframe
    with the score column added (and, when `thr` is given, rows below it removed)."""
    import torch
    out = cand_df.copy()
    if not len(out):
        out[score_col] = np.array([], np.float32)
        return out
    X = make_cutouts(out, img, prob, agg, k=k)
    with torch.no_grad():
        s = torch.sigmoid(torch.cat([
            cnn(torch.tensor(X[i:i + 512]).to(device)) for i in range(0, len(X), 512)])).cpu().numpy()
    out[score_col] = s
    if thr is not None:
        out = out[out[score_col] >= thr].reset_index(drop=True)
    return out
