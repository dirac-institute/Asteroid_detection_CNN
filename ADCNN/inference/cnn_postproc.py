"""Post-v7 stage-2 FALSE-POSITIVE filter: focal-loss cutout CNN (drop-in alternative to the RF in
``rf_postproc``). For each v7 candidate it scores a 48x48 3-channel cutout [diffim/sigma, v7_prob,
v7_agg] centred on the candidate, then keeps detections with score >= threshold.

Trained on the dedicated 50 GB train2 dataset (focal loss, width-40, 30 epochs) ->
``experiments/heliolinc/rejecter_data/cnn_focal_final.pt``. On held-out test_5sigma it removes ~56% of
FP at 95% recall (AUC 0.932); deployed here at the OPERATING POINT that matches the old RF's FP-per-panel
(thr 0.63 -> 72.5 FP/panel, the same density as RF@0.5, but recall 0.754 vs the RF's 0.736).

The CNN score is written into the same ``score_rf`` column as the RF path, so the downstream catalog
schema / HelioLinC inputs are unchanged.
"""
from __future__ import annotations
import numpy as np

# threshold matching the old RF (rf_postproc DEFAULT_THR=0.5) at equal FP/panel on held-out test_5sigma
# (RF: 72.5 FP/panel, recall 0.736 -> CNN@0.63: 72.5 FP/panel, recall 0.754). Override per call if needed.
CNN_DEFAULT_THR = 0.63
CUTOUT_K = 48
NET_WIDTH = 40


def _build_net(width: int = NET_WIDTH):
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
    """Load the focal cutout CNN (width-40 state_dict) in eval mode on `device` (CPU default; the
    catalog feature-workers hide the GPU)."""
    import torch
    net = _build_net().to(device)
    net.load_state_dict(torch.load(str(path), map_location=device))
    net.eval()
    return net


def _cutout(arr: np.ndarray, x: float, y: float, k: int = CUTOUT_K) -> np.ndarray:
    h, w = arr.shape
    x, y = int(round(x)), int(round(y))
    hh = k // 2
    out = np.zeros((k, k), np.float32)
    x0, x1, y0, y1 = max(0, x - hh), min(w, x + hh), max(0, y - hh), min(h, y + hh)
    c = arr[y0:y1, x0:x1]
    out[:c.shape[0], :c.shape[1]] = c
    return out


def apply_cnn(cand_df, cnn, img, prob, agg, *, thr: float | None = None,
              score_col: str = "score_rf", k: int = CUTOUT_K, device: str = "cpu"):
    """Score each candidate's [diffim/sigma, v7_prob, v7_agg] cutout with the CNN and write `score_col`.
    Returns the dataframe with the score column added (caller applies the >= thr cut, as with the RF)."""
    import torch
    if not len(cand_df):
        cand_df[score_col] = np.array([], np.float32)
        return cand_df
    img = np.asarray(img, np.float32); prob = np.asarray(prob, np.float32); agg = np.asarray(agg, np.float32)
    sig = float(np.median(np.abs(img - np.median(img))) * 1.4826) or 1.0
    X = np.stack([
        np.stack([_cutout(img, r.x_centroid, r.y_centroid, k) / sig,
                  _cutout(prob, r.x_centroid, r.y_centroid, k),
                  _cutout(agg, r.x_centroid, r.y_centroid, k)])
        for _, r in cand_df.iterrows()]).astype(np.float32)
    X = np.clip(X, -20, 20)
    with torch.no_grad():
        s = torch.sigmoid(torch.cat([
            cnn(torch.tensor(X[i:i + 512]).to(device)) for i in range(0, len(X), 512)])).cpu().numpy()
    cand_df = cand_df.copy()
    cand_df[score_col] = s
    return cand_df
