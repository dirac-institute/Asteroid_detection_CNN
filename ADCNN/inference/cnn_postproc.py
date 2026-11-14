"""Stage-2 false-positive filter — focal-loss cutout CNN.

For each stage-1 segmentation candidate this scores a ``CUTOUT_K x CUTOUT_K`` 3-channel cutout
``[diffim/sigma, seg_prob, seg_agg]`` with a small conv net (``depth`` blocks, widths
``w, 2w, 4w, ...``); ADCNN.inference.catalog keeps the detections whose score meets the threshold.

The architecture is sidecar-driven: ``load_cnn(path)`` reads the matching ``.json`` next
to the ``.pt`` for ``width / depth / in_ch`` so a single inference module supports any
trained size. Defaults below match the shipped checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CUTOUT_K = 96          # cutout side length (px); matches the checkpoint sidecar's k.
NET_WIDTH = 40         # base conv width; matches the deployed checkpoint.
NET_DEPTH = 4          # number of conv blocks.
SCORE_BATCH = 512      # forward micro-batch for `apply_cnn`.
CLIP_SIGMA = 20.0      # diffim cutout clip range (in units of MAD-sigma); matches training.


def build_net(width: int = NET_WIDTH, depth: int = NET_DEPTH, in_ch: int = 3):
    """The stage-2 cutout classifier: ``depth`` conv blocks (widths ``w, 2w, 4w, ...``) ->
    AdaptiveAvgPool2d(1) -> Dropout -> Linear. Fully convolutional, so the cutout size is not
    a parameter.
    """
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    layers = []
    c = in_ch
    for i in range(depth):
        w = width * (2 ** i)
        layers.append(blk(c, w))
        c = w
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    backbone = nn.Sequential(*layers)
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, 1))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.backbone = backbone
            s.head = head

        def forward(s, x):
            return s.head(s.backbone(x)).squeeze(1)
    return Net()


def load_cnn(path: str, device: str):
    """Load the focal cutout CNN. The architecture (``width / depth / in_ch``) is read from the
    sidecar JSON next to ``path``; missing keys fall back to the module defaults."""
    import torch
    p = Path(path); sc = p.with_suffix(".json")
    info = json.loads(sc.read_text()) if sc.exists() else {}
    net = build_net(width=int(info.get("width", NET_WIDTH)),
                    depth=int(info.get("depth", NET_DEPTH)),
                    in_ch=int(info.get("in_ch", 3))).to(device)
    net.load_state_dict(torch.load(str(p), map_location=device, weights_only=True))
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
    # Place the patch at its OFFSET inside the stamp. Writing it at [0,0] would shift every source
    # clipped by the LEFT or TOP edge off-centre, and `_features` analyses stamp[c-28:c+29]
    # assuming it is centred (right/bottom clipping keeps the centre either way).
    oy, ox = y0 - (y - hh), x0 - (x - hh)
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def make_cutouts(cand_df, img, prob, agg) -> np.ndarray:
    """Build the (N, 3, CUTOUT_K, CUTOUT_K) cutout stack [diffim/sigma, seg_prob, seg_agg] for each
    candidate. The diffim channel is normalised by the panel MAD-sigma and clipped to
    ``[-CLIP_SIGMA, +CLIP_SIGMA]`` (matches training)."""
    img = np.asarray(img, np.float32)
    prob = np.asarray(prob, np.float32)
    agg = np.asarray(agg, np.float32)
    # The canonical panel MAD-sigma (preprocessing.diffim_mad_sigma, nonzero pixels), computed once
    # during candidate extraction: median(|x - median(x)|) here instead returned exactly 0 on any
    # panel >=50% masked and saturated the diffim channel.
    sig = float(cand_df["panel_sigma"].iloc[0])
    X = np.stack([
        np.stack([_cutout(img, r.x_centroid, r.y_centroid) / sig,
                  _cutout(prob, r.x_centroid, r.y_centroid),
                  _cutout(agg, r.x_centroid, r.y_centroid)])
        for _, r in cand_df.iterrows()]).astype(np.float32)
    # Scrub NaN/inf to finite before the model sees them.
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=CLIP_SIGMA, neginf=-CLIP_SIGMA),
                   -CLIP_SIGMA, CLIP_SIGMA)


def apply_cnn(cand_df, cnn, img, prob, agg, *, device):
    """Score each candidate's cutout with the CNN. Returns a copy of `cand_df` with a `score`
    column."""
    import torch
    out = cand_df.copy()
    X = make_cutouts(out, img, prob, agg)
    with torch.no_grad():
        s = torch.sigmoid(torch.cat([
            cnn(torch.tensor(X[i:i + SCORE_BATCH]).to(device))
            for i in range(0, len(X), SCORE_BATCH)])).cpu().numpy()
    out["score"] = s
    return out
