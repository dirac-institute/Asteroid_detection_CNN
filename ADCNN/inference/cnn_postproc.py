"""Stage-2 FALSE-POSITIVE filter — focal-loss cutout CNN.

Stage 1 is the segmentation model (``predict`` + ``candidates``); stage 2 rejects false
positives. For each segmentation candidate this scores a 96x96 3-channel cutout
``[diffim/sigma, seg_prob, seg_agg]`` with a small conv net (``depth`` blocks, widths
w, 2w, ...) and keeps detections whose score >= the deployed threshold.

Why a cutout CNN (and not the legacy 72-feature RandomForest): the CNN sees the raw local
diffim context the hand-built features summarised, so it separates faint trails from
residual/artefact false positives better. Trained on the train2 cutout dataset (focal loss,
``ADCNN.training.cnn_postproc``).

Architecture is sidecar-driven: ``load_cnn(path)`` reads the matching ``.json`` next to the
``.pt`` for ``width / depth / k / aux_dim`` so a single inference module supports any trained
size without code edits. The deployed defaults below match the shipped checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# Default operating point (override per call via the sidecar's "threshold" entry). Do NOT tune
# this on the eval set — it's the combined 5sigma+ADCNN FP/panel-budget operating point set on
# val2 by ``ADCNN.training.cnn_postproc.calibrate_combined_threshold`` and persisted in the
# sidecar. The value below is the fallback when the sidecar is absent.
CNN_DEFAULT_THR = 0.6
CUTOUT_K = 96          # cutout side length (px); shipped CNN cache + training defaults to 96.
NET_WIDTH = 40         # base conv width; matches the deployed checkpoint.
NET_DEPTH = 4          # number of conv blocks (depth-4 was the FP-budget breakthrough).


def build_net(width: int = NET_WIDTH, depth: int = NET_DEPTH, in_ch: int = 3, k: int = CUTOUT_K,
              aux_dim: int = 0):
    """The stage-2 cutout classifier: ``depth`` conv blocks (widths ``w, 2w, 4w, ...``) ->
    AdaptiveAvgPool2d(1) -> Dropout -> Linear. Fully convolutional w.r.t. ``k``, so larger
    cutouts work with no code change. If ``aux_dim > 0`` a small MLP on the candidate catalog
    features is concatenated before the head (kept for compatibility; deployed shape is
    aux_dim=0). The architecture is defined here ONCE; training imports it from this module.
    """
    import torch
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
    aux_head = None
    head_in = c
    if aux_dim > 0:
        aux_head = nn.Sequential(
            nn.Linear(aux_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32))
        head_in = c + 32
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(head_in, 1))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.backbone = backbone
            s.aux_head = aux_head
            s.head = head

        def forward(s, x, aux=None):
            z = s.backbone(x)
            if s.aux_head is not None:
                if aux is None:
                    raise ValueError("model was built with aux_dim>0 but aux was not provided")
                z = torch.cat([z, s.aux_head(aux)], dim=1)
            return s.head(z).squeeze(1)
    return Net()


def load_cnn(path: str, device: str = "cpu"):
    """Load the focal cutout CNN. The architecture (width / depth / k / aux_dim) is read from
    the sidecar JSON next to ``path``; missing keys fall back to the module defaults so the
    legacy depth=3 / k=48 checkpoint still loads."""
    import torch
    p = Path(path); sc = p.with_suffix(".json")
    info = json.loads(sc.read_text()) if sc.exists() else {}
    net = build_net(width=int(info.get("width", NET_WIDTH)),
                    depth=int(info.get("depth", NET_DEPTH)),
                    in_ch=int(info.get("in_ch", 3)),
                    k=int(info.get("k", CUTOUT_K)),
                    aux_dim=int(info.get("aux_dim", 0))).to(device)
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
    out[:c.shape[0], :c.shape[1]] = c
    return out


def make_cutouts(cand_df, img, prob, agg, *, k: int = CUTOUT_K) -> np.ndarray:
    """Build the (N, 3, k, k) cutout stack [diffim/sigma, seg_prob, seg_agg] for each candidate.
    The diffim channel is normalised by the panel MAD-sigma and clipped to [-20, 20] (matches
    training in ``ADCNN.training.cnn_postproc``)."""
    img = np.asarray(img, np.float32)
    prob = np.asarray(prob, np.float32)
    agg = np.asarray(agg, np.float32)
    # MAD-sigma over FINITE pixels only: real DP2 diffims carry NaN/masked pixels.
    finite = img[np.isfinite(img)]
    sig = float(np.median(np.abs(finite - np.median(finite))) * 1.4826) if finite.size else 1.0
    sig = sig or 1.0
    if not len(cand_df):
        return np.zeros((0, 3, k, k), np.float32)
    X = np.stack([
        np.stack([_cutout(img, r.x_centroid, r.y_centroid, k) / sig,
                  _cutout(prob, r.x_centroid, r.y_centroid, k),
                  _cutout(agg, r.x_centroid, r.y_centroid, k)])
        for _, r in cand_df.iterrows()]).astype(np.float32)
    # scrub NaN/inf to finite before the model sees them; no-op on clean sim data.
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)


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
