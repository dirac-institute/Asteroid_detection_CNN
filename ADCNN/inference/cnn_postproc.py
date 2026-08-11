"""Stage-2 false-positive filter — focal-loss cutout CNN.

For each stage-1 segmentation candidate this scores a ``k x k`` 3-channel cutout
``[diffim/sigma, seg_prob, seg_agg]`` with a small conv net (``depth`` blocks, widths
``w, 2w, 4w, ...``) and keeps detections whose score meets the deployed threshold.

The architecture is sidecar-driven: ``load_cnn(path)`` reads the matching ``.json`` next
to the ``.pt`` for ``width / depth / k`` so a single inference module supports any
trained size. Defaults below match the shipped checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ADCNN.data.preprocessing import diffim_mad_sigma


# Default operating point — set by the combined-FPP-budget calibration on the calibration
# set in ``ADCNN.training.cnn_postproc.calibrate_combined_threshold`` and persisted in the
# sidecar JSON. The literal below is the fallback when the sidecar is absent.
CNN_DEFAULT_THR = 0.6
CUTOUT_K = 96          # cutout side length (px); shipped CNN cache + training defaults to 96.
NET_WIDTH = 40         # base conv width; matches the deployed checkpoint.
NET_DEPTH = 4          # number of conv blocks.
SCORE_BATCH = 512      # forward micro-batch for `apply_cnn`.
CLIP_SIGMA = 20.0      # diffim cutout clip range (in units of MAD-sigma); matches training.


def build_net(width: int = NET_WIDTH, depth: int = NET_DEPTH, in_ch: int = 3, k: int = CUTOUT_K):
    """The stage-2 cutout classifier: ``depth`` conv blocks (widths ``w, 2w, 4w, ...``) ->
    AdaptiveAvgPool2d(1) -> Dropout -> Linear. Fully convolutional w.r.t. ``k``, so larger
    cutouts work with no code change. The architecture is defined here ONCE; training imports
    it from this module.
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
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, 1))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.backbone = backbone
            s.head = head

        def forward(s, x):
            return s.head(s.backbone(x)).squeeze(1)
    return Net()


def load_cnn(path: str, device: str = "cpu"):
    """Load the focal cutout CNN. The architecture (``width / depth / k``) is read from the
    sidecar JSON next to ``path``; missing keys fall back to the module defaults."""
    import torch
    p = Path(path); sc = p.with_suffix(".json")
    info = json.loads(sc.read_text()) if sc.exists() else {}
    net = build_net(width=int(info.get("width", NET_WIDTH)),
                    depth=int(info.get("depth", NET_DEPTH)),
                    in_ch=int(info.get("in_ch", 3)),
                    k=int(info.get("k", CUTOUT_K))).to(device)
    net.load_state_dict(torch.load(str(p), map_location=device, weights_only=True))
    net.eval()
    return net


def read_threshold(path: str, default: float = CNN_DEFAULT_THR) -> float:
    """Read the calibrated CNN threshold from the sidecar JSON next to ``path``; fall back
    to ``default`` (== ``CNN_DEFAULT_THR``) when the sidecar is absent or missing the entry.
    """
    sc = Path(path).with_suffix(".json")
    if not sc.exists():
        return float(default)
    return float(json.loads(sc.read_text()).get("threshold", default))


def _cutout(arr: np.ndarray, x: float, y: float, k: int = CUTOUT_K) -> np.ndarray:
    """k x k patch of `arr` centred on (x, y), zero-padded where it runs off the panel edge."""
    h, w = arr.shape
    x, y = int(round(x)), int(round(y))
    hh = k // 2
    out = np.zeros((k, k), np.float32)
    x0, x1, y0, y1 = max(0, x - hh), min(w, x + hh), max(0, y - hh), min(h, y + hh)
    c = arr[y0:y1, x0:x1]
    # Place the patch at its OFFSET inside the stamp. Writing it at [0,0] left-shifts every source
    # clipped by the LEFT or TOP edge, so it is no longer centred -- and `_features` analyses
    # stamp[c-28:c+29] assuming it is. MEASURED: in the y<48 band the pre-link ring flag disagreed
    # with the (correct) QA path on 70.3% of detections vs 0.0% in the control band, i.e. the ring
    # veto was blind along the low edges. Right/bottom edges were always correct, which is why this
    # went unseen. Matches ADCNN.qa.alert_cutouts._cut, as ripple_flag's docstring promises.
    oy, ox = y0 - (y - hh), x0 - (x - hh)
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def make_cutouts(cand_df, img, prob, agg, *, k: int = CUTOUT_K) -> np.ndarray:
    """Build the (N, 3, k, k) cutout stack [diffim/sigma, seg_prob, seg_agg] for each candidate.
    The diffim channel is normalised by the panel MAD-sigma and clipped to ``[-CLIP_SIGMA, +CLIP_SIGMA]``
    (matches training in ``ADCNN.training.cnn_postproc``)."""
    img = np.asarray(img, np.float32)
    prob = np.asarray(prob, np.float32)
    agg = np.asarray(agg, np.float32)
    # MAD-sigma over FINITE pixels only: real DP2 diffims carry NaN/masked pixels.
    finite = img[np.isfinite(img)]
    # Use the CANONICAL estimator. This line used median(|x - median(x)|), a different formula that
    # returns exactly 0 on any panel >=50% masked -> `sig or 1.0` -> 1.0 -> the diffim channel becomes
    # img/1.0 clipped to +-20, i.e. saturated (5 of 120 sampled panels). It also cost 1.007 s/panel
    # measured, vs 0.615 s for the canonical one, ~284 CPU-min/night -- while cand_df["panel_sigma"]
    # already holds the answer. On clean panels the two agree to 1e-5, so this is a degenerate-panel
    # and cost fix, not a behaviour change on normal data.
    if "panel_sigma" in getattr(cand_df, "columns", []) and len(cand_df):
        sig = float(cand_df["panel_sigma"].iloc[0])
    else:
        sig = diffim_mad_sigma(img)
    sig = sig or 1.0
    if not len(cand_df):
        return np.zeros((0, 3, k, k), np.float32)
    X = np.stack([
        np.stack([_cutout(img, r.x_centroid, r.y_centroid, k) / sig,
                  _cutout(prob, r.x_centroid, r.y_centroid, k),
                  _cutout(agg, r.x_centroid, r.y_centroid, k)])
        for _, r in cand_df.iterrows()]).astype(np.float32)
    # Scrub NaN/inf to finite before the model sees them; no-op on clean sim data.
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=CLIP_SIGMA, neginf=-CLIP_SIGMA),
                   -CLIP_SIGMA, CLIP_SIGMA)


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
            cnn(torch.tensor(X[i:i + SCORE_BATCH]).to(device))
            for i in range(0, len(X), SCORE_BATCH)])).cpu().numpy()
    out[score_col] = s
    if thr is not None:
        out = out[out[score_col] >= thr].reset_index(drop=True)
    return out
