"""v7 (reg2, deployed) model: UNet + orientation head + Hough-like line aggregator.

The aggregator is the architectural ingredient v4 lacked: an explicit
"vote along a thin oriented line" operator. v4 demonstrated that the raw
UNet can suppress background, but cannot integrate sub-noise per-pixel
evidence along a line direction. The aggregator gives it that pathway.

The mechanism:
  raw_seg_logits  : (B, 1, H, W)  — UNet's per-pixel segmentation logits
  agg_per_angle   : (B, A, H, W)  — directional MEAN of raw_seg_logits
                                     over a length-L line at each of A angles
  agg_max         : (B, 1, H, W)  — max over angles
  final_logits    : (B, 1, H, W)  = raw + alpha * agg_max
                                     where alpha is a learnable scalar (init 0)

The directional kernels are thin lines (1-pixel wide along the angle)
drawn through the center of an LxL stamp via Bresenham. Sums to 1 per
kernel so the operation is a true MEAN over the line.

At init, alpha=0 → the head behaves like v4. As training progresses, the
optimizer can grow alpha if line aggregation helps. The combine is in
logit space (additive), which matches the way evidence accumulates.

Also exposes `orient_sin, orient_cos` like v4 for the orientation aux loss.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ADCNN.core.model import UNetResSE


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Thin line through the center of an LxL stamp at the given angle.
    The kernel sums to 1 (true average over the line).
    """
    L = int(length)
    assert L % 2 == 1, "kernel length must be odd"
    k = np.zeros((L, L), dtype=np.float32)
    c = L // 2
    dx = math.cos(math.radians(angle_deg))
    dy = math.sin(math.radians(angle_deg))
    # Sample along the line. Range of t = -c .. c.
    for t in range(-c, c + 1):
        x = int(round(c + t * dx))
        y = int(round(c + t * dy))
        if 0 <= x < L and 0 <= y < L:
            k[y, x] = 1.0
    s = k.sum()
    if s > 0:
        k /= s
    return k


class LineAggregator(nn.Module):
    """For each of `n_angles` orientations 0..180°, compute the mean of the
    input over a length-`kernel_len` line through each pixel. Then take the
    max across angles. Returns (B, 1, H, W).

    Multi-scale: if `kernel_lens` is a tuple, run the aggregation at each
    scale and max-pool the per-scale outputs as well. This handles trails
    of different lengths.
    """
    def __init__(self, kernel_lens=(11, 21, 41), n_angles: int = 12):
        super().__init__()
        self.n_angles = int(n_angles)
        self.kernel_lens = tuple(int(L) for L in kernel_lens)
        # Pre-build kernels per scale: each (n_angles, 1, L, L), sum=1 per slice.
        self._buf_names: list[str] = []
        for L in self.kernel_lens:
            kerns = np.stack(
                [_line_kernel(L, ang) for ang in np.linspace(0, 180, self.n_angles, endpoint=False)],
                axis=0,
            )[:, None]  # (n_angles, 1, L, L)
            name = f"kernels_L{L}"
            self.register_buffer(name, torch.from_numpy(kerns))
            self._buf_names.append(name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W). For each scale, conv2d with `n_angles` filters → (B, A, H, W).
        outs = []
        for name, L in zip(self._buf_names, self.kernel_lens):
            w = getattr(self, name)  # (A, 1, L, L)
            pad = L // 2
            o = F.conv2d(x, w, padding=pad)  # (B, A, H, W)
            outs.append(o.amax(dim=1, keepdim=True))  # max over orientations: (B, 1, H, W)
        if len(outs) == 1:
            return outs[0]
        stk = torch.stack(outs, dim=0)  # (S, B, 1, H, W)
        return stk.amax(dim=0)  # max over scales


class UNetResSEOrientHough(nn.Module):
    def __init__(self, in_ch: int = 3, widths=(48, 96, 192, 384, 768),
                 kernel_lens=(11, 21, 41), n_angles: int = 12, p_drop: float = 0.0):
        super().__init__()
        # UNet outputs 3 channels: seg_logit, orient_sin (pre-tanh), orient_cos (pre-tanh).
        self.backbone = UNetResSE(in_ch=in_ch, out_ch=3, widths=widths, p_drop=p_drop)
        self.line_agg = LineAggregator(kernel_lens=kernel_lens, n_angles=n_angles)
        # Learnable scalar mixing weight; init 0 so the network starts as
        # plain UNet + orient head (≡ v4 with 3-channel input) and grows
        # the aggregation contribution as training progresses.
        self.agg_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        raw_seg_logits = out[:, 0:1]
        orient_sin = torch.tanh(out[:, 1:2])
        orient_cos = torch.tanh(out[:, 2:3])

        # Directional aggregation of the raw seg LOGITS (additive evidence).
        agg = self.line_agg(raw_seg_logits)  # (B, 1, H, W)

        seg_logits = raw_seg_logits + self.agg_alpha * agg
        return seg_logits, orient_sin, orient_cos, raw_seg_logits, agg
