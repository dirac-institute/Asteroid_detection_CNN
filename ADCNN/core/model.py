"""UNet-ResSE backbone — the feature extractor under the v7 diffim detector.

A residual squeeze-and-excite U-Net. ``UNetResSEOrientHough`` (in ``diffim_model.py``)
wraps this backbone with ``out_ch=3`` (segmentation logit + orientation sin2β/cos2β)
plus a Hough-style line aggregator. The block implementations below are kept exactly
as the deployed reg2 weights were trained against — do not alter their arithmetic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-excite channel attention: global-avg-pool → 1×1 bottleneck → sigmoid gate."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


def _norm(c, groups=8):
    """GroupNorm with `groups` groups, falling back to 1 group when c is not divisible."""
    g = min(groups, c) if c % groups == 0 else 1
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    """Pre-activation residual block (GroupNorm-SiLU-conv ×2) with optional SE gate
    and a 1×1 projection shortcut when in/out channels differ."""
    def __init__(self, c_in, c_out, k=3, act=nn.SiLU, se=True):
        super().__init__()
        p = k // 2
        self.proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)
        self.bn1 = _norm(c_in);  self.c1 = nn.Conv2d(c_in, c_out, k, padding=p, bias=False)
        self.bn2 = _norm(c_out); self.c2 = nn.Conv2d(c_out, c_out, k, padding=p, bias=False)
        self.act = act();        self.se = SEBlock(c_out) if se else nn.Identity()

    def forward(self, x):
        h = self.act(self.bn1(x)); h = self.c1(h)
        h = self.act(self.bn2(h)); h = self.c2(h)
        h = self.se(h)
        return h + self.proj(x)


class Down(nn.Module):
    """U-Net encoder stage: 2× max-pool then a ResBlock."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.rb = ResBlock(c_in, c_out)

    def forward(self, x):
        return self.rb(self.pool(x))


class Up(nn.Module):
    """U-Net decoder stage: transpose-conv upsample, pad-to-skip, concat skip, 2 ResBlocks."""
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in, 2, stride=2)
        self.rb1 = ResBlock(c_in + c_skip, c_out)
        self.rb2 = ResBlock(c_out, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(-2) - x.size(-2); dw = skip.size(-1) - x.size(-1)
        if dh or dw:  # pad up to the skip size (tile=128 is power-of-two so this rarely fires)
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
        x = torch.cat([x, skip], 1)
        x = self.rb1(x); x = self.rb2(x)
        return x


class UNetResSE(nn.Module):
    """Residual squeeze-excite U-Net backbone (5 levels).

    In v7/reg2: ``in_ch=3`` (signed diffim, local-std, DIA mask), ``out_ch=3``
    (seg logit + orientation sin2β/cos2β), ``widths=(24,48,96,192,384)``. Spatial
    dropout is applied at the bottleneck and pre-head; ``p_drop=0`` → Identity
    (byte-identical to the un-regularised net).
    """
    def __init__(self, in_ch=1, out_ch=1, widths=(48, 96, 192, 384, 768), p_drop=0.0):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.SiLU(True), ResBlock(w[0], w[0]))
        self.d1 = Down(w[0], w[1]); self.d2 = Down(w[1], w[2]); self.d3 = Down(w[2], w[3]); self.d4 = Down(w[3], w[4])
        self.u1 = Up(w[4], w[3], w[3]); self.u2 = Up(w[3], w[2], w[2]); self.u3 = Up(w[2], w[1], w[1]); self.u4 = Up(w[1], w[0], w[0])
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.head = nn.Conv2d(w[0], out_ch, 1)

    def forward(self, x):
        s0 = self.stem(x); s1 = self.d1(s0); s2 = self.d2(s1); s3 = self.d3(s2); b = self.drop(self.d4(s3))
        x = self.u1(b, s3); x = self.u2(x, s2); x = self.u3(x, s1); x = self.u4(x, s0)
        return self.head(self.drop(x))
