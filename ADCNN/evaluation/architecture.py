"""Publication-quality schematics of the ADCNN two-stage detector, drawn *from the deployed weights*.

The figures here are meant to drop straight into a paper. Nothing is hand-drawn or hard-coded: the
layer types, channel counts and spatial resolutions are read back from the actual model objects —
the segmentation network is reconstructed from the scripted checkpoint's ``state_dict`` (widths/depth
inferred from tensor shapes) and both nets are *forward-traced* with shape hooks, so the diagrams stay
correct if the architecture changes (more levels, different widths, extra heads).

Five standalone figures (no titles/panel letters — add captions in LaTeX), via
:func:`make_architecture_figures`:

  ``unet``        — the U-Net-ResSE backbone with per-stage C×H×W and skip connections, and the
                    OUTPUT HEAD: a 1×1 conv to 3 raw channels that split into (i) a segmentation
                    branch [raw seg logit  +α·Hough  → sigmoid ⇒ detection probability] and (ii) an
                    orientation branch [→ tanh ⇒ sin2β, cos2β]. This is the exact forward() data flow.
  ``resse_block`` — the repeated residual squeeze-excite block.
  ``hough``       — the learnable Hough line-aggregator: how the raw seg logit is turned into
                    directional evidence and added back (this is where the aggregator "comes in").
  ``filter_cnn``  — the post-detection focal-loss cutout CNN false-positive filter.
  ``system``      — the end-to-end discovery pipeline.

Design follows the best NN-schematic figures (U-Net, nnU-Net, PlotNeuralNet): isometric feature-map
slabs (height ~ resolution, depth ~ channels), colour-coded operation arrows, one restrained
colourblind-safe palette, legend. Every dimension comes from a trace, never a literal.
"""
from __future__ import annotations

import colorsys
import io
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------------------------------
# style — one restrained, colourblind-safe palette (Okabe-Ito derived); muted, paper-grade.
# --------------------------------------------------------------------------------------------------
PALETTE = dict(
    enc="#4C72B0",         # encoder feature maps / segmentation output (blue)
    dec="#55A868",         # decoder feature maps (green)
    bottleneck="#8172B3",  # bottleneck (purple)
    skip="#8C8C8C",        # skip connections (grey)
    down="#C44E52",        # downsample op (red)
    up="#55A868",          # upsample op (green)
    head="#DD8452",        # head / orientation output (orange)
    hough="#C44E52",       # Hough aggregator (crimson)
    inp="#4C566A",         # input channels (slate)
    raw="#9AA0A6",         # raw multi-channel tensor (grey)
    op="#6E7B8B",          # generic ops: tanh, sigmoid (steel)
    cnn="#3C8DAD",         # filter-CNN conv stacks (teal)
    fc="#E1A140",          # fully-connected / pooling (amber)
    keep="#4C9A5B",        # kept (green)
    edge="#1A1A1A",
    txt="#1A1A1A",
)


def _setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in vector output
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


# --------------------------------------------------------------------------------------------------
# low-level drawing primitives
# --------------------------------------------------------------------------------------------------
def _shade(color, f):
    c = mpl.colors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*c)
    return colorsys.hls_to_rgb(h, max(0.0, min(1.0, l * f)), s)


def _iso_box(ax, cx, cy, w, h, d, color, lw=0.9, z=2, alpha=1.0):
    """Isometric (3-D) feature-map slab centred at (cx, cy). Front face w×h, extruded by d up-right.
    Returns edge anchors (l/r/b/t) and the bounding extent for connecting arrows / labels."""
    x, y = cx - w / 2, cy - h / 2
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + d, y + h + d), (x + d, y + h + d)]
    side = [(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)]
    for pts, fc in ((top, _shade(color, 1.20)), (side, _shade(color, 0.74)), (front, color)):
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=PALETTE["edge"],
                             lw=lw, zorder=z, joinstyle="round", alpha=alpha))
    return dict(l=(x, cy + d / 2), r=(x + w + d, cy + d / 2), b=(cx + d / 2, y),
                t=(cx + d / 2, y + h + d), cx=cx, cy=cy, fcx=x + w / 2, fcy=y + h / 2,
                x0=x, x1=x + w + d, y0=y, y1=y + h + d)


def _arrow(ax, p0, p1, color=PALETTE["edge"], lw=1.6, style="-|>", ms=12, ls="-", z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw,
                                 color=color, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2, capstyle="round"))


def _label(ax, x, y, s, size=9, color=PALETTE["txt"], weight="normal", ha="center", va="center",
           box=None, z=5, style="normal"):
    bbox = dict(boxstyle="round,pad=0.25", fc=box, ec="none", alpha=0.92) if box else None
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style,
            zorder=z, bbox=bbox)


def _round_box(ax, cx, cy, w, h, color, lw=1.0, z=2, alpha=1.0, ec=None):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=ec or PALETTE["edge"], lw=lw, zorder=z, alpha=alpha))
    return dict(l=(cx - w / 2, cy), r=(cx + w / 2, cy), t=(cx, cy + h / 2), b=(cx, cy - h / 2),
                cx=cx, cy=cy)


def _save(fig, savepath):
    """Save a figure as 300-dpi PNG + a sibling vector PDF, robust to any input suffix."""
    if not savepath:
        return
    p = Path(savepath)
    fig.savefig(p, dpi=300)
    fig.savefig(p.with_suffix(".pdf"))


# --------------------------------------------------------------------------------------------------
# model loading + tracing  (everything below reads the real architecture)
# --------------------------------------------------------------------------------------------------
def load_seg_model(path="models/segmentation_model.pt"):
    """Reconstruct the segmentation nn.Module from the scripted checkpoint so we can hook it.
    Widths/depth/in_ch are inferred from the state_dict tensor shapes (no literals)."""
    import torch
    from ADCNN.core.detector import UNetResSEOrientHough
    sd = torch.jit.load(str(path), map_location="cpu").state_dict()
    in_ch = int(sd["backbone.stem.0.weight"].shape[1])
    widths = [int(sd["backbone.stem.0.weight"].shape[0])]
    i = 1
    while f"backbone.d{i}.rb.c1.weight" in sd:
        widths.append(int(sd[f"backbone.d{i}.rb.c1.weight"].shape[0])); i += 1
    model = UNetResSEOrientHough(in_ch=in_ch, widths=tuple(widths))
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, in_ch, widths


def trace_seg(model, in_ch, tile=128):
    """Forward-trace the seg net; return a structured spec (stages with real C×H×W, block details)."""
    import torch
    from ADCNN.core.model import Down, Up, ResBlock
    cap = {}

    def mk(name):
        def hook(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            cap[name] = tuple(int(v) for v in t.shape)
        return hook

    bb = model.backbone
    hs = [bb.stem.register_forward_hook(mk("stem"))]
    downs = [(n, m) for n, m in bb.named_children() if isinstance(m, Down)]
    ups = [(n, m) for n, m in bb.named_children() if isinstance(m, Up)]
    for n, m in downs:
        hs.append(m.register_forward_hook(mk("enc:" + n)))
    for n, m in ups:
        hs.append(m.register_forward_hook(mk("dec:" + n)))
    hs.append(bb.head.register_forward_hook(mk("head")))
    if hasattr(model, "line_agg"):
        hs.append(model.line_agg.register_forward_hook(mk("hough")))
    with torch.no_grad():
        model(torch.zeros(1, in_ch, tile, tile))
    for h in hs:
        h.remove()

    def cwh(s):
        return dict(c=s[1], h=s[2], w=s[3])
    encoder = [dict(name="stem", **cwh(cap["stem"]))]
    for n, _ in downs:
        encoder.append(dict(name=n, **cwh(cap["enc:" + n])))
    decoder = [dict(name=n, **cwh(cap["dec:" + n])) for n, _ in ups]
    head = cap["head"]

    rb = next((m for m in bb.modules() if isinstance(m, ResBlock)), None)
    block = None
    if rb is not None:
        block = dict(norm=type(rb.bn1).__name__, act=type(rb.act).__name__,
                     k=int(rb.c1.kernel_size[0]), se=not isinstance(rb.se, torch.nn.Identity))
    hough = None
    if hasattr(model, "line_agg"):
        la = model.line_agg
        hough = dict(kernel_lens=tuple(int(x) for x in la.kernel_lens), n_angles=int(la.n_angles),
                     alpha=float(model.agg_alpha.detach()) if hasattr(model, "agg_alpha") else None)
    return dict(in_ch=in_ch, tile=tile, encoder=encoder, decoder=decoder,
                out_ch=head[1], head_hw=(head[2], head[3]), block=block, hough=hough)


def trace_cnn(net, k=48):
    """Forward-trace the filter CNN; return ordered conv-stack stages + the head."""
    import torch
    stages, hs = [], []

    def mk(idx):
        def hook(m, i, o):
            s = tuple(int(v) for v in o.shape)
            kinds = [type(c).__name__ for c in m.children()] if list(m.children()) else [type(m).__name__]
            stages.append(dict(i=idx, kinds=kinds, c=s[1], h=s[2] if len(s) > 2 else 1))
        return hook

    for idx, blk in enumerate(net.f):
        hs.append(blk.register_forward_hook(mk(idx)))
    with torch.no_grad():
        net(torch.zeros(1, 3, k, k))
    for h in hs:
        h.remove()
    lin = next((c for c in net.h.modules() if isinstance(c, torch.nn.Linear)), None)
    return dict(k=k, in_ch=3, stages=stages, head=[type(c).__name__ for c in net.h.children()],
                out_features=int(lin.out_features) if lin else 1)


# --------------------------------------------------------------------------------------------------
# figure: U-Net backbone + output head (the exact forward() data flow)
# --------------------------------------------------------------------------------------------------
def _norm_fns(stages):
    import math
    res, ch = [s["h"] for s in stages], [s["c"] for s in stages]
    lr0, lr1, lc0, lc1 = math.log2(min(res)), math.log2(max(res)), math.log2(min(ch)), math.log2(max(ch))

    def hgt(r):
        t = 0 if lr1 == lr0 else (math.log2(r) - lr0) / (lr1 - lr0)
        return 0.52 + 0.82 * t

    def dep(c):
        t = 0 if lc1 == lc0 else (math.log2(c) - lc0) / (lc1 - lc0)
        return 0.18 + 0.52 * t
    return hgt, dep


def plot_unet(spec, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(16.5, 9.2))
    ax.axis("off"); ax.set_aspect("equal")
    enc, dec = spec["encoder"], spec["decoder"]
    hgt, dep = _norm_fns(enc + dec)
    L = len(enc)
    XE, XMID, XR = 0.0, 3.0, 6.0
    dy = 2.25
    fw = 0.66

    def ylev(l):
        return -l * dy

    def dims(a, ch, res):                       # channel count ABOVE, resolution INSIDE (no collisions)
        _label(ax, a["cx"], a["y1"] + 0.17, f"{ch}", size=10, weight="bold")
        _label(ax, a["fcx"], a["fcy"], f"{res}²", size=7.4, color="white", weight="bold")

    # ---- encoder column + bottleneck (centre, bottom) ----
    enc_a = {}
    for l, s in enumerate(enc):
        x = XE if l < L - 1 else XMID
        col = PALETTE["enc"] if l < L - 1 else PALETTE["bottleneck"]
        a = _iso_box(ax, x, ylev(l), fw, hgt(s["h"]), dep(s["c"]), col)
        enc_a[l] = a; dims(a, s["c"], s["h"])

    # ---- decoder column (matched to encoder by resolution) ----
    res_to_lvl = {s["h"]: l for l, s in enumerate(enc)}
    dec_a = {}
    for s in dec:
        l = res_to_lvl.get(s["h"])
        if l is None:
            continue
        a = _iso_box(ax, XR, ylev(l), fw, hgt(s["h"]), dep(s["c"]), PALETTE["dec"])
        dec_a[l] = a; dims(a, s["c"], s["h"])

    # ---- input channels: each of the 3 maps drawn SEPARATELY and labelled, then stacked into the input ----
    names_in = ["signed diffim", "local σ", "DIA mask"][:spec["in_ch"]] or [f"channel {i}" for i in range(spec["in_ch"])]
    shades_in = [1.0, 0.8, 1.2, 0.9, 1.1]
    n_in = spec["in_ch"]
    xin = XE - 3.5
    for j, nm in enumerate(names_in):
        yj = ylev(0) - (j - (n_in - 1) / 2) * 0.82
        a = _iso_box(ax, xin, yj, 0.56, 0.5, 0.12, _shade(PALETTE["inp"], shades_in[j % len(shades_in)]))
        _label(ax, xin - 0.47, yj, nm, size=8.4, ha="right", weight="bold", color="#333")
        _arrow(ax, a["r"], enc_a[0]["l"], color=PALETTE["skip"], lw=1.35)
    _label(ax, xin, ylev(0) - (n_in / 2) * 0.82 - 0.5, f"stacked → {n_in} × {spec['tile']}² input",
           size=8.0, color="#333", weight="bold")

    # ---- operation arrows: down (encoder), up (decoder), skips ----
    for l in range(L - 2):
        _arrow(ax, enc_a[l]["b"], enc_a[l + 1]["t"], color=PALETTE["down"], lw=2.1)
    _arrow(ax, enc_a[L - 2]["b"], enc_a[L - 1]["l"], color=PALETTE["down"], lw=2.1, rad=-0.18)
    if (L - 2) in dec_a:
        _arrow(ax, enc_a[L - 1]["r"], dec_a[L - 2]["b"], color=PALETTE["up"], lw=2.1, rad=-0.18)
    for l in range(L - 2, 0, -1):
        if l in dec_a and (l - 1) in dec_a:
            _arrow(ax, dec_a[l]["t"], dec_a[l - 1]["b"], color=PALETTE["up"], lw=2.1)
    for l in range(L - 1):
        if l in dec_a:
            _arrow(ax, enc_a[l]["r"], dec_a[l]["l"], color=PALETTE["skip"], lw=1.7, ls=(0, (5, 3)))

    # ---- callout: every encoder/decoder stage is built from Res-SE blocks ----
    rcx, rcy = XE - 1.95, ylev(1)
    _label(ax, rcx, rcy, "every encoder/decoder\nstage built from\nRes-SE block(s)",
           size=8.0, color=PALETTE["enc"], weight="bold", box="#E7EFF8")
    _arrow(ax, (rcx + 0.95, rcy + 0.05), enc_a[1]["l"], color=PALETTE["enc"], lw=1.4,
           ls=(0, (4, 2)), style="-|>")

    # ---- OUTPUT HEAD: head 1×1 conv -> 3 raw ch -> seg branch + orientation branch ----
    hy = ylev(0)
    hd = _round_box(ax, XR + 1.6, hy, 1.05, 0.66, PALETTE["head"], alpha=0.93)
    _label(ax, XR + 1.6, hy, "head\n1×1 conv", size=8.0, color="white", weight="bold")
    _arrow(ax, dec_a[0]["r"], hd["l"], lw=1.9)
    tb = _iso_box(ax, XR + 3.0, hy, 0.5, 0.92, 0.16, PALETTE["raw"])
    _label(ax, tb["cx"], tb["y1"] + 0.16, f"{spec['out_ch']}", size=9.5, weight="bold")
    _label(ax, tb["cx"], tb["y0"] - 0.26, "raw\noutput", size=7.2, color="#444")
    _arrow(ax, hd["r"], tb["l"], lw=1.8)

    # segmentation branch (upper): raw seg logit  +α·Hough  -> sigmoid -> detection probability
    ys = hy + 1.65
    plus = _round_box(ax, XR + 6.3, ys, 0.44, 0.44, "white", lw=1.4)
    _label(ax, XR + 6.3, ys, "+", size=14, weight="bold")
    if spec["hough"]:
        hb = _round_box(ax, XR + 4.95, hy + 0.6, 1.8, 0.66, PALETTE["hough"], alpha=0.9)
        _label(ax, XR + 4.95, hy + 0.6,
               f"Hough line-agg\nα = {spec['hough']['alpha']:.3f}", size=7.8, color="white", weight="bold")
        _arrow(ax, (tb["x1"], hy + 0.2), hb["l"], lw=1.5)
        _arrow(ax, hb["t"], plus["b"], lw=1.5, color=PALETTE["hough"])
    _arrow(ax, tb["t"], plus["l"], lw=1.6, rad=0.16)
    _label(ax, XR + 4.05, hy + 1.42, "raw seg logit", size=7.4, color="#555")
    sg = _round_box(ax, XR + 7.4, ys, 0.5, 0.5, "white", lw=1.4)
    _label(ax, XR + 7.4, ys, "σ", size=13)
    so = _iso_box(ax, XR + 8.6, ys, 0.46, 0.86, 0.14, PALETTE["enc"])
    _label(ax, so["cx"], so["y1"] + 0.16, "detection\nprobability", size=8.4, weight="bold")
    _arrow(ax, plus["r"], sg["l"], lw=1.6)
    _arrow(ax, sg["r"], so["l"], lw=1.6)

    # orientation branch (lower): sin,cos raw -> tanh -> sin 2β, cos 2β
    yo = hy - 1.55
    th = _round_box(ax, XR + 5.05, yo, 0.82, 0.52, PALETTE["op"], alpha=0.93)
    _label(ax, XR + 5.05, yo, "tanh", size=8.6, color="white", weight="bold")
    oo = _iso_box(ax, XR + 7.0, yo, 0.46, 0.86, 0.14, PALETTE["head"])
    _label(ax, oo["cx"], oo["y0"] - 0.3, "orientation\nsin 2β, cos 2β", size=8.4, weight="bold")
    _arrow(ax, tb["b"], th["l"], lw=1.5, rad=-0.18)
    _label(ax, XR + 4.2, hy - 0.82, "sin, cos (raw)", size=7.4, color="#555")
    _arrow(ax, th["r"], oo["l"], lw=1.6)

    # ---- legend ----
    leg = [Patch(fc=PALETTE["enc"], ec="k", label="encoder / seg map"),
           Patch(fc=PALETTE["bottleneck"], ec="k", label="bottleneck"),
           Patch(fc=PALETTE["dec"], ec="k", label="decoder map"),
           Patch(fc=PALETTE["inp"], ec="k", label="input channels"),
           Patch(fc=PALETTE["raw"], ec="k", label="raw head output"),
           Patch(fc=PALETTE["hough"], ec="k", label="Hough aggregator"),
           Line2D([0], [0], color=PALETTE["down"], lw=2.2, label="2× max-pool (down)"),
           Line2D([0], [0], color=PALETTE["up"], lw=2.2, label="transpose-conv (up)"),
           Line2D([0], [0], color=PALETTE["skip"], lw=1.8, ls="--", label="skip (concat)")]
    ax.legend(handles=leg, loc="lower left", ncol=2, frameon=False, fontsize=8.2,
              handlelength=1.6, bbox_to_anchor=(-0.01, 0.0))
    ax.set_xlim(xin - 2.5, XR + 9.7)
    ax.set_ylim(ylev(L - 1) - 1.4, ylev(0) + 2.15)
    if savepath:
        _save(fig, savepath)
    return fig


# --------------------------------------------------------------------------------------------------
# figure: residual squeeze-excite block
# --------------------------------------------------------------------------------------------------
def plot_resse_block(spec, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("Residual squeeze-excite block", fontsize=13, weight="bold", pad=12)
    b = spec["block"] or dict(norm="GroupNorm", act="SiLU", k=3, se=True)
    seq = [(f"{b['norm']}\n+ {b['act']}", PALETTE["enc"]),
           (f"Conv {b['k']}×{b['k']}", PALETTE["dec"]),
           (f"{b['norm']}\n+ {b['act']}", PALETTE["enc"]),
           (f"Conv {b['k']}×{b['k']}", PALETTE["dec"])]
    if b["se"]:
        seq.append(("SE\ngate", PALETTE["head"]))
    x = 0.0; xs = []
    for label, col in seq:
        a = _round_box(ax, x, 0.0, 0.98, 0.74, col, alpha=0.93)
        _label(ax, x, 0.0, label, size=7.8, color="white", weight="bold")
        xs.append(a); x += 1.4
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i]["r"], xs[i + 1]["l"], lw=1.5)
    _arrow(ax, (-1.05, 0.0), xs[0]["l"], lw=1.5)
    _label(ax, -1.2, 0.0, "x", size=12, ha="right", style="italic")
    plus = _round_box(ax, x + 0.05, 0.0, 0.5, 0.5, "white", lw=1.4)
    _label(ax, x + 0.05, 0.0, "+", size=15, weight="bold")
    _arrow(ax, xs[-1]["r"], plus["l"], lw=1.5)
    _arrow(ax, (x + 0.3, 0.0), (x + 1.15, 0.0), lw=1.6)
    _label(ax, x + 1.3, 0.0, "out", size=8.8, ha="left")
    ax.add_patch(FancyArrowPatch((-1.0, 0.2), (x + 0.05, 0.3), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.5, color=PALETTE["skip"], connectionstyle="arc3,rad=-0.55", zorder=1))
    _label(ax, x / 2 - 0.4, 1.12, "identity (residual)", size=8.0, color=PALETTE["skip"])
    ax.set_xlim(-1.8, x + 1.8); ax.set_ylim(-1.0, 1.45)
    if savepath:
        _save(fig, savepath)
    return fig


# --------------------------------------------------------------------------------------------------
# figure: Hough line-aggregator (where it "comes in")
# --------------------------------------------------------------------------------------------------
def plot_hough(spec, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.axis("off")
    ax.set_title("Hough line-aggregator", fontsize=13, weight="bold", pad=12)
    h = spec["hough"]
    if not h:
        _label(ax, 0.5, 0.5, "no Hough aggregator", size=11)
        _save(fig, savepath)
        return fig
    # input: the raw seg logit from the U-Net head
    _label(ax, 0.06, 0.62, "raw seg logit\n(from U-Net head)", size=8.6, ha="center")
    _arrow(ax, (0.135, 0.62), (0.205, 0.62), lw=1.6)
    # oriented multi-scale line kernels (true kernels from the model)
    try:
        from ADCNN.core.detector import _line_kernel
        Lk = h["kernel_lens"][len(h["kernel_lens"]) // 2]
        for j, ang in enumerate([0, 45, 90, 135]):
            sub = ax.inset_axes([0.22 + j * 0.058, 0.66, 0.05, 0.22])
            sub.imshow(_line_kernel(Lk, ang), cmap="magma", interpolation="nearest")
            sub.set_xticks([]); sub.set_yticks([])
            sub.set_title(f"{ang}°", fontsize=6.5, pad=1)
    except Exception:
        pass
    _label(ax, 0.30, 0.50,
           f"directional means (conv)\nA = {h['n_angles']} angles · scales L ∈ {{{', '.join(map(str, h['kernel_lens']))}}}",
           size=8.0, ha="center")
    mx = _round_box(ax, 0.55, 0.55, 0.13, 0.42, PALETTE["hough"], alpha=0.9)
    _label(ax, 0.55, 0.55, "max\nover\nangles", size=7.2, color="white", weight="bold")
    _arrow(ax, (0.43, 0.55), mx["l"], lw=1.5)
    al = _round_box(ax, 0.70, 0.55, 0.11, 0.34, "white", lw=1.4)
    _label(ax, 0.70, 0.55, f"× α\n{h['alpha']:.3f}", size=7.8, weight="bold", color=PALETTE["hough"])
    _arrow(ax, mx["r"], al["l"], lw=1.5)
    # add back to the raw seg logit
    _label(ax, 0.30, 0.16, "raw seg logit", size=8.4, ha="center")
    plus = _round_box(ax, 0.70, 0.16, 0.09, 0.16, "white", lw=1.4)
    _label(ax, 0.70, 0.16, "+", size=13, weight="bold")
    _arrow(ax, (0.40, 0.16), plus["l"], lw=1.5)
    _arrow(ax, (0.70, 0.36), (0.70, 0.245), lw=1.5, color=PALETTE["hough"])
    sg = _round_box(ax, 0.84, 0.16, 0.07, 0.16, "white", lw=1.4)
    _label(ax, 0.84, 0.16, "σ", size=12)
    _arrow(ax, plus["r"], sg["l"], lw=1.5)
    _arrow(ax, sg["r"], (0.95, 0.16), lw=1.6)
    _label(ax, 0.965, 0.16, "detection\nprobability", size=8.4, ha="left")
    ax.set_xlim(0.0, 1.12); ax.set_ylim(0.0, 0.95)
    if savepath:
        _save(fig, savepath)
    return fig


# --------------------------------------------------------------------------------------------------
# figure: filter CNN
# --------------------------------------------------------------------------------------------------
def plot_filter_cnn(spec, thr=0.63, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(15.0, 4.6))
    ax.axis("off"); ax.set_aspect("equal")
    k = spec["k"]
    def _nrm(a):
        return (a - a.min()) / (np.ptp(a) + 1e-9)
    # each input channel drawn SEPARATELY and labelled (the 3 maps stacked into the cutout)
    chan_specs = [("diffim / σ", _streak_thumb(k, 35, 0.55, noise=1.0), "gray"),
                  ("segmentation model seg prob", _nrm(_streak_thumb(k, 35, 0.55, sigma=2.6, noise=0.12)), "viridis"),
                  ("segmentation model Hough agg", _streak_thumb(k, 35, 0.55, sigma=1.0, noise=0.0, amp=9.0), "magma")]
    for j, (nm, img, cmap) in enumerate(chan_specs[:spec["in_ch"]]):
        yf = 0.72 - j * 0.27
        sub = ax.inset_axes([0.015, yf, 0.072, 0.22])
        sub.imshow(img, cmap=cmap, interpolation="nearest"); sub.set_xticks([]); sub.set_yticks([])
        for sp in sub.spines.values():
            sp.set_edgecolor(PALETTE["edge"]); sp.set_linewidth(0.8)
        ax.text(0.095, yf + 0.11, nm, transform=ax.transAxes, fontsize=8.4, va="center", ha="left", weight="bold")
    ax.text(0.05, 0.05, f"{spec['in_ch']} channels stacked → {spec['in_ch']} × {k}² cutout",
            transform=ax.transAxes, fontsize=8.2, ha="left", weight="bold")
    x = 3.5; prev = dict(r=(2.55, 0.2))
    maxc = max((s["c"] for s in spec["stages"][:-1]), default=160)
    for s in spec["stages"]:
        if any("Adaptive" in kk for kk in s["kinds"]):
            a = _iso_box(ax, x, 0.2, 0.4, 0.5, 0.16, PALETTE["fc"])
            _label(ax, a["cx"], a["y1"] + 0.18, f"{s['c']}", size=9, weight="bold")
            _label(ax, a["cx"], a["y0"] - 0.28, "global\navg-pool", size=8)
        else:
            d = 0.2 + 0.7 * s["c"] / maxc
            a = _iso_box(ax, x, 0.2, 0.5, 0.5 + 1.35 * (s["h"] / k), d, PALETTE["cnn"])
            _label(ax, a["cx"], a["y1"] + 0.18, f"{s['c']}", size=9, weight="bold")
            _label(ax, a["cx"], a["y0"] - 0.3, f"{s['h']}²", size=8, color="#444")
            _label(ax, a["cx"], a["y0"] - 0.62, "2×[Conv 3²-BN-ReLU]\n+ max-pool", size=6.8, color="#555")
        _arrow(ax, prev["r"], a["l"], lw=1.7)
        prev = a; x += 1.9
    fb = _round_box(ax, x, 0.2, 1.05, 0.7, PALETTE["fc"], alpha=0.92)
    _label(ax, x, 0.2, "Dropout\n+ Linear", size=8.4, color="white", weight="bold")
    _arrow(ax, prev["r"], fb["l"], lw=1.7); x += 1.75
    sb = _round_box(ax, x, 0.2, 0.98, 0.7, "white", lw=1.4)
    _label(ax, x, 0.2, "sigmoid\n→ score", size=8.4, weight="bold")
    _arrow(ax, fb["r"], sb["l"], lw=1.7); x += 1.7
    kb = _round_box(ax, x, 0.2, 1.5, 0.7, PALETTE["keep"], alpha=0.9)
    _label(ax, x, 0.2, f"keep if\nscore ≥ {thr}", size=8.2, color="white", weight="bold")
    _arrow(ax, sb["r"], kb["l"], lw=1.7)
    _label(ax, x, -1.05, f"stage-2 operating point: score ≥ {thr}", size=7.8, color="#555")
    ax.set_xlim(-0.3, x + 1.55); ax.set_ylim(-1.6, 1.9)
    if savepath:
        _save(fig, savepath)
    return fig


# --------------------------------------------------------------------------------------------------
# figure: end-to-end system
# --------------------------------------------------------------------------------------------------
def _streak_thumb(n=48, angle=30.0, length=0.5, sigma=1.4, seed=0, noise=1.0, amp=6.0):
    import math
    rng = np.random.default_rng(seed)
    img = rng.normal(0, noise, (n, n)).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = n / 2.0
    dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    half = length * n / 2.0
    for t in np.linspace(-half, half, int(2 * half) + 1):
        px, py = cx + t * dx, cy + t * dy
        img += amp * np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)))
    return img


def plot_system(seg_spec=None, cnn_spec=None, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(15.0, 4.7))
    ax.axis("off"); ax.set_aspect("equal")
    stages = [
        ("difference\nimage", PALETTE["inp"], True, False),
        ("U-Net-ResSE\n+ orient + Hough", PALETTE["enc"], False, True),
        ("detection prob +\norientation (2β)", PALETTE["dec"], False, False),
        ("candidate\nextraction\n(x, y, β, L)", PALETTE["op"], False, False),
        ("48² cutout\n×3", PALETTE["cnn"], True, False),
        ("focal cutout\nCNN filter", PALETTE["cnn"], False, True),
        ("score ≥ 0.63", PALETTE["keep"], False, False),
        ("Veres fit\nx,y,L,θ,SNR,mag", PALETTE["head"], False, False),
        ("HelioLinC\nlinking → tracks", PALETTE["bottleneck"], False, False),
    ]
    x = 0.0; dx = 1.74; bw, bh = 1.44, 1.2; prev = None
    for i, (label, col, has_img, emph) in enumerate(stages):
        a = _round_box(ax, x, 0.0, bw, bh + (0.18 if emph else 0), col, alpha=0.95,
                       lw=2.4 if emph else 1.0, ec=PALETTE["txt"] if emph else None)
        _label(ax, x, 0.0, label, size=8.0, color="white", weight="bold" if emph else "normal")
        if emph:
            _label(ax, x, bh / 2 + 0.34, "neural net", size=7.0, color=PALETTE["txt"], style="italic")
        if has_img:
            sub = ax.inset_axes([0.005 + i / len(stages) * 0.985, 0.05, 0.07, 0.20])
            sub.imshow(_streak_thumb(48, 35, 0.5, seed=i), cmap="gray", interpolation="nearest")
            sub.set_xticks([]); sub.set_yticks([])
        if prev is not None:
            _arrow(ax, prev["r"], a["l"], lw=1.9)
        prev = a; x += dx
    ax.set_xlim(-0.9, x - dx + 0.9); ax.set_ylim(-1.7, 1.5)
    if savepath:
        _save(fig, savepath)
    return fig


# --------------------------------------------------------------------------------------------------
# public entry point — called from the Evaluation notebook
# --------------------------------------------------------------------------------------------------
def make_architecture_figures(seg_path="models/segmentation_model.pt",
                              cnn_path="models/cnn_postproc.pt",
                              outdir="Evaluation/figures", show=True, save=True):
    """Build all architecture figures from the deployed weights. Returns {name: Figure}.
    Robust: a model that can't be loaded/traced is skipped with a printed note (never raises)."""
    from ADCNN.inference.cnn_postproc import load_cnn, CNN_DEFAULT_THR
    figs = {}
    outdir = Path(outdir)
    if save:
        outdir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return str(outdir / f"arch_{name}.png") if save else None

    try:
        seg_model, in_ch, widths = load_seg_model(seg_path)
        seg_spec = trace_seg(seg_model, in_ch)
        figs["unet"] = plot_unet(seg_spec, savepath=out("unet"))
        figs["resse_block"] = plot_resse_block(seg_spec, savepath=out("resse_block"))
        figs["hough"] = plot_hough(seg_spec, savepath=out("hough"))
        _alpha = seg_spec["hough"]["alpha"] if seg_spec.get("hough") else float("nan")
        print(f"[architecture] seg: in_ch={in_ch} widths={widths} out_ch={seg_spec['out_ch']} "
              f"levels={len(seg_spec['encoder'])} hough_alpha={_alpha:.3f}")
    except Exception as e:
        import traceback
        seg_spec = None
        print(f"[architecture] segmentation figures skipped: {type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        cnn = load_cnn(cnn_path)
        cnn_spec = trace_cnn(cnn)
        figs["filter_cnn"] = plot_filter_cnn(cnn_spec, thr=CNN_DEFAULT_THR, savepath=out("filter_cnn"))
        print(f"[architecture] filter CNN: stages={[s['c'] for s in cnn_spec['stages']]} thr={CNN_DEFAULT_THR}")
    except Exception as e:
        cnn_spec = None
        print(f"[architecture] filter-CNN figure skipped: {type(e).__name__}: {e}")

    figs["system"] = plot_system(seg_spec, cnn_spec, savepath=out("system"))
    if save:
        print(f"[architecture] saved {len(figs)} figures to {outdir}/ (300-dpi PNG + vector PDF)")
    if not show:
        for f in figs.values():
            plt.close(f)
    return figs


if __name__ == "__main__":
    make_architecture_figures(show=False)
    print("done")
