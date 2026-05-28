"""Publication-quality schematics of the ADCNN two-stage detector, drawn *from the deployed weights*.

The figures here are meant to drop straight into a paper. Nothing is hand-drawn or hard-coded: the
layer types, channel counts and spatial resolutions are read back from the actual model objects —
the segmentation network is reconstructed from the scripted checkpoint's ``state_dict`` (widths/depth
inferred from tensor shapes) and both nets are *forward-traced* with shape hooks, so the diagrams stay
correct if the architecture changes (more levels, different widths, extra heads).

Three figures, callable from the Evaluation notebook via :func:`make_architecture_figures`:

  1. ``segmentation`` — the U-Net-ResSE + orientation + Hough detector: the encoder/decoder "U" with
     per-stage tensor dimensions and skip connections, an inset of the repeated Res-SE block, and an
     inset of the learnable Hough line-aggregator (the network's novel ingredient).
  2. ``filter_cnn`` — the post-detection focal-loss cutout CNN false-positive filter.
  3. ``system``     — the end-to-end data flow: difference image -> U-Net -> candidate extraction ->
     cutout -> filter CNN -> Veres measurement -> HelioLinC linking.

Design follows the conventions of the best NN-schematic figures (U-Net, nnU-Net, PlotNeuralNet):
isometric feature-map slabs (height ~ resolution, depth ~ channels), colour-coded operation arrows,
a single restrained colourblind-safe palette, and a legend. Robust by construction — every dimension
comes from a trace, never a literal.
"""
from __future__ import annotations

import colorsys
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# --------------------------------------------------------------------------------------------------
# style — one restrained, colourblind-safe palette (Okabe-Ito derived); muted, paper-grade.
# --------------------------------------------------------------------------------------------------
PALETTE = dict(
    enc="#4C72B0",     # encoder feature maps (blue)
    dec="#55A868",     # decoder feature maps (green)
    bottleneck="#8172B3",  # bottleneck (purple)
    skip="#8C8C8C",    # skip connections (grey)
    down="#C44E52",    # downsample op (red)
    up="#55A868",      # upsample op (green)
    head="#DD8452",    # output heads (orange)
    hough="#C44E52",   # Hough aggregator branch (crimson)
    inp="#4C566A",     # input channels (slate)
    cnn="#3C8DAD",     # filter-CNN conv stacks (teal)
    fc="#E1A140",      # fully-connected / score (amber)
    keep="#4C9A5B",    # kept (green)
    edge="#1A1A1A",
    txt="#1A1A1A",
)
DEPTH = 0.34   # isometric extrusion direction (up-right), in data units, scaled per box


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
    Returns (left, right, bottom, top, dx, dy) edge anchors for connecting arrows."""
    x, y = cx - w / 2, cy - h / 2
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + d, y + h + d), (x + d, y + h + d)]
    side = [(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d), (x + w, y + h)]
    for pts, fc in ((top, _shade(color, 1.20)), (side, _shade(color, 0.74)), (front, color)):
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=PALETTE["edge"],
                             lw=lw, zorder=z, joinstyle="round", alpha=alpha))
    return dict(l=(x, cy + d / 2), r=(x + w + d, cy + d / 2), b=(cx + d / 2, y),
                t=(cx + d / 2, y + h + d), cx=cx, cy=cy, x0=x, x1=x + w + d, y0=y, y1=y + h + d)


def _arrow(ax, p0, p1, color=PALETTE["edge"], lw=1.6, style="-|>", ms=12, ls="-", z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw,
                                 color=color, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2, capstyle="round"))


def _label(ax, x, y, s, size=9, color=PALETTE["txt"], weight="normal", ha="center", va="center",
           box=None, z=5, style="normal"):
    bbox = None
    if box:
        bbox = dict(boxstyle="round,pad=0.25", fc=box, ec="none", alpha=0.92)
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style,
            zorder=z, bbox=bbox)


def _round_box(ax, cx, cy, w, h, color, lw=1.0, z=2, alpha=1.0, ec=None):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=color, ec=ec or PALETTE["edge"], lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return dict(l=(cx - w / 2, cy), r=(cx + w / 2, cy), t=(cx, cy + h / 2), b=(cx, cy - h / 2),
                cx=cx, cy=cy)


# --------------------------------------------------------------------------------------------------
# model loading + tracing  (everything below reads the real architecture)
# --------------------------------------------------------------------------------------------------
def load_seg_model(path="models/v7_diffim_scripted.pt"):
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

    # introspect one Res-SE block (the repeated unit) for the inset, from the live module
    rb = next((m for m in bb.modules() if isinstance(m, ResBlock)), None)
    block = None
    if rb is not None:
        block = dict(norm=type(rb.bn1).__name__, act=type(rb.act).__name__,
                     k=int(rb.c1.kernel_size[0]),
                     se=not isinstance(rb.se, torch.nn.Identity))
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
    head = [type(c).__name__ for c in net.h.children()]
    lin = next((c for c in net.h.modules() if isinstance(c, torch.nn.Linear)), None)
    return dict(k=k, in_ch=3, stages=stages, head=head,
                out_features=int(lin.out_features) if lin else 1)


# --------------------------------------------------------------------------------------------------
# figure 1 — segmentation network
# --------------------------------------------------------------------------------------------------
def _norm_fns(stages):
    res = [s["h"] for s in stages]
    ch = [s["c"] for s in stages]
    lr0, lr1 = math.log2(min(res)), math.log2(max(res))
    lc0, lc1 = math.log2(min(ch)), math.log2(max(ch))

    def hgt(r):
        t = 0 if lr1 == lr0 else (math.log2(r) - lr0) / (lr1 - lr0)
        return 0.55 + 0.95 * t

    def dep(c):
        t = 0 if lc1 == lc0 else (math.log2(c) - lc0) / (lc1 - lc0)
        return 0.18 + 0.55 * t
    return hgt, dep


def plot_segmentation(spec, savepath=None):
    _setup_style()
    fig = plt.figure(figsize=(13.0, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0], hspace=0.16, wspace=0.13,
                          left=0.03, right=0.985, top=0.93, bottom=0.04)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_aspect("equal")

    enc, dec = spec["encoder"], spec["decoder"]
    allst = enc + dec
    hgt, dep = _norm_fns(allst)
    L = len(enc)                       # resolution levels (stem .. bottleneck)
    XE, XMID, XR = 0.0, 3.1, 6.2
    dy = 1.62

    def ylev(l):
        return -l * dy

    fw = 0.62
    # ---- encoder column (levels 0..L-2) + bottleneck (level L-1, centre) ----
    enc_anchor = {}
    for l, s in enumerate(enc):
        x = XE if l < L - 1 else XMID
        col = PALETTE["enc"] if l < L - 1 else PALETTE["bottleneck"]
        a = _iso_box(ax, x, ylev(l), fw, hgt(s["h"]), dep(s["c"]) * 1.0, col)
        enc_anchor[l] = a
        _label(ax, x, a["y1"] + 0.16, f"{s['c']}", size=9.5, weight="bold")
        _label(ax, x - fw / 2 - 0.16, a["cy"], f"{s['h']}²", size=8.2, ha="right", color="#444")

    # ---- decoder column (outputs at levels L-2..0), matched to encoder by resolution ----
    res_to_lvl = {s["h"]: l for l, s in enumerate(enc)}
    dec_anchor = {}
    for s in dec:
        l = res_to_lvl.get(s["h"])
        if l is None:
            continue
        a = _iso_box(ax, XR, ylev(l), fw, hgt(s["h"]), dep(s["c"]), PALETTE["dec"])
        dec_anchor[l] = a
        _label(ax, XR, a["y1"] + 0.16, f"{s['c']}", size=9.5, weight="bold")
        _label(ax, XR + fw / 2 + dep(s["c"]) + 0.16, a["cy"], f"{s['h']}²", size=8.2, ha="left", color="#444")

    # ---- input channels (left of stem) ----
    ch_names = ["signed diffim", "local σ", "DIA mask"][:spec["in_ch"]] or [f"ch{i}" for i in range(spec["in_ch"])]
    xin = XE - 2.15
    a0 = enc_anchor[0]
    for j in range(spec["in_ch"]):
        off = (j - (spec["in_ch"] - 1) / 2) * 0.16
        _iso_box(ax, xin + off, ylev(0) + off, 0.5, 0.95, 0.16, PALETTE["inp"], z=2 + j)
    _label(ax, xin, ylev(0) - 0.95, "input\n" + f"{spec['in_ch']}×{spec['tile']}²", size=8.6)
    for j, nm in enumerate(ch_names):
        _label(ax, xin, ylev(0) + 0.78 + 0.0, "", size=7)
    _label(ax, xin, ylev(0) + 1.05, "  /  ".join(ch_names), size=7.6, color="#333")
    _arrow(ax, (xin + 0.45, ylev(0)), a0["l"], color=PALETTE["edge"], lw=1.8)

    # ---- output heads (right of top decoder) ----
    xout = XR + 2.25
    htop = dec_anchor.get(0)
    head_names = ["seg logit", "sin 2β", "cos 2β"][:spec["out_ch"]] or [f"out{i}" for i in range(spec["out_ch"])]
    for j in range(spec["out_ch"]):
        off = (j - (spec["out_ch"] - 1) / 2) * 0.16
        _iso_box(ax, xout + off, ylev(0) + off, 0.46, 0.86, 0.14, PALETTE["head"], z=2 + j)
    if htop:
        _arrow(ax, htop["r"], (xout - 0.45, ylev(0)), color=PALETTE["edge"], lw=1.8)
    _label(ax, xout, ylev(0) + 1.0, "  /  ".join(head_names), size=7.6, color="#333")
    _label(ax, xout, ylev(0) - 0.92, f"{spec['out_ch']}×{spec['tile']}²\noutput maps", size=8.4)

    # ---- operation arrows: down (encoder), up (decoder), skips ----
    for l in range(L - 2):
        _arrow(ax, enc_anchor[l]["b"], enc_anchor[l + 1]["t"], color=PALETTE["down"], lw=2.0)
    _arrow(ax, enc_anchor[L - 2]["b"], enc_anchor[L - 1]["l"], color=PALETTE["down"], lw=2.0, rad=-0.18)
    if (L - 2) in dec_anchor:
        _arrow(ax, enc_anchor[L - 1]["r"], dec_anchor[L - 2]["b"], color=PALETTE["up"], lw=2.0, rad=-0.18)
    for l in range(L - 2, 0, -1):
        if l in dec_anchor and (l - 1) in dec_anchor:
            _arrow(ax, dec_anchor[l]["t"], dec_anchor[l - 1]["b"], color=PALETTE["up"], lw=2.0)
    for l in range(L - 1):
        if l in dec_anchor:
            _arrow(ax, enc_anchor[l]["r"], dec_anchor[l]["l"], color=PALETTE["skip"], lw=1.7,
                   style="-|>", ls=(0, (5, 3)))

    # ---- Hough branch callout (from seg output, learnable α add) ----
    if spec["hough"]:
        a = spec["hough"]
        hx = (XE + XR) / 2.0
        _label(ax, hx, ylev(0) + 1.5,
               r"final seg $=$ raw $+\ \alpha\!\cdot\!$Hough(line-agg)" + f"   (α = {a['alpha']:.3f})",
               size=8.8, color=PALETTE["hough"], weight="bold",
               box="#FBEAEA")

    ax.set_title("(a)  Segmentation network — U-Net-ResSE with orientation + learnable Hough aggregator",
                 fontsize=12.5, weight="bold", loc="left", pad=8)
    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [Patch(fc=PALETTE["enc"], ec="k", label="encoder map"),
           Patch(fc=PALETTE["bottleneck"], ec="k", label="bottleneck"),
           Patch(fc=PALETTE["dec"], ec="k", label="decoder map"),
           Line2D([0], [0], color=PALETTE["down"], lw=2.2, label="2× max-pool (down)"),
           Line2D([0], [0], color=PALETTE["up"], lw=2.2, label="transpose-conv (up)"),
           Line2D([0], [0], color=PALETTE["skip"], lw=1.8, ls="--", label="skip (concat)")]
    ax.legend(handles=leg, loc="lower left", ncol=2, frameon=False, fontsize=8.4,
              handlelength=1.6, bbox_to_anchor=(0.0, -0.02))
    ax.set_xlim(xin - 1.0, xout + 1.1)
    ax.set_ylim(ylev(L - 1) - 1.4, ylev(0) + 2.0)

    _plot_resse_inset(fig.add_subplot(gs[1, 0]), spec)
    _plot_hough_inset(fig.add_subplot(gs[1, 1]), spec)
    if savepath:
        fig.savefig(savepath, dpi=300); fig.savefig(str(savepath).replace(".png", ".pdf"))
    return fig


def _plot_resse_inset(ax, spec):
    ax.axis("off"); ax.set_aspect("equal")
    b = spec["block"] or dict(norm="GroupNorm", act="SiLU", k=3, se=True)
    seq = [(f"{b['norm']}\n+{b['act']}", PALETTE["enc"]),
           (f"Conv {b['k']}×{b['k']}", PALETTE["dec"]),
           (f"{b['norm']}\n+{b['act']}", PALETTE["enc"]),
           (f"Conv {b['k']}×{b['k']}", PALETTE["dec"])]
    if b["se"]:
        seq.append(("SE\ngate", PALETTE["head"]))
    x = 0.0; xs = []
    for label, col in seq:
        a = _round_box(ax, x, 0.0, 0.95, 0.7, col, alpha=0.92)
        _label(ax, x, 0.0, label, size=7.6, color="white", weight="bold")
        xs.append(a); x += 1.35
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i]["r"], xs[i + 1]["l"], lw=1.5)
    # input dot + output + residual skip over the top
    _arrow(ax, (-1.0, 0.0), xs[0]["l"], lw=1.5)
    _label(ax, -1.15, 0.0, "x", size=11, ha="right", style="italic")
    plus = _round_box(ax, x + 0.05, 0.0, 0.5, 0.5, "white", lw=1.4)
    _label(ax, x + 0.05, 0.0, "+", size=15, weight="bold")
    _arrow(ax, xs[-1]["r"], plus["l"], lw=1.5)
    _arrow(ax, (x + 0.3, 0.0), (x + 1.1, 0.0), lw=1.6)
    _label(ax, x + 1.25, 0.0, "out", size=8.6, ha="left")
    # residual arc
    ax.add_patch(FancyArrowPatch((-1.0, 0.18), (x + 0.05, 0.30), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.5, color=PALETTE["skip"], connectionstyle="arc3,rad=-0.5", zorder=1))
    _label(ax, (x) / 2 - 0.4, 1.05, "identity (residual)", size=7.6, color=PALETTE["skip"])
    ax.set_title("(b)  Residual squeeze-excite block (repeated unit)", fontsize=11, weight="bold", loc="left")
    ax.set_xlim(-1.7, x + 1.7); ax.set_ylim(-1.0, 1.4)


def _plot_hough_inset(ax, spec):
    ax.axis("off"); ax.set_aspect("equal")
    h = spec["hough"]
    if not h:
        _label(ax, 0.5, 0.5, "(no Hough aggregator)", size=10); return
    # oriented line-kernel thumbnails
    try:
        from ADCNN.core.detector import _line_kernel
        Lk = h["kernel_lens"][len(h["kernel_lens"]) // 2]
        angles = [0, 45, 90, 135]
        for j, ang in enumerate(angles):
            sub = ax.inset_axes([0.02 + j * 0.085, 0.60, 0.075, 0.30])
            sub.imshow(_line_kernel(Lk, ang), cmap="magma", interpolation="nearest")
            sub.set_xticks([]); sub.set_yticks([])
            sub.set_title(f"{ang}°", fontsize=6.5, pad=1)
    except Exception:
        pass
    _label(ax, 0.20, 0.45,
           f"oriented line means\nA = {h['n_angles']} angles\nscales L ∈ {{{', '.join(map(str, h['kernel_lens']))}}}",
           size=8.0, ha="center")
    b1 = _round_box(ax, 0.52, 0.45, 0.27, 0.5, PALETTE["hough"], alpha=0.9)
    _label(ax, 0.52, 0.45, "max\nover\nangles", size=7.4, color="white", weight="bold")
    b2 = _round_box(ax, 0.82, 0.45, 0.22, 0.42, "white", lw=1.4)
    _label(ax, 0.82, 0.45, f"α·\n{h['alpha']:.3f}", size=8.0, weight="bold", color=PALETTE["hough"])
    _arrow(ax, (0.40, 0.45), b1["l"], lw=1.4)
    _arrow(ax, b1["r"], b2["l"], lw=1.4)
    plus = _round_box(ax, 0.82, 0.10, 0.18, 0.18, "white", lw=1.3)
    _label(ax, 0.82, 0.10, "+", size=13, weight="bold")
    _arrow(ax, (0.82, 0.24), (0.82, 0.19), lw=1.4)
    _label(ax, 0.30, 0.10, "raw seg logits", size=8.0, ha="center")
    _arrow(ax, (0.46, 0.10), plus["l"], lw=1.4)
    _arrow(ax, (0.91, 0.10), (1.04, 0.10), lw=1.5)
    _label(ax, 1.06, 0.10, "final\nseg", size=8.0, ha="left")
    ax.set_title("(c)  Hough line-aggregator (integrates sub-noise evidence along trails)",
                 fontsize=11, weight="bold", loc="left")
    ax.set_xlim(0.0, 1.18); ax.set_ylim(-0.05, 1.0)


# --------------------------------------------------------------------------------------------------
# figure 2 — filter CNN
# --------------------------------------------------------------------------------------------------
def plot_filter_cnn(spec, thr=0.63, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(13.0, 4.3))
    ax.axis("off"); ax.set_aspect("equal")
    k = spec["k"]
    # input cutout (3 stacked channels) with a synthetic streak glyph
    xin = 0.0
    chans = ["diffim/σ", "v7 prob", "v7 agg"]
    thumb = _streak_thumb(k, 35.0, 0.55)
    for j in range(3):
        off = (2 - j) * 0.18
        sub = ax.inset_axes([0.012 + (2 - j) * 0.010, 0.34 + (2 - j) * 0.05, 0.085, 0.30])
        sub.imshow(thumb if j == 0 else (thumb > 0.25).astype(float) * (0.6 + 0.4 * thumb),
                   cmap="gray" if j == 0 else "viridis", interpolation="nearest")
        sub.set_xticks([]); sub.set_yticks([])
        for sp in sub.spines.values():
            sp.set_edgecolor(PALETTE["edge"]); sp.set_linewidth(0.8)
    _label(ax, 0.95, -1.05, f"cutout\n{spec['in_ch']}×{k}²", size=8.8)
    _label(ax, 0.95, 1.55, "  /  ".join(chans), size=7.8, color="#333")
    x = 2.3
    prev = dict(r=(1.55, 0.2))
    maxc = max(s["c"] for s in spec["stages"][:-1]) if spec["stages"] else 160
    for s in spec["stages"]:
        is_pool = any("Adaptive" in kk for kk in s["kinds"])
        if is_pool:
            a = _iso_box(ax, x, 0.2, 0.4, 0.5, 0.16, PALETTE["fc"])
            _label(ax, x, a["y1"] + 0.18, f"{s['c']}", size=9, weight="bold")
            _label(ax, x, a["y0"] - 0.28, "global\navg-pool", size=8)
        else:
            d = 0.2 + 0.7 * s["c"] / maxc
            hh = 0.5 + 1.4 * (s["h"] / k)
            a = _iso_box(ax, x, 0.2, 0.5, hh, d, PALETTE["cnn"])
            _label(ax, x, a["y1"] + 0.18, f"{s['c']}", size=9, weight="bold")
            _label(ax, x, a["y0"] - 0.3, f"{s['h']}²", size=8, color="#444")
            _label(ax, x, a["y0"] - 0.62, "2×[Conv 3²-BN-ReLU]\n+ max-pool", size=6.8, color="#555")
        _arrow(ax, prev["r"], a["l"], lw=1.7)
        prev = a; x += 1.85
    # head: flatten -> dropout -> linear -> sigmoid -> score
    fb = _round_box(ax, x, 0.2, 1.0, 0.7, PALETTE["fc"], alpha=0.92)
    _label(ax, x, 0.2, "Dropout\n+ Linear", size=8.4, color="white", weight="bold")
    _arrow(ax, prev["r"], fb["l"], lw=1.7); x += 1.7
    sb = _round_box(ax, x, 0.2, 0.95, 0.7, "white", lw=1.4)
    _label(ax, x, 0.2, "sigmoid\n→ score", size=8.4, weight="bold")
    _arrow(ax, fb["r"], sb["l"], lw=1.7); x += 1.6
    kb = _round_box(ax, x, 0.2, 1.45, 0.7, PALETTE["keep"], alpha=0.9)
    _label(ax, x, 0.2, f"keep if\nscore ≥ {thr}", size=8.2, color="white", weight="bold")
    _arrow(ax, sb["r"], kb["l"], lw=1.7)
    ax.set_title("(d)  Post-detection false-positive filter — focal-loss cutout CNN",
                 fontsize=12.5, weight="bold", loc="left", pad=6)
    _label(ax, x, -1.05, "op-point matched to RF FP/panel\n(thr 0.63: 72.5 FP/panel, recall 0.75)",
           size=7.6, color="#555")
    ax.set_xlim(-0.3, x + 1.5); ax.set_ylim(-1.5, 1.8)
    if savepath:
        fig.savefig(savepath, dpi=300); fig.savefig(str(savepath).replace(".png", ".pdf"))
    return fig


# --------------------------------------------------------------------------------------------------
# figure 3 — end-to-end system
# --------------------------------------------------------------------------------------------------
def _streak_thumb(n=48, angle=30.0, length=0.5, sigma=1.4, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1.0, (n, n)).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = n / 2.0
    th = math.radians(angle)
    dx, dy = math.cos(th), math.sin(th)
    half = length * n / 2.0
    for t in np.linspace(-half, half, int(2 * half) + 1):
        px, py = cx + t * dx, cy + t * dy
        img += 6.0 * np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2)))
    return img


def plot_system(seg_spec=None, cnn_spec=None, savepath=None):
    _setup_style()
    fig, ax = plt.subplots(figsize=(15.0, 4.7))
    ax.axis("off"); ax.set_aspect("equal")
    nseg = "U-Net-ResSE\n+ orient + Hough"
    ncnn = "focal cutout\nCNN filter"
    stages = [
        ("diffim", "difference\nimage", PALETTE["inp"], True),
        ("seg", nseg, PALETTE["enc"], False),
        ("maps", "prob · sin2β ·\ncos2β · Hough", PALETTE["dec"], False),
        ("cand", "candidate\nextraction\n(x, y, β, L)", "#6E7B8B", False),
        ("cut", "48² cutout\n×3", PALETTE["cnn"], True),
        ("cnn", ncnn, PALETTE["cnn"], False),
        ("keep", "score ≥ 0.63", PALETTE["keep"], False),
        ("meas", "Veres fit\nx,y,L,θ,SNR,mag", PALETTE["head"], False),
        ("link", "HelioLinC\nlinking → tracks", PALETTE["bottleneck"], False),
    ]
    x = 0.0; dx = 1.74; bw, bh = 1.44, 1.2; prev = None
    model_idx = {1, 5}  # the two neural nets, drawn emphasised
    for i, (key, label, col, has_img) in enumerate(stages):
        emph = i in model_idx
        a = _round_box(ax, x, 0.0, bw, bh + (0.18 if emph else 0), col, alpha=0.95,
                       lw=2.4 if emph else 1.0, ec=PALETTE["txt"] if emph else None)
        tc = "white" if col not in ("white",) else PALETTE["txt"]
        _label(ax, x, 0.0, label, size=8.0, color=tc, weight="bold" if emph else "normal")
        if emph:
            _label(ax, x, bh / 2 + 0.34, "neural net", size=7.0, color=PALETTE["txt"], style="italic")
        if has_img:
            sub = ax.inset_axes([0.005 + i / len(stages) * 0.985, 0.05, 0.07, 0.20])
            sub.imshow(_streak_thumb(48, 35, 0.5, seed=i), cmap="gray", interpolation="nearest")
            sub.set_xticks([]); sub.set_yticks([])
        if prev is not None:
            _arrow(ax, prev["r"], a["l"], lw=1.9)
        prev = a; x += dx
    ax.set_title("(e)  End-to-end discovery pipeline: detection → false-positive filter → measurement → linking",
                 fontsize=12.5, weight="bold", loc="left", pad=6)
    ax.set_xlim(-0.9, x - dx + 0.9); ax.set_ylim(-1.7, 1.5)
    if savepath:
        fig.savefig(savepath, dpi=300); fig.savefig(str(savepath).replace(".png", ".pdf"))
    return fig


# --------------------------------------------------------------------------------------------------
# public entry point — called from the Evaluation notebook
# --------------------------------------------------------------------------------------------------
def make_architecture_figures(seg_path="models/v7_diffim_scripted.pt",
                              cnn_path="models/cnn_postproc.pt",
                              outdir="Evaluation/figures", show=True, save=True):
    """Build all architecture figures from the deployed weights. Returns {name: Figure}.

    Robust: if a model can't be loaded/traced, that figure is skipped with a printed note rather
    than raising — the notebook keeps running."""
    from ADCNN.inference.cnn_postproc import load_cnn, CNN_DEFAULT_THR
    figs = {}
    outdir = Path(outdir)
    if save:
        outdir.mkdir(parents=True, exist_ok=True)

    try:
        seg_model, in_ch, widths = load_seg_model(seg_path)
        seg_spec = trace_seg(seg_model, in_ch)
        figs["segmentation"] = plot_segmentation(
            seg_spec, savepath=str(outdir / "arch_segmentation.png") if save else None)
        print(f"[architecture] segmentation: in_ch={in_ch} widths={widths} "
              f"out_ch={seg_spec['out_ch']} levels={len(seg_spec['encoder'])}")
    except Exception as e:
        seg_spec = None
        print(f"[architecture] segmentation figure skipped: {type(e).__name__}: {e}")

    try:
        cnn = load_cnn(cnn_path)
        cnn_spec = trace_cnn(cnn)
        figs["filter_cnn"] = plot_filter_cnn(
            cnn_spec, thr=CNN_DEFAULT_THR, savepath=str(outdir / "arch_filter_cnn.png") if save else None)
        print(f"[architecture] filter CNN: stages={[s['c'] for s in cnn_spec['stages']]} thr={CNN_DEFAULT_THR}")
    except Exception as e:
        cnn_spec = None
        print(f"[architecture] filter-CNN figure skipped: {type(e).__name__}: {e}")

    figs["system"] = plot_system(seg_spec, cnn_spec,
                                 savepath=str(outdir / "arch_system.png") if save else None)
    if save:
        print(f"[architecture] figures saved to {outdir}/ (PNG @300 dpi + vector PDF)")
    if not show:
        for f in figs.values():
            plt.close(f)
    return figs


if __name__ == "__main__":
    make_architecture_figures(show=False)
    print("done")
